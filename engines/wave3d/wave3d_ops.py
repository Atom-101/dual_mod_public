"""Explicit layer-batched math for the wave3d engines (replaces vmap(functional_call)).

Every per-layer linear becomes ONE torch.bmm over the stacked weights of the
active layers ([G, out, in], bf16 under autocast, cast once per training step);
the elementwise refine_kv chain (models/dualmod/attention.py) is written out for the
hero config (mult_res ctx, xu gate input, convex sigmoid gates, k-norm, one raw
k-mod head, entry-norm clamp) with the SAME dtype discipline autocast gives the
module code: 1-D gains/biases fp32, fp32 norm/sigmoid/clamp/rope math, bf16
GEMM I/O.

Layout: rows are free -- inputs are [G, M, d] (M = rows of one layer's window);
`rows_dims` says how M factors ((B, W) batch-major in the vmap engine, (W, B)
time-major in Wave3D-X) so rope can broadcast cos/sin over the position axis.

Deliberate deviation (documented): the three biased LoRA up-projections (w_gk.b,
w_gv.b, v_gate.up) feed sigmoids directly; the bias is added in fp32 to the
GEMM output right before the sigmoid instead of inside the GEMM epilogue
(baddbmm / addmm with a broadcast bias measured 4x slower than bmm). This skips
one bf16 rounding of the logits -- a <=0.5-ulp difference of the same order as
the batched-vs-unbatched GEMM reordering noise, covered by the validation
tolerances.
"""

import torch
import torch.nn.functional as F

EPS_NORM = 1e-5      # RMSNorm (attn_norm / mlp_norm / norm_ctx)
EPS_HEAD = 1e-6      # per-head norms (norm_kctx / clamp)


def check_cfg(cfg):
    """The explicit chain is written for the hero MRK config; refuse anything else."""
    assert cfg.attn_mode != "vanilla" and cfg.enable_key_mod and cfg.enable_value_mod
    assert cfg.rmsnorm_on_o and cfg.gate_input == "xu", (cfg.rmsnorm_on_o, cfg.gate_input)
    assert getattr(cfg, "ctx_mod", "linear") == "mult_res", cfg.ctx_mod
    assert cfg.gate_rank > 0 and cfg.ctx_rank > 0
    assert cfg.gate_type == "convex_sigmoid" and cfg.gate_type_v == "convex"
    assert getattr(cfg, "k_ctx_norm", "none") == "rms"
    assert getattr(cfg, "v_ctx_norm", "none") == "none"
    assert getattr(cfg, "v_norm", "none") == "none"
    assert getattr(cfg, "kmod_mode", "raw") == "raw"
    assert getattr(cfg, "kmod_vraw_bound", "none") == "none"
    assert cfg.n_kv_heads_eff == cfg.n_heads


def act_dtype():
    """bf16 under cuda autocast (what nn.Linear would emit), else None (= keep)."""
    if torch.is_autocast_enabled("cuda"):
        return torch.get_autocast_dtype("cuda")
    return None


def lin(x, w):
    """x [G, M, in] @ w[G, out, in]^T -> [G, M, out]. Under autocast bmm casts
    its inputs to bf16 exactly like nn.Linear does."""
    return torch.bmm(x, w.transpose(1, 2))


def rms(x, w, eps=EPS_NORM):
    """RMSNorm with per-layer gain w [G, d]; fp32 math, result in x.dtype (module-identical)."""
    xf = x.float()
    out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (out * w.float().unsqueeze(1)).to(x.dtype)


def head_rms(x, gain, h, dh):
    """norm_kctx: per-head RMS (fp32, eps 1e-6) * per-(layer, head) gain."""
    shp = x.shape
    xh = x.reshape(*shp[:-1], h, dh).float()
    xh = xh * torch.rsqrt(xh.pow(2).mean(-1, keepdim=True) + EPS_HEAD)
    xh = xh * gain.float().view(gain.shape[0], *([1] * (x.dim() - 2)), h, 1)
    return xh.to(x.dtype).reshape(shp)


def clamp_vpp(v, tau, h, dh):
    """Entry-norm clamp: per (row, head) L2 radius <= tau (fp32 norm)."""
    shp = v.shape
    vh = v.reshape(*shp[:-1], h, dh).float()
    n = vh.norm(dim=-1, keepdim=True)
    scale = (tau / n.clamp_min(1e-20)).clamp(max=1.0)
    return (vh * scale).to(v.dtype).reshape(shp)


def rope(x, cos, sin):
    """x [..., dh], cos/sin broadcastable to x[..., :dh//2]; fp32 math -> x.dtype
    (== dualmod.rope.apply_rope)."""
    xf = x.float()
    x1, x2 = xf.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(x.dtype)


def sig_gate(logits, bias, out_dtype):
    """sigmoid((logits + bias).float()) -> out_dtype. bias [G, d] fp32 (see module doc)."""
    return torch.sigmoid(logits.float() + bias.float().unsqueeze(1)).to(out_dtype)


# ----------------------------------------------------------------------------- merged weights
UPS5 = ("attn.w_kctx.up.weight", "attn.w_gk.b.weight", "attn.w_gv.b.weight",
        "attn.v_gate.up.weight", "attn.v_map.up.weight")
GA2 = ("attn.w_gk.a.weight", "attn.w_gv.a.weight")           # N-concat: g_in -> 288
UD2 = ("attn.w_kctx.down.weight", "attn.v_map.down.weight")  # N-concat: u -> 288
MERGED = {"attn.ga2": GA2, "attn.ud2": UD2, "attn.ups5": UPS5}


def combine_params(P):
    """Add the merged refine weights to a stacked-param dict (stack dim 0 = layers):
    ga2 [G, 288, 2d], ud2 [G, 288, d] (N-concat, same input), ups5 [G, 5, d, 144]
    (5 up-projections with different inputs -> one bmm of batch 5G). 12 LoRA GEMM
    launches per group become 6; the CPU launch cost, not FLOPs, was the limiter."""
    if "attn.ups5" in P:
        return P
    P = dict(P)
    P["attn.ga2"] = torch.cat([P[GA2[0]], P[GA2[1]]], dim=1)
    P["attn.ud2"] = torch.cat([P[UD2[0]], P[UD2[1]]], dim=1)
    P["attn.ups5"] = torch.stack([P[n] for n in UPS5], dim=1)
    return P


# ----------------------------------------------------------------------------- bodies
def prep(P, x, cos, sin, h, dh, rows_dims):
    """Block prep for the NEW chunk. x [G, M, d] fp32 residual rows, M = prod(rows_dims);
    cos/sin broadcast over [G, *rows_dims, h, dh//2]. Returns (xhat_fp32, xh_act,
    k_flat, v_flat, q_r, k_r): flats [G, M, d], q_r/k_r rotated [G, *rows_dims, h, dh]
    (v_flat.view(G, *rows_dims, h, dh) is the raw V)."""
    G, M, d = x.shape
    xhat = rms(x, P["attn_norm.weight"])                  # fp32 (x is fp32)
    dt = act_dtype()
    xb = xhat if dt is None else xhat.to(dt)             # the cast nn.Linear would do
    q = lin(xb, P["attn.wq.weight"])
    k_flat = lin(xb, P["attn.wk.weight"])
    v_flat = lin(xb, P["attn.wv.weight"])                # v_norm == none
    q_r = rope(q.view(G, *rows_dims, h, dh), cos, sin)
    k_r = rope(k_flat.view(G, *rows_dims, h, dh), cos, sin)
    return xhat, xb, k_flat, v_flat, q_r, k_r


def refine(P, xb, o, kf, vf, cos, sin, h, dh, dcut, tau, lam, rows_dims):
    """refine_kv (mult_res / xu / convex gates / k-norm / raw k-mod head / clamp)
    + rope-at-write. xb: xhat (cast to the activation dtype here if needed); o:
    merged heads [G, M, d]; kf/vf: raw k/v flat. cos/sin broadcast over
    [G, *rows_dims, h, dh//2]. Returns (K_ref, V_ref) [G, *rows_dims, h, dh]."""
    G, M, d = o.shape
    P = combine_params(P)
    u = rms(o, P["attn.norm_ctx.weight"])
    if xb.dtype != u.dtype:
        xb = xb.to(u.dtype)
    g_in = torch.cat([xb, u], dim=-1)
    r = P["attn.ud2"].shape[1] // 2
    # LoRA downs: 3 GEMMs (g_in -> [gk_d | gv_d], u -> [k_d | vm_d], xb -> vg_d)
    gd = lin(g_in, P["attn.ga2"])
    ud = lin(u, P["attn.ud2"])
    vg_d = lin(xb, P["attn.v_gate.down.weight"])
    # 5 ups in ONE bmm: batch (G, 5) of [M, r] x [r, d]
    down5 = torch.stack([ud[..., :r], gd[..., :r], gd[..., r:], vg_d, ud[..., r:]], dim=1)
    up5 = torch.bmm(down5.reshape(G * 5, M, r),
                    P["attn.ups5"].reshape(G * 5, d, r).transpose(1, 2)).view(G, 5, M, d)
    k_ctx, gk, gv, vg, vm = (up5[:, i] for i in range(5))
    # keys
    k_ctx = head_rms(k_ctx, P["attn.k_gain"], h, dh)
    g = sig_gate(gk, P["attn.w_gk.b.bias"], kf.dtype)
    kpp = g * kf + (1.0 - g) * k_ctx
    # values: mult_res ctx transform
    v_in = lam * u + sig_gate(vg, P["attn.v_gate.up.bias"], u.dtype) * torch.tanh(vm)
    v_ctx = lin(lin(v_in, P["attn.w_vctx.down.weight"]), P["attn.w_vctx.up.weight"])
    g2 = sig_gate(gv, P["attn.w_gv.b.bias"], vf.dtype)
    vpp = g2 * vf + (1.0 - g2) * v_ctx
    if dcut > 0:                                          # k-mod-only heads keep RAW values
        vpp = torch.cat([vf[..., :dcut], vpp[..., dcut:]], dim=-1)
    if tau > 0:
        vpp = clamp_vpp(vpp, tau, h, dh)
    K_ref = rope(kpp.view(G, *rows_dims, h, dh), cos, sin)
    return K_ref, vpp.view(G, *rows_dims, h, dh)


def prep_a(P, xs):
    """prep piece 1 (Wave3DX deferred-dW path): stack of the G chunk residuals
    [C, B, d] -> [G, C*B, d] (fused into the norm's read), attn_norm + cast."""
    x = torch.stack(xs)
    x = x.reshape(x.shape[0], -1, x.shape[-1])     # reshape: a 1-chunk stack of a strided view is not viewable
    xhat = rms(x, P["attn_norm.weight"])
    dt = act_dtype()
    return xhat, (xhat if dt is None else xhat.to(dt))


def prep_b(q, k_flat, cos, sin, h, dh, rows_dims):
    """prep piece 2: rope of the (externally computed) q / k projections."""
    G = q.shape[0]
    return (rope(q.view(G, *rows_dims, h, dh), cos, sin),
            rope(k_flat.view(G, *rows_dims, h, dh), cos, sin))


def tail_a(P, xs, h_wo):
    """tail piece 1: x1 = stack(x chunks) + wo(o) (h_wo = the wo GEMM output), mlp_norm + cast."""
    x = torch.stack(xs)
    x1 = x.reshape(x.shape[0], -1, x.shape[-1]) + h_wo
    hn = rms(x1, P["mlp_norm.weight"])
    dt = act_dtype()
    return x1, (hn if dt is None else hn.to(dt))


def tail_b(a1, a3):
    """tail piece 2: swiglu gate of the w1 / w3 GEMM outputs."""
    return F.silu(a1) * a3


def tail(P, x, o_c):
    """x + wo(o) + mlp(mlp_norm(.)). x [G, M, d] fp32, o_c [G, M, d] activation dtype."""
    x = x + lin(o_c, P["attn.wo.weight"])
    hn = rms(x, P["mlp_norm.weight"])
    dt = act_dtype()
    hb = hn if dt is None else hn.to(dt)
    y = F.silu(lin(hb, P["mlp.w1.weight"])) * lin(hb, P["mlp.w3.weight"])
    return x + lin(y, P["mlp.w2.weight"])


def refine_tail(Pg, xh_ws, o_lists, kf_ws, vf_ws, css, sns, rows_list, Pt, tail_x, tail_os,
                h, dh, dcut, tau, lam):
    """ONE compiled region for a whole step's post-attention math: the refine of
    every window group (each o_lists[g] is the list of per-layer time-major o
    rows, stacked here so the stack fuses with its consumers) plus the tail of
    the committing layers (Pt None => no tail this step). Static shapes per
    step pattern => one inductor graph per (K, t mod K, ramp phase)."""
    outs = []
    for P, xh, os, kf, vf, cs, sn, rows in zip(Pg, xh_ws, o_lists, kf_ws, vf_ws, css, sns, rows_list):
        o = torch.stack(os) if len(os) > 1 else os[0].unsqueeze(0)
        outs.append(refine(P, xh, o, kf, vf, cs, sn, h, dh, dcut, tau, lam, rows))
    xn = None
    if Pt is not None:
        xn = tail(Pt, tail_x, torch.stack(tail_os))
    return outs, xn
