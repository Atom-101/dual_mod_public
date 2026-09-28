"""WaveScan forward scan-cell (plans/megakernel.md §4), chunked jacobi form.

One code path serves both modes:
  - jacobi(K) on a chunk of c positions: sweep 0 reads committed cache + RAW
    intra-chunk entries; sweep s reads committed + sweep-(s-1) intra rows.
    The diagonal (self) entry is ALWAYS the raw k,v — appended as an extra
    softmax column so it never mixes with the refined intra rows.
  - exact: c=1 chunks (the intra block is empty, so one sweep reproduces the
    eager per-position step bit-for-bit), or equivalently jacobi(K=c) on any
    chunk (nilpotency: after s sweeps, intra positions < s are exact — T-K3).

The cell is expressed in torch ops (cuBLAS GEMMs, explicit fp32 softmax) so it
is CUDA-graph-capturable, autograd-recomputable in the backward cell, and
bit-comparable to the eager oracle. Gate/blend math reuses the repo's
`DualModAttention.refine_kv` verbatim — the semantic contract lives in one
place. Numerics: fp32 softmax + fp32 gate sigmoid regardless of storage dtype
(§4 Numerics); an all-fp32 run needs no special flag, just fp32 inputs.
"""

import torch
import torch.nn.functional as F

from models.dualmod.rope import apply_rope

# ctx_mod variants the PACKED diff cell reproduces exactly (all others fall
# back to the verbatim refine_kv cell — graphs.py routes on this set).
# kmod_vraw_heads is implemented as a +30 pin on the head-0..N v-gate logits:
# sigmoid(30) = 1 - 9e-14, so v'' == v_raw to fp32 rounding and the pinned
# logits carry zero gradient — identical to the eager splice (dead gate rows).
PACKED_CTX_MODS = ("linear", "mult", "mult_res", "fusion")
_GATE_PIN = 30.0


def _ctx_variant(attn, xhat_flat, u_flat, mod):
    """[N,2d] k_ctx | v_ctx for the supported ctx_mod set (mirrors refine_kv
    exactly, incl. fp32 gate sigmoid)."""
    k_ctx = attn.w_kctx(u_flat)
    if mod == "mult":
        v_in = torch.sigmoid(attn.v_gate(xhat_flat).float()).to(u_flat.dtype) \
            * torch.tanh(attn.v_map(u_flat))
    elif mod == "mult_res":
        v_in = u_flat + torch.sigmoid(
            attn.v_gate(xhat_flat).float()).to(u_flat.dtype) \
            * torch.tanh(attn.v_map(u_flat))
    else:  # fusion
        v_in = u_flat + F.silu(
            attn.v_pre(torch.cat([xhat_flat, u_flat], dim=-1)))
    # v_ctx_norm right after w_vctx (norm_vctx is a no-op for "none")
    return torch.cat([k_ctx, attn.norm_vctx(attn.w_vctx(v_in))], dim=-1)


def _neg_inf(dtype):
    return torch.finfo(dtype).min


def make_intra_mask(c: int, device, dtype=torch.float32):
    """[c, c] additive mask blocking intra columns s >= row t (self is handled
    by the appended raw column, so the diagonal is blocked here too)."""
    m = torch.full((c, c), _neg_inf(torch.float32), device=device, dtype=dtype)
    return m.triu(0)  # upper incl. diagonal = -inf, strictly-lower = 0


def chunk_attend(q_r, k_self_r, v_raw, K_comm, V_comm, K_intra, V_intra,
                 intra_mask, scale):
    """Attention read for all c positions of a chunk against committed cache +
    intra rows + raw self column. All [B,h,*,d_h]; returns o [B,h,c,d_h].

    q_r, k_self_r rotated; K_comm/K_intra rotated refined rows; V_* refined
    values; v_raw the raw self values. intra_mask [c,c] additive fp32.
    """
    s_intra = q_r @ K_intra.transpose(-1, -2) * scale            # [B,h,c,c]
    s_intra = s_intra.float() + intra_mask
    s_self = (q_r * k_self_r).sum(-1, keepdim=True).float() * scale  # [B,h,c,1]
    if K_comm is not None and K_comm.shape[2] > 0:
        s_comm = (q_r @ K_comm.transpose(-1, -2) * scale).float()    # [B,h,c,a]
        scores = torch.cat([s_comm, s_intra, s_self], dim=-1)
    else:
        scores = torch.cat([s_intra, s_self], dim=-1)
    p = torch.softmax(scores, dim=-1).to(v_raw.dtype)
    p_rows, p_self = p[..., :-1], p[..., -1:]
    if V_comm is not None and V_comm.shape[2] > 0:
        v_all = torch.cat([V_comm, V_intra], dim=2)
    else:
        v_all = V_intra
    return p_rows @ v_all + p_self * v_raw


def chunk_cell_forward(attn, xhat_c, cos_c, sin_c, K_comm, V_comm, n_sweeps,
                       intra_mask=None):
    """Full forward cell for one (layer, chunk): raw projections -> K sweeps ->
    (o_cat [B,c,d], K_rows [B,h,c,d_h] rotated, V_rows [B,h,c,d_h]).

    attn: the repo DualModAttention module (weights + refine_kv semantics).
    xhat_c: pre-normed chunk input [B,c,d]. cos_c/sin_c: rope rows for the
    chunk's absolute positions. K_comm/V_comm: committed cache views
    [B,h,a,d_h] (None or 0-length for the first chunk).
    """
    c = xhat_c.shape[1]
    if intra_mask is None:
        intra_mask = make_intra_mask(c, xhat_c.device)
    q_r = apply_rope(attn._split(attn.wq(xhat_c)), cos_c, sin_c)
    k_flat = attn.wk(xhat_c)
    v_flat = attn.project_v(xhat_c)               # v_norm applied at source
    k_self_r = apply_rope(attn._split(k_flat), cos_c, sin_c)
    v_raw = attn._split(v_flat)

    K_intra, V_intra = k_self_r, v_raw            # sweep 0 reads raw intra rows
    o_cat = None
    for _ in range(n_sweeps):
        o = chunk_attend(q_r, k_self_r, v_raw, K_comm, V_comm,
                         K_intra, V_intra, intra_mask, attn.scale)
        o_cat = attn._merge(o)
        kpp, vpp = attn.refine_kv(xhat_c, o_cat, k_flat, v_flat)
        K_intra = apply_rope(attn._split(kpp), cos_c, sin_c)
        V_intra = attn._split(vpp)
    return o_cat, K_intra, V_intra


def _merge_setup(q_r, K_comm, V_comm, scale, out_dtype):
    """Sweep-invariant part of the attention read (flash/online-softmax
    algebra): unnormalized committed accumulator A_comm, normalizer Z_comm,
    row max m_comm. Computed ONCE per cell; each sweep merges its (tiny)
    intra+self block against these instead of re-running the a-sized GEMMs."""
    s_comm = (q_r @ K_comm.transpose(-1, -2)).float() * scale     # [B,h,c,a]
    m_comm = s_comm.amax(dim=-1, keepdim=True)
    e_comm = torch.exp(s_comm - m_comm)
    Z_comm = e_comm.sum(dim=-1, keepdim=True)
    A_comm = (e_comm.to(out_dtype) @ V_comm).float()              # [B,h,c,dh]
    return A_comm, Z_comm, m_comm


def _merge_attend(s_tail, A_comm, Z_comm, m_comm, V_intra, v_raw, out_dtype):
    """One sweep's read: merge the fp32 tail scores [B,h,c,c+1] (intra block +
    raw-self column) with the committed accumulator. Exact softmax result up
    to fp rounding."""
    m_tail = s_tail.amax(dim=-1, keepdim=True)
    if m_comm is not None:
        m = torch.maximum(m_tail, m_comm)
        w = torch.exp(m_comm - m)
        e_tail = torch.exp(s_tail - m)
        Z = Z_comm * w + e_tail.sum(dim=-1, keepdim=True)
        o = A_comm * w + (e_tail[..., :-1].to(out_dtype) @ V_intra).float() \
            + e_tail[..., -1:] * v_raw.float()
    else:
        e_tail = torch.exp(s_tail - m_tail)
        Z = e_tail.sum(dim=-1, keepdim=True)
        o = (e_tail[..., :-1].to(out_dtype) @ V_intra).float() \
            + e_tail[..., -1:] * v_raw.float()
    return (o / Z).to(out_dtype)


def chunk_cell_forward_inplace(attn, xhat_c, cos_c, sin_c, K_cache, V_cache,
                               pos0, n_sweeps, tail_mask, fused=None):
    """Launch-count-optimized forward cell (no_grad / graph-capture path).

    Semantics identical to chunk_cell_forward; the intra-chunk double buffer
    lives IN the cache rows [pos0, pos0+c) (scratch during the cell's own
    sweeps, final values on exit — append-only for every other cell), so the
    attention read is ONE contiguous GEMM over cache[:, :, :pos0+c] instead of
    cat(committed, intra) per sweep. `tail_mask` is the [c, c] additive mask
    for the trailing intra block. `fused` (module ref) enables the Triton
    fused sigmoid+blend+rope+write kernel and the packed ctx/gate GEMMs.
    """
    B, c, _ = xhat_c.shape
    q_r = apply_rope(attn._split(attn.wq(xhat_c)), cos_c, sin_c)
    k_flat = attn.wk(xhat_c)
    v_flat = attn.project_v(xhat_c)               # v_norm applied at source
    k_self_r = apply_rope(attn._split(k_flat), cos_c, sin_c)
    v_raw = attn._split(v_flat)
    K_cache[:, :, pos0:pos0 + c] = k_self_r.to(K_cache.dtype)
    V_cache[:, :, pos0:pos0 + c] = v_raw.to(V_cache.dtype)
    K_intra = K_cache[:, :, pos0:pos0 + c]
    V_intra = V_cache[:, :, pos0:pos0 + c]
    dt = v_raw.dtype
    if pos0 > 0:
        A_comm, Z_comm, m_comm = _merge_setup(
            q_r, K_cache[:, :, :pos0], V_cache[:, :, :pos0], attn.scale, dt)
    else:
        A_comm = Z_comm = m_comm = None
    s_self = (q_r * k_self_r).sum(-1, keepdim=True).float() * attn.scale
    o_cat = None
    for _ in range(n_sweeps):
        s_tail = torch.cat(
            [(q_r @ K_intra.transpose(-1, -2)).float() * attn.scale + tail_mask,
             s_self], dim=-1)
        o = _merge_attend(s_tail, A_comm, Z_comm, m_comm, V_intra, v_raw, dt)
        o_cat = attn._merge(o)
        if fused is not None:
            fused.refine_write(attn, xhat_c, o_cat, k_flat, v_flat,
                               cos_c, sin_c, K_cache, V_cache, pos0)
        else:
            kpp, vpp = attn.refine_kv(xhat_c, o_cat, k_flat, v_flat)
            K_cache[:, :, pos0:pos0 + c] = apply_rope(
                attn._split(kpp), cos_c, sin_c).to(K_cache.dtype)
            V_cache[:, :, pos0:pos0 + c] = attn._split(vpp).to(V_cache.dtype)
    return o_cat


def chunk_cell_forward_diff(attn, xhat_c, cos_c, sin_c, K_comm, V_comm,
                            n_sweeps, intra_mask, use_triton_refine=True):
    """Differentiable fast cell for the BACKWARD recompute: same math as
    chunk_cell_forward, but the committed-prefix scores are hoisted out of the
    sweep loop, the ctx/gate projections run as two packed GEMMs (weights
    concatenated differentiably once per cell, so param grads flow), and the
    pointwise refine chain is ONE FusedRefineFn autograd node instead of ~25
    elementwise ops (use_triton_refine=False leaves the chain in plain torch
    for torch.compile/inductor to fuse instead)."""
    from .triton_kernels import FusedRefineFn
    B, c, d = xhat_c.shape
    h, dh = attn.n_heads, attn.head_dim
    N = B * c
    q_r = apply_rope(attn._split(attn.wq(xhat_c)), cos_c, sin_c)
    k_flat = attn.wk(xhat_c)
    v_flat = attn.project_v(xhat_c)               # v_norm applied at source
    k_self_r = apply_rope(attn._split(k_flat), cos_c, sin_c)
    v_raw = attn._split(v_flat)
    W_ctx = torch.cat([attn.w_kctx.weight, attn.w_vctx.weight], dim=0)
    W_g = torch.cat([attn.w_gk.weight, attn.w_gv.weight], dim=0)
    b_g = torch.cat([attn.w_gk.bias, attn.w_gv.bias], dim=0)

    mod = getattr(attn.cfg, "ctx_mod", "linear")
    nvr = getattr(attn.cfg, "kmod_vraw_heads", 0)
    kmode = getattr(attn.cfg, "kmod_mode", "raw")
    pin_mask = zero_mask = None
    if nvr > 0 and kmode == "raw":
        pin_mask = torch.zeros(1, 2 * d, dtype=torch.bool, device=xhat_c.device)
        pin_mask[0, d:d + nvr * dh] = True
    elif nvr > 0:
        # gated_norm: gate logits stay LIVE; the head's v_ctx slice is
        # zeroed so the convex blend degenerates to the write gate exactly
        # (masked_fill is differentiable -> those w_vctx rows read as dead,
        # matching the eager cat-zero splice)
        zero_mask = torch.zeros(1, 2 * d, dtype=torch.bool,
                                device=xhat_c.device)
        zero_mask[0, d:d + nvr * dh] = True

    a = K_comm.shape[2]
    dt = v_raw.dtype
    s_self = (q_r * k_self_r).sum(-1, keepdim=True).float() * attn.scale
    if a > 0:
        A_comm, Z_comm, m_comm = _merge_setup(q_r, K_comm, V_comm,
                                              attn.scale, dt)
    else:
        A_comm = Z_comm = m_comm = None
    K_intra, V_intra = k_self_r, v_raw
    o_cat = None
    for _ in range(n_sweeps):
        s_tail = torch.cat(
            [(q_r @ K_intra.transpose(-1, -2)).float() * attn.scale
             + intra_mask, s_self], dim=-1)
        o = _merge_attend(s_tail, A_comm, Z_comm, m_comm, V_intra, v_raw, dt)
        o_cat = attn._merge(o)
        u = attn.norm_ctx(o_cat)
        if mod == "linear":
            ctx = u.reshape(N, d) @ W_ctx.transpose(0, 1)
            if getattr(attn.cfg, "v_ctx_norm", "none") == "rms":
                # ctxg v-block: per-head norm AFTER the packed GEMM; the
                # gated_norm kmod zero (below) comes LAST, matching the
                # eager order (refine_kv norms w_vctx's output, THEN
                # cat-zeroes the head slice)
                ctx = torch.cat([ctx[:, :d], attn.norm_vctx(ctx[:, d:])],
                                dim=-1)
        else:
            ctx = _ctx_variant(attn, xhat_c.reshape(N, d), u.reshape(N, d),
                               mod)
        if zero_mask is not None:
            # gated_norm kmod: no ctx term for the head -> pure write gate
            ctx = ctx.masked_fill(zero_mask, 0.0)
        glog = torch.cat([xhat_c, u], dim=-1).reshape(N, 2 * d) \
            @ W_g.transpose(0, 1) + b_g
        if pin_mask is not None:
            # kmod heads: v-gate logits pinned -> g=1 -> raw v committed;
            # pinned entries carry zero grad (matches the eager splice)
            glog = torch.where(pin_mask, _GATE_PIN, glog)
        if use_triton_refine:
            K_intra, V_intra = FusedRefineFn.apply(
                k_flat, v_flat, ctx, glog, cos_c, sin_c, B, c, h, dh)
        else:
            g = torch.sigmoid(glog.float()).to(k_flat.dtype).reshape(B, c, 2 * d)
            ctx_r = ctx.reshape(B, c, 2 * d)
            kpp = g[..., :d] * k_flat + (1 - g[..., :d]) * ctx_r[..., :d]
            vpp = g[..., d:] * v_flat + (1 - g[..., d:]) * ctx_r[..., d:]
            K_intra = apply_rope(attn._split(kpp), cos_c, sin_c)
            V_intra = attn._split(vpp)
    return o_cat, K_intra, V_intra
