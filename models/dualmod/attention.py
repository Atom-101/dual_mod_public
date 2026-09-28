"""DualModAttention (plan.md §3).

Three modes:
  - "vanilla":    standard causal attention via SDPA (fast baseline path).
  - "sequential": exact per-position scan; refined k'', v'' written to the cache
                  (training path in v0). Lives in scan.py, driven from here.
  - "deq":        K-sweep parallel approximation stub (plan.md §6, phase 2 / T4 only).

The per-position semantics used by BOTH teacher-forced training and incremental
decoding are `attend_one` + `refine_kv` (+ rotate-at-write). That is the single
code path required by plan.md goal 3 / test T3.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DualModConfig
from .rope import apply_rope


class LoRAGate(nn.Module):
    """Factored gate projection: in -> r -> out. .bias/.weight alias the
    down-proj so init/telemetry code paths keep working."""

    def __init__(self, din, r, dout):
        super().__init__()
        self.a = nn.Linear(din, r, bias=False)
        self.b = nn.Linear(r, dout, bias=True)

    def forward(self, x):
        return self.b(self.a(x))

    @property
    def bias(self):
        return self.b.bias

    @property
    def weight(self):
        return self.b.weight


class LoRALinear(nn.Module):
    """Factored din->r->dout projection whose `.weight` property MATERIALIZES
    the full [dout, din] matrix (up @ down) so the fused engine's K-sweep
    kernels + telemetry that read a dense d×d weight keep working unchanged.
    Eager forward is up(down(x)) (+bias) so autograd trains the factors; the
    engine's manual backward projects the composed dW onto the factors. Bias
    is optional (v_gate carries one). This is the ctx-operator twin of
    LoRAGate — the difference is `.weight` here is the full product (the
    kernels want a dense matrix), whereas LoRAGate.weight aliases the up-proj
    (the engine materializes b@a itself for the packed gate GEMMs)."""

    def __init__(self, din, r, dout, bias=False):
        super().__init__()
        self.down = nn.Linear(din, r, bias=False)
        self.up = nn.Linear(r, dout, bias=bias)

    def forward(self, x):
        return self.up(self.down(x))

    @property
    def weight(self):
        # full [dout, din] = up.weight[dout,r] @ down.weight[r,din]
        return self.up.weight @ self.down.weight

    @property
    def bias(self):
        return self.up.bias


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (out * self.weight.float()).to(x.dtype)


class DualModAttention(nn.Module):
    def __init__(self, cfg: DualModConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.n_heads = cfg.n_heads          # query heads (== heads used by mechanism paths)
        self.n_kv_heads = cfg.n_kv_heads_eff
        self.head_dim = cfg.head_dim
        self.d_attn = cfg.d_attn            # n_heads * head_dim (== d for default MHA)
        d_kv = self.n_kv_heads * self.head_dim
        self.scale = 1.0 / math.sqrt(cfg.head_dim)

        # For the default config (MHA, head_dim=d/h) these are all d×d, exactly as before.
        # F1/F2 widen the query/output projections at fixed head_dim (vanilla path only).
        self.wq = nn.Linear(d, self.d_attn, bias=False)
        self.wk = nn.Linear(d, d_kv, bias=False)
        self.wv = nn.Linear(d, d_kv, bias=False)
        self.wo = nn.Linear(self.d_attn, d, bias=False)

        # v_norm gain (see project_v). Constraint rationale: unbounded
        # per-token value amplitude is a softmax-EXEMPT salience channel — a
        # token can flood downstream reads by writing a large ||v|| no matter
        # what the attention weights or the gates say (the flooding
        # pathology). RMS-norming v_raw moves amplitude coding into the
        # bounded gates; the learned per-head scalar gain keeps one loudness
        # dof per head. 1-D param by design: lands in the no-decay optimizer
        # group (train_flagship doctrine), correct for a gain.
        if getattr(cfg, "v_norm", "none") == "rms":
            self.v_gain = nn.Parameter(torch.ones(self.n_kv_heads))

        if cfg.attn_mode != "vanilla":
            # context projections (full d×d: cross-head mixing is intentional, §3.7).
            # Only the enabled paths get parameters, so key-only / value-only runs
            # carry no unused params (clean DDP without find_unused_parameters).
            # keys ALWAYS convex (legacy gate_type); the value gate has its own
            # convexity switch (gate_type_v). independent-v widens w_gv to 2d.
            gate_out_k = 2 * d if cfg.gate_type == "independent" else d
            gate_out_v = 2 * d if getattr(cfg, "gate_type_v", "convex") \
                == "independent" else d
            def _ctx_mat(din, dout, bias=False):
                # ctx_rank>0: LoRA-factor (din->r->dout); the fused engine reads
                # a materialized dout×din `.weight`. Off => dense nn.Linear
                # (byte-identical to the pre-ctx_rank code).
                if cfg.ctx_rank > 0:
                    return LoRALinear(din, cfg.ctx_rank, dout, bias=bias)
                return nn.Linear(din, dout, bias=bias)

            if cfg.enable_key_mod:
                self.w_kctx = _ctx_mat(d, d)
                if cfg.gate_rank > 0:
                    self.w_gk = LoRAGate(cfg.gate_in_dim, cfg.gate_rank,
                                         gate_out_k)
                else:
                    self.w_gk = nn.Linear(cfg.gate_in_dim, gate_out_k,
                                          bias=True)
                # K-norm gain (see norm_kctx): per-head gain on the ctx KEY
                # candidate, init at the survivors' equilibrium k_ctx RMS so
                # step-0 read-sharpness lands on the healthy attractor. 1-D ->
                # no-decay group (v_gain/ctx_gain doctrine).
                if getattr(cfg, "k_ctx_norm", "none") == "rms":
                    self.k_gain = nn.Parameter(
                        torch.full((self.n_heads,),
                                   float(getattr(cfg, "k_gain_init", 1.75))))
            if cfg.enable_value_mod:
                self.w_vctx = _ctx_mat(d, d)
                if cfg.gate_rank > 0:
                    self.w_gv = LoRAGate(cfg.gate_in_dim, cfg.gate_rank,
                                         gate_out_v)
                else:
                    self.w_gv = nn.Linear(cfg.gate_in_dim, gate_out_v,
                                          bias=True)
                # v_ctx_norm gain (see norm_vctx). Rationale: bounds the
                # write amplitude of the ctx channel (fusion's candidate is
                # unbounded — identity+silu); direction preserved (unlike
                # tanh) so depth-seeking dynamics survive; with v_norm +
                # independent gates this completes the symmetric write law
                #   v'' = g_r*N(v_raw) + g_c*N(v_ctx).
                # 1-D param by design -> no-decay optimizer group (same
                # doctrine as v_gain).
                if getattr(cfg, "v_ctx_norm", "none") == "rms":
                    self.ctx_gain = nn.Parameter(torch.ones(self.n_heads))
            # ctx_mod arms (LSTM-gap): pre-MLP feeding the (kept) output
            # projection — w_kctx/w_vctx remain the small-init output layers
            # so §3.1 init and telemetry are untouched. linear = no-op.
            cm = getattr(cfg, "ctx_mod", "linear")
            if cm == "vmlp":
                self.v_pre = nn.Linear(d, d, bias=False)
            elif cm in ("fusion", "puremlp", "puremlp_hw"):
                self.v_pre = nn.Linear(2 * d, d, bias=False)
            elif cm == "kmlp":
                self.k_pre = nn.Linear(d, d, bias=False)
            elif cm in ("mult", "mult_res"):
                # LSTM-decomposition unit: input-programmed diagonal gate
                # over a dense nonlinear state map (non-commuting update).
                # mult_res = residual form (identity at zero-H init) for
                # STAGED growth on formed models
                self.v_gate = _ctx_mat(d, d, bias=True)
                self.v_map = _ctx_mat(d, d)
            elif cm == "linpoint":
                # nonlinear parameterization born at the EXACT linear
                # transport: silu(x)-silu(-x)=x, so [I;-I] -> silu -> [I,-I]
                # computes identity through a nonlinearity; gradients are
                # free to reshape from there (webui experiment #3)
                self.v_pre = nn.Linear(d, 2 * d, bias=False)
                self.v_comb = nn.Linear(2 * d, d, bias=False)
            elif cm == "glu":
                # SwiGLU-shaped ctx transform: multiplication present, both
                # operands input-dependent (the trainable-mult middle rung)
                self.v_gate = nn.Linear(2 * d, d, bias=False)
                self.v_map = nn.Linear(2 * d, d, bias=False)
            # zero-init pre layers: residual branch is EXACT identity at
            # init (silu(0)=0; grad flows via silu'(0)=0.5)
            for pre in ("v_pre", "k_pre"):
                if hasattr(self, pre):
                    nn.init.zeros_(getattr(self, pre).weight)
            if cfg.rmsnorm_on_o and (cfg.enable_key_mod or cfg.enable_value_mod):
                self.norm_ctx = RMSNorm(d)

    # ---- head bookkeeping -------------------------------------------------

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T, d] -> [B, h, T, d_h]"""
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        """[B, h, T, d_h] -> [B, T, d]"""
        B, h, T, dh = x.shape
        return x.transpose(1, 2).reshape(B, T, h * dh)

    # ---- value projection (the ONE site for v_norm) ------------------------

    def project_v(self, xhat):
        """Value projection -> FLAT [.., d_kv] v_raw. Every path (eager scan,
        deq/static, decode, wavescan cells, fused engines) must form v through
        this method so v_norm semantics stay engine-identical.

        v_norm="rms": per-head RMS over head_dim in fp32 (eps 1e-6, like
        norm_ctx) with a learned per-head gain. Applies BEFORE any
        kmod_vraw_heads splice — the verbatim channel becomes unit-scale.
        v_norm="none": byte-identical to wv(xhat)."""
        v = self.wv(xhat)
        if getattr(self.cfg, "v_norm", "none") != "rms":
            return v
        shp = v.shape
        vh = v.view(*shp[:-1], self.n_kv_heads, self.head_dim).float()
        vh = vh * torch.rsqrt(vh.pow(2).mean(-1, keepdim=True) + 1e-6)
        vh = vh * self.v_gain.float().unsqueeze(-1)
        return vh.to(v.dtype).view(shp)

    # ---- ctx value candidate norm (the ONE site for v_ctx_norm) ------------

    def norm_vctx(self, v_ctx):
        """Per-head RMS norm of the ctx value candidate v_ctx (the OUTPUT of
        w_vctx, for every ctx_mod) — fp32, eps 1e-6, learned per-head gain
        ctx_gain — exactly mirroring project_v's treatment of v_raw. Every
        path that forms v_ctx (refine_kv, packed diff cell, FusedRefine,
        FusedSweepCell) must norm through this method so semantics stay
        engine-identical. v_ctx_norm="none": returns the input unchanged.

        Applied right after w_vctx, BEFORE the gated_norm kmod zeroing (the
        zero must come last). reshape (not view): callers may pass
        non-contiguous packed-GEMM slices."""
        if getattr(self.cfg, "v_ctx_norm", "none") != "rms":
            return v_ctx
        shp = v_ctx.shape
        vh = v_ctx.reshape(*shp[:-1], self.n_heads, self.head_dim).float()
        vh = vh * torch.rsqrt(vh.pow(2).mean(-1, keepdim=True) + 1e-6)
        vh = vh * self.ctx_gain.float().unsqueeze(-1)
        return vh.to(v_ctx.dtype).reshape(shp)

    def norm_kctx(self, k_ctx):
        """K-norm: per-head RMS norm of the ctx KEY candidate k_ctx (OUTPUT of
        w_kctx), fp32 eps 1e-6, learned per-head gain k_gain — the k-side twin
        of norm_vctx. Applied right after w_kctx, BEFORE the convex blend, so
        at g=1 the raw read is untouched (vanilla FA exact). k_ctx_norm="none":
        returns the input unchanged."""
        if getattr(self.cfg, "k_ctx_norm", "none") != "rms":
            return k_ctx
        shp = k_ctx.shape
        kh = k_ctx.reshape(*shp[:-1], self.n_heads, self.head_dim).float()
        kh = kh * torch.rsqrt(kh.pow(2).mean(-1, keepdim=True) + 1e-6)
        kh = kh * self.k_gain.float().unsqueeze(-1)
        return kh.to(k_ctx.dtype).reshape(shp)

    # ---- the shared per-position step (train scan ≡ decode) ---------------

    def _logn_rows(self, s):
        """Length-adaptive logit scale for full [.., T, T] causal score matrices: row i has i+1 keys."""
        _ref = getattr(self.cfg, "attn_logn_ref", 0) or 0
        if not _ref or _ref <= 1:
            return s
        T = s.shape[-2]
        n = torch.arange(1, T + 1, device=s.device, dtype=torch.float32)
        f = (n.log() / math.log(_ref)).clamp_min(0.0).view(*([1] * (s.dim() - 2)), T, 1)
        return torch.where(torch.isinf(s), s, s * f.to(s.dtype))

    def attend_one(self, q_r, k_self_r, v_self, K_past, V_past, need_probs=False,
                   collect=None):
        """One position's attention read (§3 steps 3-4).

        q_r, k_self_r, v_self: [B, h, 1, d_h] — rotated query, RAW rotated self
        key, RAW self value. K_past/V_past: [B, h, P, d_h] refined cache (may be
        None or zero-length). fp32 softmax. Returns o [B, h, 1, d_h].
        """
        W = getattr(self.cfg, "local_window", 0) or 0
        if W and K_past is not None and K_past.shape[2] > W - 1:  # sliding window: W-1 past + self
            K_past, V_past = K_past[:, :, -(W - 1):], V_past[:, :, -(W - 1):]
        if K_past is None or K_past.shape[2] == 0:
            k_all, v_all = k_self_r, v_self
        else:
            k_all = torch.cat([K_past, k_self_r], dim=2)
            v_all = torch.cat([V_past, v_self], dim=2)
        s = (q_r @ k_all.transpose(-1, -2)) * self.scale        # [B,h,1,P+1]
        _ref = getattr(self.cfg, "attn_logn_ref", 0) or 0
        if _ref > 1 and k_all.shape[2] > 1:
            s = s * (math.log(k_all.shape[2]) / math.log(_ref))   # length-adaptive logit scale
        if collect is not None:
            # drift arm 1b: the ctx-READ pre-softmax logit max — exactly what
            # QK-norm would govern. Growth PRECEDING the chain_rho jump =>
            # read-side gain (QK-norm indicated); flat => recurrence
            # self-amplified. Per-head max over keys, kept for the p99.9.
            collect.setdefault("attn_logit_max", []).append(
                s.detach().abs().amax(dim=-1).reshape(-1).float())
        a = torch.softmax(s.float(), dim=-1)
        o = a.to(v_all.dtype) @ v_all
        return o, (a if need_probs else None)

    def refine_kv(self, xhat, o_cat, k_raw, v_raw, collect=None):
        """§3 steps 6-9: context norm, ctx projections, gates, convex blend.

        All inputs [B, t, d] (t may be 1 for a single step or T for deq sweeps).
        Returns (k'', v'') in *unrotated* key space — caller rotates at write.
        """
        cfg = self.cfg
        if cfg.attn_mode == "vanilla" or not (cfg.enable_key_mod or cfg.enable_value_mod):
            return k_raw, v_raw
        u = self.norm_ctx(o_cat) if cfg.rmsnorm_on_o else o_cat
        if cfg.gate_input == "xu":
            g_in = torch.cat([xhat, u], dim=-1)
        elif cfg.gate_input == "u":
            g_in = u
        else:
            g_in = xhat
        kpp, vpp = k_raw, v_raw
        mod = getattr(cfg, "ctx_mod", "linear")
        if cfg.enable_key_mod:
            # residual pre-MLP: identity at init (== baseline), nonlinearity
            # grows if useful — the non-residual form left the ctx path dead
            k_in = u + F.silu(self.k_pre(u)) if mod == "kmlp" else u
            k_ctx = self.norm_kctx(self.w_kctx(k_in))
            kpp = self._blend(self.w_gk(g_in), k_raw, k_ctx, collect, "k")
        if cfg.enable_value_mod:
            if mod == "vmlp":
                v_in = u + F.silu(self.v_pre(u))
            elif mod == "fusion":
                v_in = u + F.silu(self.v_pre(torch.cat([xhat, u], dim=-1)))
            elif mod in ("puremlp", "puremlp_hw"):
                # NO +u shortcut: W2 small-init grows on random nonlinear
                # features (ELM regime), then back-trains W1 (healthy init)
                v_in = F.silu(self.v_pre(torch.cat([xhat, u], dim=-1)))
            elif mod == "mult":
                v_in = torch.sigmoid(self.v_gate(xhat).float()).to(u.dtype) \
                    * torch.tanh(self.v_map(u))
            elif mod == "mult_res":
                lam = getattr(cfg, "mult_res_lambda", 1.0)
                v_in = lam * u + torch.sigmoid(self.v_gate(xhat).float()).to(u.dtype) \
                    * torch.tanh(self.v_map(u))
            elif mod == "linpoint":
                v_in = self.v_comb(F.silu(self.v_pre(u)))
            elif mod == "glu":
                xu = torch.cat([xhat, u], dim=-1)
                v_in = F.silu(self.v_gate(xu)) * self.v_map(xu)
            else:
                v_in = u
            v_ctx = self.norm_vctx(self.w_vctx(v_in))
            nvr = getattr(cfg, "kmod_vraw_heads", 0)
            kmode = getattr(cfg, "kmod_mode", "raw")
            if nvr > 0 and kmode == "gated_norm":
                # write-gated kmod: the head's value has NO ctx term, so the
                # convex blend g*raw + (1-g)*ctx degenerates to a pure WRITE
                # GATE on the (v_norm-ed) raw value. Gate logits stay LIVE
                # (w_gv head rows train); w_vctx head rows go dead.
                dcut = nvr * self.head_dim
                v_ctx = torch.cat([torch.zeros_like(v_ctx[..., :dcut]),
                                   v_ctx[..., dcut:]], dim=-1)
            vpp = self._blend(self.w_gv(g_in), v_raw, v_ctx, collect, "v")
            if nvr > 0 and kmode == "raw":
                # k-mod-only heads: contextualized addressing, RAW values
                # (flooding is a value-path phenomenon; these heads keep
                # verbatim content addressable via modulated keys)
                dcut = nvr * self.head_dim
                raw = v_raw[..., :dcut]
                if getattr(cfg, "kmod_vraw_bound", "none") == "tanh":
                    # loudness cap: verbatim direction kept, magnitude
                    # bounded — ungated raw v floods the stream (L6-h0 loop)
                    raw = torch.tanh(raw.float()).to(raw.dtype)
                vpp = torch.cat([raw, vpp[..., dcut:]], dim=-1)
            vpp = self.clamp_vpp(vpp)
        if collect is not None:
            collect.setdefault("vpp", []).append(vpp.detach().float())
        return kpp, vpp

    def clamp_vpp(self, vpp):
        """Entry-norm clamp (the storm governor): cap each (position, head)
        value vector at L2 radius v_clamp_tau. Identity for ||v''||<=tau; the
        projection v'' * tau/||v''|| for the tail. fp32 norm; the
        head-broadcast scale is min(1, tau/||.||). One site, mirrors project_v
        / norm_vctx so every path (eager, fused) clamps identically."""
        tau = getattr(self.cfg, "v_clamp_tau", 0.0)
        if not tau or tau <= 0:
            return vpp
        shp = vpp.shape
        vh = vpp.reshape(*shp[:-1], self.n_kv_heads, self.head_dim).float()
        n = vh.norm(dim=-1, keepdim=True)
        scale = (tau / n.clamp_min(1e-20)).clamp(max=1.0)
        return (vh * scale).to(vpp.dtype).reshape(shp)

    def _blend(self, gate_logits, raw, ctx, collect, tag):
        """§3 step 9 for one of the two paths; fp32 gate sigmoid.

        The value path (tag=="v") follows cfg.gate_type_v; keys always follow
        the legacy cfg.gate_type. independent-v: two sigmoids over the [2d]
        logits give a raw gate and a ctx gate (non-convex box [0,1]^2)."""
        if tag == "v":
            gt = "independent" if getattr(self.cfg, "gate_type_v", "convex") \
                == "independent" else "convex_sigmoid"
        else:
            gt = self.cfg.gate_type
        g_ctx = None
        if gt == "convex_sigmoid":
            g = torch.sigmoid(gate_logits.float()).to(raw.dtype)
            out = g * raw + (1.0 - g) * ctx
        elif gt == "independent":
            g1, g2 = torch.sigmoid(gate_logits.float()).to(raw.dtype).chunk(2, dim=-1)
            out = g1 * raw + g2 * ctx
            g, g_ctx = g1, g2
        else:  # unbounded: g is an unconstrained linear output, same blend formula
            g = gate_logits.to(raw.dtype)
            out = g * raw + (1.0 - g) * ctx
        if collect is not None:
            # store g_raw as "g_v" (comparable to convex telemetry); the ctx
            # gate lands in the new "g_v_ctx" slot. ratio_v unchanged.
            collect.setdefault(f"g_{tag}", []).append(g.detach().float())
            if g_ctx is not None:
                collect.setdefault(f"g_{tag}_ctx", []).append(g_ctx.detach().float())
            ratio = ctx.detach().float().norm(dim=-1) / (raw.detach().float().norm(dim=-1) + 1e-8)
            collect.setdefault(f"ratio_{tag}", []).append(ratio)
        return out

    # ---- full-sequence forward paths ---------------------------------------

    def forward(self, xhat, cos, sin, collect=None):
        """xhat: pre-normed [B, T, d]; cos/sin: rope cache rows for positions 0..T-1.
        Returns the layer's residual contribution W_O o_cat, [B, T, d]."""
        mode = self.cfg.attn_mode
        if mode == "vanilla":
            return self._forward_vanilla(xhat, cos, sin)
        if mode == "sequential":
            from .scan import sequential_scan
            o_cat = sequential_scan(self, xhat, cos, sin, collect=collect)
            return self.wo(o_cat)
        if mode == "static":
            return self._forward_static(xhat, cos, sin, collect=collect)
        return self._forward_deq(xhat, cos, sin)

    def _forward_vanilla(self, xhat, cos, sin):
        B, T, _ = xhat.shape
        # query heads and (possibly fewer) KV heads share head_dim, so RoPE is untouched
        q = self.wq(xhat).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(xhat).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.project_v(xhat).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        W = getattr(self.cfg, "local_window", 0) or 0
        if (getattr(self.cfg, "attn_logn_ref", 0) or 0) > 1:
            # length-adaptive logit scale: explicit scores (SDPA cannot scale per row)
            if self.n_kv_heads != self.n_heads:
                rep = self.n_heads // self.n_kv_heads
                k = k.repeat_interleave(rep, dim=1); v = v.repeat_interleave(rep, dim=1)
            s = (q @ k.transpose(-1, -2)) * self.scale
            i = torch.arange(T, device=q.device)
            mask = i[None, :] <= i[:, None]
            if W:
                mask &= i[None, :] > i[:, None] - W
            s = s.masked_fill(~mask, float("-inf"))
            s = self._logn_rows(s)
            o = torch.softmax(s.float(), dim=-1).to(v.dtype) @ v
        elif W:
            i = torch.arange(T, device=q.device)
            mask = (i[None, :] <= i[:, None]) & (i[None, :] > i[:, None] - W)
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                               enable_gqa=self.n_heads != self.n_kv_heads)
        else:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                               enable_gqa=self.n_heads != self.n_kv_heads)
        o = o.transpose(1, 2).reshape(B, T, self.d_attn)
        return self.wo(o)

    def _forward_deq(self, xhat, cos, sin):
        """§6 K-sweep parallel approximation (stub; NOT the v0 training path).

        Sweep 0 attends over the raw cache; each later sweep rebuilds the refined
        cache from the previous sweep's o, in parallel over positions. The self
        (diagonal) entry stays RAW in every sweep. deq_sweeps = total number of
        attention passes; deq_sweeps == T reproduces the sequential scan exactly.
        """
        B, T, d = xhat.shape
        k_flat = self.wk(xhat)
        v_flat = self.project_v(xhat)
        q_r = apply_rope(self._split(self.wq(xhat)), cos, sin)
        k_r = apply_rope(self._split(k_flat), cos, sin)      # raw rotated keys
        v = self._split(v_flat)

        strict_lower = torch.ones(T, T, dtype=torch.bool, device=xhat.device).tril(-1)
        W = getattr(self.cfg, "local_window", 0) or 0
        if W:                                                 # sliding window: keys j > i - W
            strict_lower &= torch.ones(T, T, dtype=torch.bool, device=xhat.device).triu(-(W - 1))
        diag_raw = (q_r * k_r).sum(-1) * self.scale           # [B,h,T] raw self scores

        K_cache, V_cache = k_r, v                             # sweep-0 cache = raw
        o = None
        for m in range(self.cfg.deq_sweeps):
            s = (q_r @ K_cache.transpose(-1, -2)) * self.scale
            s = s.masked_fill(~strict_lower, float("-inf"))   # mask upper AND diagonal
            s = torch.diagonal_scatter(s, diag_raw, dim1=-2, dim2=-1)  # raw diagonal back
            s = self._logn_rows(s)
            a = torch.softmax(s.float(), dim=-1).to(v.dtype)
            a_diag = a.diagonal(dim1=-2, dim2=-1)             # [B,h,T]
            a_off = a * strict_lower
            o = a_off @ V_cache + a_diag.unsqueeze(-1) * v    # raw self value on the diagonal
            if m < self.cfg.deq_sweeps - 1:
                kpp, vpp = self.refine_kv(xhat, self._merge(o), k_flat, v_flat)
                K_cache = apply_rope(self._split(kpp), cos, sin)
                V_cache = self._split(vpp)
        return self.wo(self._merge(o))

    def _forward_static(self, xhat, cos, sin, collect=None):
        """F3 control (static branch): the context branch reads a projection of x̂_j
        instead of u_j = RMSNorm(o_j). Everything else — W_Kctx/W_Vctx, gates, gate
        input width, RMSNorm_ctx, init curriculum — is byte-identical to B; only the
        *source* of the context branch changes from the attention aggregate (carries
        the past) to the token's own normed embedding (carries nothing about the past).

        Consequence: k'', v'' are pure functions of the token, the recurrence vanishes,
        and the whole layer is a single parallel causal-attention pass (SDPA speed).
        The self (diagonal) entry stays RAW, exactly as B (§4-a).
        """
        Bsz, T, d = xhat.shape
        k_flat = self.wk(xhat)
        v_flat = self.project_v(xhat)
        q_r = apply_rope(self._split(self.wq(xhat)), cos, sin)
        k_self_r = apply_rope(self._split(k_flat), cos, sin)   # raw rotated self keys
        v_raw = self._split(v_flat)

        # static refined cache: refine_kv with o_cat := x̂ (deletes contextual content)
        kpp, vpp = self.refine_kv(xhat, xhat, k_flat, v_flat, collect=collect)
        K = apply_rope(self._split(kpp), cos, sin)             # [B,h,T,d_h]
        V = self._split(vpp)

        # one causal pass over the static cache, RAW diagonal (same read pattern as B)
        strict_lower = torch.ones(T, T, dtype=torch.bool, device=xhat.device).tril(-1)
        W = getattr(self.cfg, "local_window", 0) or 0
        if W:
            strict_lower &= torch.ones(T, T, dtype=torch.bool, device=xhat.device).triu(-(W - 1))
        diag_raw = (q_r * k_self_r).sum(-1) * self.scale       # [B,h,T]
        s = (q_r @ K.transpose(-1, -2)) * self.scale
        s = s.masked_fill(~strict_lower, float("-inf"))
        s = torch.diagonal_scatter(s, diag_raw, dim1=-2, dim2=-1)
        s = self._logn_rows(s)
        a = torch.softmax(s.float(), dim=-1).to(V.dtype)
        a_diag = a.diagonal(dim1=-2, dim2=-1)
        a_off = a * strict_lower
        o = a_off @ V + a_diag.unsqueeze(-1) * v_raw
        return self.wo(self._merge(o))

    # ---- incremental decode -------------------------------------------------

    def decode_step(self, xhat_j, pos, cache, cos, sin):
        """One decode step at absolute position `pos` (same math as the scan step).

        xhat_j: [B, 1, d] pre-normed. cache: {"K": [B,h,P,d_h]|None, "V": ...} of
        refined, pre-rotated keys / refined values; appended in place (append-only).
        """
        q = self._split(self.wq(xhat_j))
        k = self._split(self.wk(xhat_j))
        v = self._split(self.project_v(xhat_j))
        cos_j, sin_j = cos[pos:pos + 1], sin[pos:pos + 1]
        q_r = apply_rope(q, cos_j, sin_j)
        k_self_r = apply_rope(k, cos_j, sin_j)

        o_j, _ = self.attend_one(q_r, k_self_r, v, cache["K"], cache["V"])
        o_cat = self._merge(o_j)

        kpp, vpp = self.refine_kv(xhat_j, o_cat, self._merge(k), self._merge(v))
        kpp_r = apply_rope(self._split(kpp), cos_j, sin_j)    # rotate blended key at write
        vpp_h = self._split(vpp)
        if cache["K"] is None:
            cache["K"], cache["V"] = kpp_r, vpp_h
        else:
            cache["K"] = torch.cat([cache["K"], kpp_r], dim=2)
            cache["V"] = torch.cat([cache["V"], vpp_h], dim=2)
        return self.wo(o_cat)
