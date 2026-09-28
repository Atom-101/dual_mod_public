"""Triton fused refine chain (megakernel.md §4): sigmoid gates + convex blend
+ RoPE rotate-at-write + strided cache write, ONE kernel per sweep instead of
~25 pointwise torch ops. Profiling showed the Phase A step is dominated by
2-5 µs elementwise/cat kernels, not GEMMs — this is the hot spot.

Semantics mirror DualModAttention.refine_kv + apply_rope: fp32 sigmoid, blend,
fp32 rotation, cast to cache dtype at the write (fp32 build is bit-comparable
to the torch path; bf16 differs only in blend rounding order, inside the 2e-2
tolerance).

FusedRefine also packs the four ctx/gate projections into two GEMMs
(u @ [W_kctx|W_vctx]^T and [x̂;u] @ [W_gk|W_gv]^T) against per-layer packed
weight buffers refreshed once per step (pack() runs inside the forward
prologue graph, so training weight updates are picked up every step).

Used on the no-grad forward path only; the backward cell recomputes through
the reference torch ops (autograd) — in fp32 both paths are identical, in
bf16 the recompute is a bf16-rounding-level perturbation of the forward point.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_refine_kernel(
    k_ptr, v_ptr, ctx_ptr, glog_ptr, cos_ptr, sin_ptr, kout_ptr, vout_ptr,
    d: tl.constexpr, dh: tl.constexpr, c: tl.constexpr,
    stride_out_b, stride_out_h, stride_out_t,
    N, OUT_BF16: tl.constexpr, HALF: tl.constexpr, HALFP: tl.constexpr,
):
    """One program per row n of N=B*c. Processes all d dims via HALF=d//2
    'a-lanes' (RoPE pair convention: a_dim = head*dh + off, b_dim = a_dim +
    dh//2, off < dh//2). ctx is packed [N, 2d] = (k_ctx | v_ctx); glog is
    packed [N, 2d] = (gk | gv) logits. Outputs are strided views of the cache
    slice rows [pos0, pos0+c) with layout [B, h, T, dh].

    HALFP = next_power_of_2(HALF): tl.arange needs a pow2 extent, so non-pow2
    widths (e.g. d768 -> HALF=384) run with padded lanes masked off (msk).
    Pure elementwise kernel — other=0.0 is safe everywhere (no reductions)."""
    n = tl.program_id(0)
    if n >= N:
        return
    p = n % c                                     # position within the chunk
    lane = tl.arange(0, HALFP)                    # padded d//2 a-lanes
    msk = lane < HALF
    hh = lane // (dh // 2)
    off = lane % (dh // 2)
    a_dim = hh * dh + off
    b_dim = a_dim + dh // 2

    # --- key path: blend then rotate ---
    k_a = tl.load(k_ptr + n * d + a_dim, mask=msk, other=0.0).to(tl.float32)
    k_b = tl.load(k_ptr + n * d + b_dim, mask=msk, other=0.0).to(tl.float32)
    kc_a = tl.load(ctx_ptr + n * 2 * d + a_dim, mask=msk, other=0.0).to(tl.float32)
    kc_b = tl.load(ctx_ptr + n * 2 * d + b_dim, mask=msk, other=0.0).to(tl.float32)
    g_a = tl.sigmoid(tl.load(glog_ptr + n * 2 * d + a_dim, mask=msk,
                             other=0.0).to(tl.float32))
    g_b = tl.sigmoid(tl.load(glog_ptr + n * 2 * d + b_dim, mask=msk,
                             other=0.0).to(tl.float32))
    kpp_a = g_a * k_a + (1.0 - g_a) * kc_a
    kpp_b = g_b * k_b + (1.0 - g_b) * kc_b
    cos = tl.load(cos_ptr + p * (dh // 2) + off, mask=msk, other=0.0)
    sin = tl.load(sin_ptr + p * (dh // 2) + off, mask=msk, other=0.0)
    rot_a = kpp_a * cos - kpp_b * sin
    rot_b = kpp_b * cos + kpp_a * sin

    # --- value path: blend only ---
    v_a = tl.load(v_ptr + n * d + a_dim, mask=msk, other=0.0).to(tl.float32)
    v_b = tl.load(v_ptr + n * d + b_dim, mask=msk, other=0.0).to(tl.float32)
    vc_a = tl.load(ctx_ptr + n * 2 * d + d + a_dim, mask=msk, other=0.0).to(tl.float32)
    vc_b = tl.load(ctx_ptr + n * 2 * d + d + b_dim, mask=msk, other=0.0).to(tl.float32)
    gv_a = tl.sigmoid(tl.load(glog_ptr + n * 2 * d + d + a_dim, mask=msk,
                              other=0.0).to(tl.float32))
    gv_b = tl.sigmoid(tl.load(glog_ptr + n * 2 * d + d + b_dim, mask=msk,
                              other=0.0).to(tl.float32))
    vpp_a = gv_a * v_a + (1.0 - gv_a) * vc_a
    vpp_b = gv_b * v_b + (1.0 - gv_b) * vc_b

    # --- strided write into cache rows: [B, h, T, dh], row n = (b=n//c, p) ---
    b_idx = n // c
    out_base = b_idx * stride_out_b + hh * stride_out_h + p * stride_out_t
    if OUT_BF16:
        tl.store(kout_ptr + out_base + off, rot_a.to(tl.bfloat16), mask=msk)
        tl.store(kout_ptr + out_base + off + dh // 2, rot_b.to(tl.bfloat16),
                 mask=msk)
        tl.store(vout_ptr + out_base + off, vpp_a.to(tl.bfloat16), mask=msk)
        tl.store(vout_ptr + out_base + off + dh // 2, vpp_b.to(tl.bfloat16),
                 mask=msk)
    else:
        tl.store(kout_ptr + out_base + off, rot_a, mask=msk)
        tl.store(kout_ptr + out_base + off + dh // 2, rot_b, mask=msk)
        tl.store(vout_ptr + out_base + off, vpp_a, mask=msk)
        tl.store(vout_ptr + out_base + off + dh // 2, vpp_b, mask=msk)


class FusedRefine:
    """Per-model packed ctx/gate weights + the fused kernel launcher."""

    def __init__(self, model):
        cfg = model.cfg
        from .scan_cell_fwd import PACKED_CTX_MODS
        self.mod = getattr(cfg, "ctx_mod", "linear")
        self.nvr = getattr(cfg, "kmod_vraw_heads", 0)
        self.kmode = getattr(cfg, "kmod_mode", "raw")
        assert cfg.gate_type == "convex_sigmoid" and cfg.gate_input == "xu" \
            and cfg.enable_key_mod and cfg.enable_value_mod and cfg.rmsnorm_on_o \
            and self.mod in PACKED_CTX_MODS \
            and getattr(cfg, "kmod_vraw_bound", "none") == "none" \
            and getattr(cfg, "gate_type_v", "convex") == "convex", \
            "FusedRefine covers the standard dual-mod cell + PACKED_CTX_MODS " \
            "x kmod; use the reference path for other variants " \
            "(independent-v bakes a different value blend)"
        self.model = model
        d = cfg.d_model
        dev = next(model.parameters()).device
        dt = model.blocks[0].attn.w_kctx.weight.dtype
        L = cfg.n_layers
        self.W_ctx = [torch.zeros(2 * d, d, device=dev, dtype=dt) for _ in range(L)]
        self.W_g = [torch.zeros(2 * d, 2 * d, device=dev, dtype=dt) for _ in range(L)]
        self.b_g = [torch.zeros(2 * d, device=dev, dtype=dt) for _ in range(L)]
        self._li_of = {id(blk.attn): li for li, blk in enumerate(model.blocks)}

    @torch.no_grad()
    def pack(self):
        """Refresh packed weights from the live params (graph-captured in the
        forward prologue, so it runs once per step)."""
        d = self.model.cfg.d_model

        def gate_dense(g):
            w = g.weight
            return w if w.shape[1] == 2 * d else g.b.weight @ g.a.weight

        for li, blk in enumerate(self.model.blocks):
            a = blk.attn
            self.W_ctx[li][:d].copy_(a.w_kctx.weight)
            self.W_ctx[li][d:].copy_(a.w_vctx.weight)
            self.W_g[li][:d].copy_(gate_dense(a.w_gk))
            self.W_g[li][d:].copy_(gate_dense(a.w_gv))
            self.b_g[li][:d].copy_(a.w_gk.bias)
            self.b_g[li][d:].copy_(a.w_gv.bias)

    def refine_write(self, attn, xhat_c, o_cat, k_flat, v_flat, cos_c, sin_c,
                     K_cache, V_cache, pos0):
        li = self._li_of[id(attn)]
        B, c, d = xhat_c.shape
        N = B * c
        u = attn.norm_ctx(o_cat)
        if self.mod == "linear":
            ctx = (u.reshape(N, d) @ self.W_ctx[li].transpose(0, 1))   # [N, 2d]
            if getattr(attn.cfg, "v_ctx_norm", "none") == "rms":
                # v_ctx_norm torch-side, BEFORE the kernel (kernel untouched);
                # variants get it inside _ctx_variant. gated_norm zero below
                # stays LAST (eager order)
                ctx = torch.cat([ctx[:, :d], attn.norm_vctx(ctx[:, d:])],
                                dim=-1)
        else:
            from .scan_cell_fwd import _ctx_variant
            ctx = _ctx_variant(attn, xhat_c.reshape(N, d), u.reshape(N, d),
                               self.mod)
        g_in = torch.cat([xhat_c, u], dim=-1).reshape(N, 2 * d)
        glog = torch.addmm(self.b_g[li], g_in, self.W_g[li].transpose(0, 1))
        if self.nvr > 0:
            if self.kmode == "raw":
                # kmod heads: pin v-gate logits -> g=1 -> raw v committed
                glog[:, d:d + self.nvr * attn.head_dim] = 30.0
            else:
                # gated_norm: gate logits LIVE; zero the head's v_ctx slice
                # so the convex blend reduces to the pure write gate
                ctx[:, d:d + self.nvr * attn.head_dim] = 0.0
        dh = attn.head_dim
        kout = K_cache[:, :, pos0:]        # base at pos0; kernel writes p<c
        vout = V_cache[:, :, pos0:]
        assert K_cache.dtype in (torch.bfloat16, torch.float32)
        _fused_refine_kernel[(N,)](
            k_flat.reshape(N, d), v_flat.reshape(N, d), ctx, glog,
            cos_c.contiguous(), sin_c.contiguous(), kout, vout,
            d=d, dh=dh, c=c,
            stride_out_b=K_cache.stride(0), stride_out_h=K_cache.stride(1),
            stride_out_t=K_cache.stride(2),
            N=N, OUT_BF16=K_cache.dtype == torch.bfloat16, HALF=d // 2,
            HALFP=triton.next_power_of_2(d // 2),
            num_warps=4,
        )


@triton.jit
def _fused_refine_bwd_kernel(
    dkout_ptr, dvout_ptr,               # incoming grads (rot-space K, V)
    k_ptr, v_ptr, ctx_ptr, glog_ptr, cos_ptr, sin_ptr,
    dk_ptr, dv_ptr, dctx_ptr, dglog_ptr,
    d: tl.constexpr, dh: tl.constexpr, c: tl.constexpr,
    stride_in_b, stride_in_h, stride_in_t,
    N, HALF: tl.constexpr, HALFP: tl.constexpr,
):
    """Backward of _fused_refine_kernel. Incoming grads are [B,h,c,dh]
    (strides given); parameter-side outputs are [N,d]/[N,2d] contiguous.
    HALFP-padded lanes masked off (elementwise, other=0.0 safe)."""
    n = tl.program_id(0)
    if n >= N:
        return
    p = n % c
    b_idx = n // c
    lane = tl.arange(0, HALFP)
    msk = lane < HALF
    hh = lane // (dh // 2)
    off = lane % (dh // 2)
    a_dim = hh * dh + off
    b_dim = a_dim + dh // 2
    in_base = b_idx * stride_in_b + hh * stride_in_h + p * stride_in_t

    cos = tl.load(cos_ptr + p * (dh // 2) + off, mask=msk, other=0.0)
    sin = tl.load(sin_ptr + p * (dh // 2) + off, mask=msk, other=0.0)
    dK1 = tl.load(dkout_ptr + in_base + off, mask=msk, other=0.0).to(tl.float32)
    dK2 = tl.load(dkout_ptr + in_base + off + dh // 2, mask=msk,
                  other=0.0).to(tl.float32)
    # rot_a = a*cos - b*sin ; rot_b = b*cos + a*sin  =>
    da = dK1 * cos + dK2 * sin
    db = -dK1 * sin + dK2 * cos

    k_a = tl.load(k_ptr + n * d + a_dim, mask=msk, other=0.0).to(tl.float32)
    k_b = tl.load(k_ptr + n * d + b_dim, mask=msk, other=0.0).to(tl.float32)
    kc_a = tl.load(ctx_ptr + n * 2 * d + a_dim, mask=msk, other=0.0).to(tl.float32)
    kc_b = tl.load(ctx_ptr + n * 2 * d + b_dim, mask=msk, other=0.0).to(tl.float32)
    gl_a = tl.load(glog_ptr + n * 2 * d + a_dim, mask=msk, other=0.0).to(tl.float32)
    gl_b = tl.load(glog_ptr + n * 2 * d + b_dim, mask=msk, other=0.0).to(tl.float32)
    g_a = tl.sigmoid(gl_a)
    g_b = tl.sigmoid(gl_b)
    dt = dk_ptr.dtype.element_ty
    tl.store(dk_ptr + n * d + a_dim, (da * g_a).to(dt), mask=msk)
    tl.store(dk_ptr + n * d + b_dim, (db * g_b).to(dt), mask=msk)
    tl.store(dctx_ptr + n * 2 * d + a_dim, (da * (1.0 - g_a)).to(dt), mask=msk)
    tl.store(dctx_ptr + n * 2 * d + b_dim, (db * (1.0 - g_b)).to(dt), mask=msk)
    tl.store(dglog_ptr + n * 2 * d + a_dim,
             (da * (k_a - kc_a) * g_a * (1.0 - g_a)).to(dt), mask=msk)
    tl.store(dglog_ptr + n * 2 * d + b_dim,
             (db * (k_b - kc_b) * g_b * (1.0 - g_b)).to(dt), mask=msk)

    dV1 = tl.load(dvout_ptr + in_base + off, mask=msk, other=0.0).to(tl.float32)
    dV2 = tl.load(dvout_ptr + in_base + off + dh // 2, mask=msk,
                  other=0.0).to(tl.float32)
    v_a = tl.load(v_ptr + n * d + a_dim, mask=msk, other=0.0).to(tl.float32)
    v_b = tl.load(v_ptr + n * d + b_dim, mask=msk, other=0.0).to(tl.float32)
    vc_a = tl.load(ctx_ptr + n * 2 * d + d + a_dim, mask=msk, other=0.0).to(tl.float32)
    vc_b = tl.load(ctx_ptr + n * 2 * d + d + b_dim, mask=msk, other=0.0).to(tl.float32)
    gvl_a = tl.load(glog_ptr + n * 2 * d + d + a_dim, mask=msk,
                    other=0.0).to(tl.float32)
    gvl_b = tl.load(glog_ptr + n * 2 * d + d + b_dim, mask=msk,
                    other=0.0).to(tl.float32)
    gv_a = tl.sigmoid(gvl_a)
    gv_b = tl.sigmoid(gvl_b)
    tl.store(dv_ptr + n * d + a_dim, (dV1 * gv_a).to(dt), mask=msk)
    tl.store(dv_ptr + n * d + b_dim, (dV2 * gv_b).to(dt), mask=msk)
    tl.store(dctx_ptr + n * 2 * d + d + a_dim, (dV1 * (1.0 - gv_a)).to(dt),
             mask=msk)
    tl.store(dctx_ptr + n * 2 * d + d + b_dim, (dV2 * (1.0 - gv_b)).to(dt),
             mask=msk)
    tl.store(dglog_ptr + n * 2 * d + d + a_dim,
             (dV1 * (v_a - vc_a) * gv_a * (1.0 - gv_a)).to(dt), mask=msk)
    tl.store(dglog_ptr + n * 2 * d + d + b_dim,
             (dV2 * (v_b - vc_b) * gv_b * (1.0 - gv_b)).to(dt), mask=msk)


class FusedRefineFn(torch.autograd.Function):
    """Differentiable fused pointwise refine (blend+rope), for the backward
    cell's recompute: ONE autograd node instead of ~25. Returns contiguous
    (K_rows [B,h,c,dh] rotated, V_rows [B,h,c,dh])."""

    @staticmethod
    def forward(ctx, k_flat, v_flat, ctx_packed, glog, cos_c, sin_c, B, c, h, dh):
        N, d = B * c, h * dh
        K_rows = k_flat.new_empty(B, h, c, dh)
        V_rows = v_flat.new_empty(B, h, c, dh)
        _fused_refine_kernel[(N,)](
            k_flat.reshape(N, d), v_flat.reshape(N, d), ctx_packed, glog,
            cos_c.contiguous(), sin_c.contiguous(), K_rows, V_rows,
            d=d, dh=dh, c=c,
            stride_out_b=K_rows.stride(0), stride_out_h=K_rows.stride(1),
            stride_out_t=K_rows.stride(2),
            N=N, OUT_BF16=K_rows.dtype == torch.bfloat16, HALF=d // 2,
            HALFP=triton.next_power_of_2(d // 2), num_warps=4)
        ctx.save_for_backward(k_flat, v_flat, ctx_packed, glog, cos_c, sin_c)
        ctx.dims = (B, c, h, dh)
        return K_rows, V_rows

    @staticmethod
    def backward(ctx, dK_rows, dV_rows):
        k_flat, v_flat, ctx_packed, glog, cos_c, sin_c = ctx.saved_tensors
        B, c, h, dh = ctx.dims
        N, d = B * c, h * dh
        dk = torch.empty_like(k_flat.reshape(N, d))
        dv = torch.empty_like(v_flat.reshape(N, d))
        dctx = torch.empty_like(ctx_packed)
        dglog = torch.empty_like(glog)
        dK_rows = dK_rows.contiguous()
        dV_rows = dV_rows.contiguous()
        _fused_refine_bwd_kernel[(N,)](
            dK_rows, dV_rows,
            k_flat.reshape(N, d), v_flat.reshape(N, d), ctx_packed, glog,
            cos_c.contiguous(), sin_c.contiguous(),
            dk, dv, dctx, dglog,
            d=d, dh=dh, c=c,
            stride_in_b=dK_rows.stride(0), stride_in_h=dK_rows.stride(1),
            stride_in_t=dK_rows.stride(2),
            N=N, HALF=d // 2, HALFP=triton.next_power_of_2(d // 2),
            num_warps=4)
        return (dk.reshape(B, c, d), dv.reshape(B, c, d), dctx, dglog,
                None, None, None, None, None, None)
