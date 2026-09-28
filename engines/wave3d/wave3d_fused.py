"""Fused Triton refine chain for Wave3DX (engine flag fused_refine, DEFAULT True since 2026-09-02).

Same operator as wave3d_ops.refine (mult_res ctx / xu gate input / convex
sigmoid gates / k-norm / raw k-mod head / entry-norm clamp / rope-at-write) as
ONE autograd.Function: the GEMMs stay cuBLAS bmm's (the batch-5G LoRA-up GEMM
runs near HBM peak there), the ENTIRE pointwise chain between/after them is
three hand-written memory-bound Triton kernels per direction that read every
GEMM output exactly once:

  forward   _rms_cat_fwd  o, xb                 -> g_in = [xb | u = rms(o)*w], rstd
            bmm x3 + stack g_in, u, xb           -> down5 [G,5,M,144]
            bmm (batch 5G) down5                 -> up5 [G,5,M,d] = (k_ctx_raw, gk, gv, vg, vm)
            _vin_fwd       u, vg, vm             -> v_in = lam*u + sig(vg+b)*tanh(vm)
            bmm x2         v_in                  -> vctx_d [M,144], v_ctx [M,d]
            _kv_fwd        up5[0:3], v_ctx, kf, vf, cos/sin -> K_ref (k-norm, gate, blend, rope),
                                                    V (gate, blend, raw head, entry-norm clamp)
  backward  _kv_bwd        dK, dV + the same inputs -> d(up5)[0:3], dv_ctx, dkf, dvf, partials of
                                                    d bias_gk / d bias_gv / d k_gain
            bmm x4         vctx up/down dX + dW   (dv_in [M,d])
            _vin_bwd       vg, vm, dv_in         -> d(up5)[3:5], partial d bias_vg
            bmm x2 (5G)    ups5 dX + dW;  bmm x4 (K-concat): du = lam*dv_in + [d_ud|d_gd] @ [ud2;ga2_u]
                           (baddbmm_ in place), dxb = [d_gd|d_vgx] @ [ga2_x;vgd], and the two dW's
            _u_bwd         du -> do (rms backward), d w_ctx

Why not fuse the up-GEMMs into the epilogue (register GEMM)? Measured on B300:
a [12288,144]x[144,2304] bf16 dot tile is latency-bound in Triton (best 31 us
~260 TFLOP/s; cuBLAS at batch 3 is no better) while the batch-15 cuBLAS up5
GEMM moves 283 MB in 43 us (~HBM peak); recomputing five kinds of dots in fwd
and bwd cost more than the up5 round-trip it avoids (v1 kept in
out/_wave3d_scratch/wave3d_fused_dots_v1.py.bak). What WAS wasteful is the
inductor backward: its select_backward fusion re-read seven d-wide tensors five
times to write d(up5) (403 us) plus three fp32 d-wide intermediates -- gone here.

Tiles: head kernels handle NHG heads per program as two dh/2 column halves
(rope partner = the other half; per-head reductions = sum of both halves) so
no layout shuffles; row kernels loop over D in BD chunks. Config in CFG
(fixed => no autotune => CUDA-graph safe).

Numerics: fp32 math; in bf16 (autocast) mode the forward reproduces the module's
bf16 rounding points (GEMM outputs are bf16 anyway, sigmoid/tanh outputs, blend
products, kpp before rope) so K/V match the eager chain to GEMM noise; the
backward rounds once per stored tensor (autograd's bf16 InputBuffer accumulation
order is not reproduced). fp32 mode (no autocast): exact fp32 (validation V1).
1-D gains/biases are read as fp32 whatever their storage dtype. Triton 3.7 /
sm100 gotcha (v1): a rank-0 reduction of a dot-derived tile trips the TMEM
layout pass; here the k_gain grad is stored per row and summed in torch anyway.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from engines.wave3d import wave3d_ops as X

CFG = dict(BM_HEAD=16, NHG=1, WARPS_HEAD=2,        # head kernels: BM rows x NHG heads per program
           BM_VIN=32, BN_VIN=64, WARPS_VIN=4,       # vin kernels: BM rows x BN cols per program
           ROWS_RMS=4, BD_RMS=512, WARPS_RMS=4,     # row kernels (rms fwd / u bwd)
           ROUND_EMU=0)   # 1: also reproduce the module's bf16 INTERMEDIATE roundings in the forward
                          #    (inductor's compiled refine does not: ROUND_EMU=0 matches it bit-for-bit)


# ============================================================================ helpers
@triton.jit
def _rnd(x, ROUND: tl.constexpr):
    """bf16 rounding point of the module code (no-op in fp32 mode)."""
    if ROUND:
        x = x.to(tl.bfloat16).to(tl.float32)
    return x


@triton.jit
def _ld(ptr, rows, s_m, cols, rmask):
    """[BM, cols] tile of a (batch-offset) row-major tensor as fp32."""
    return tl.load(ptr + rows[:, None] * s_m + cols[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)


@triton.jit
def _st(ptr, rows, s_m, cols, rmask, val):
    tl.store(ptr + rows[:, None] * s_m + cols[None, :], val.to(ptr.dtype.element_ty), mask=rmask[:, None])


# ============================================================================ slot pack / gather
@triton.jit
def _cp_slot(src, s_sm, dst, s_dm, rows, rmask):
    """dst[rows, :144] = src[rows, :144]  (144 = 128 + 16 columns)."""
    k1 = tl.arange(0, 128)
    k2 = 128 + tl.arange(0, 16)
    v1 = tl.load(src + rows[:, None] * s_sm + k1[None, :], mask=rmask[:, None], other=0.0)
    v2 = tl.load(src + rows[:, None] * s_sm + k2[None, :], mask=rmask[:, None], other=0.0)
    tl.store(dst + rows[:, None] * s_dm + k1[None, :], v1, mask=rmask[:, None])
    tl.store(dst + rows[:, None] * s_dm + k2[None, :], v2, mask=rmask[:, None])


@triton.jit
def _pack_down5(ud_ptr, s_udg, s_udm, gd_ptr, s_gdg, s_gdm, vg_ptr, s_vgg, s_vgm, dst_ptr, s_dg, s_ds, M, R,
                BM: tl.constexpr):
    """down5[g] = stack(ud[:, :R], gd[:, :R], gd[:, R:], vg_d, ud[:, R:])  (torch.stack ran at ~1 TB/s)."""
    pid = tl.program_id(0)
    g = tl.program_id(1)
    rows = pid * BM + tl.arange(0, BM)
    rmask = rows < M
    _cp_slot(ud_ptr + g * s_udg, s_udm, dst_ptr + g * s_dg + 0 * s_ds, R, rows, rmask)
    _cp_slot(gd_ptr + g * s_gdg, s_gdm, dst_ptr + g * s_dg + 1 * s_ds, R, rows, rmask)
    _cp_slot(gd_ptr + g * s_gdg + R, s_gdm, dst_ptr + g * s_dg + 2 * s_ds, R, rows, rmask)
    _cp_slot(vg_ptr + g * s_vgg, s_vgm, dst_ptr + g * s_dg + 3 * s_ds, R, rows, rmask)
    _cp_slot(ud_ptr + g * s_udg + R, s_udm, dst_ptr + g * s_dg + 4 * s_ds, R, rows, rmask)


@triton.jit
def _gather_ddown(src_ptr, s_sg, s_ss, s_sm, dst_ptr, s_dg, s_dm, M, R, BM: tl.constexpr):
    """dst[g, m] = [d_down5 slots 0, 4, 1, 2, 3] = [d_ud | d_gd | d_vgx] (row-major [G, M, 5R])."""
    pid = tl.program_id(0)
    g = tl.program_id(1)
    rows = pid * BM + tl.arange(0, BM)
    rmask = rows < M
    src = src_ptr + g * s_sg
    dst = dst_ptr + g * s_dg
    _cp_slot(src + 0 * s_ss, s_sm, dst + 0 * R, s_dm, rows, rmask)
    _cp_slot(src + 4 * s_ss, s_sm, dst + 1 * R, s_dm, rows, rmask)
    _cp_slot(src + 1 * s_ss, s_sm, dst + 2 * R, s_dm, rows, rmask)
    _cp_slot(src + 2 * s_ss, s_sm, dst + 3 * R, s_dm, rows, rmask)
    _cp_slot(src + 3 * s_ss, s_sm, dst + 4 * R, s_dm, rows, rmask)


# ============================================================================ forward kernels
@triton.jit
def _rms_cat_fwd(o_ptr, xb_ptr, w_ptr, gin_ptr, rstd_ptr, M, D,
                 s_og, s_om, s_xg, s_xm, s_gg, s_gm, eps,
                 ROWS: tl.constexpr, BD: tl.constexpr):
    """g_in[:, :D] = xb, g_in[:, D:] = u = rms(o) * w  (fp32 math); rstd saved."""
    pid = tl.program_id(0)
    g = tl.program_id(1)
    rows = pid * ROWS + tl.arange(0, ROWS)
    rmask = rows < M
    ss = tl.zeros([ROWS], dtype=tl.float32)
    for d0 in range(0, D, BD):
        cols = d0 + tl.arange(0, BD)
        m = rmask[:, None] & (cols < D)[None, :]
        o = tl.load(o_ptr + g * s_og + rows[:, None] * s_om + cols[None, :], mask=m, other=0.0).to(tl.float32)
        ss += tl.sum(o * o, axis=1)
    rstd = tl.rsqrt(ss / D + eps)
    tl.store(rstd_ptr + g * M + rows, rstd, mask=rmask)
    for d0 in range(0, D, BD):
        cols = d0 + tl.arange(0, BD)
        cm = cols < D
        m = rmask[:, None] & cm[None, :]
        o = tl.load(o_ptr + g * s_og + rows[:, None] * s_om + cols[None, :], mask=m, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + g * D + cols, mask=cm, other=0.0).to(tl.float32)
        u = (o * rstd[:, None]) * w[None, :]
        xb = tl.load(xb_ptr + g * s_xg + rows[:, None] * s_xm + cols[None, :], mask=m, other=0.0)
        dst = gin_ptr + g * s_gg + rows[:, None] * s_gm + cols[None, :]
        tl.store(dst, xb.to(gin_ptr.dtype.element_ty), mask=m)
        tl.store(dst + D, u.to(gin_ptr.dtype.element_ty), mask=m)


@triton.jit
def _vin_fwd(up_ptr, s_upg, s_ups, bvg_ptr, u_ptr, s_ug, s_um, out_ptr, M, D, lam,
             BM: tl.constexpr, BN: tl.constexpr, ROUND: tl.constexpr):
    """v_in = lam*u + sigmoid(vg + b) * tanh(vm)   (vg = up5 slot 3, vm = slot 4)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    g = tl.program_id(2)
    rows = pid_m * BM + tl.arange(0, BM)
    rmask = rows < M
    cols = pid_n * BN + tl.arange(0, BN)
    vg = _ld(up_ptr + g * s_upg + 3 * s_ups, rows, D, cols, rmask)
    vm = _ld(up_ptr + g * s_upg + 4 * s_ups, rows, D, cols, rmask)
    b = tl.load(bvg_ptr + g * D + cols).to(tl.float32)
    s = _rnd(tl.sigmoid(vg + b[None, :]), ROUND)
    t = _rnd(libdevice.tanh(vm), ROUND)
    u = _ld(u_ptr + g * s_ug, rows, s_um, cols, rmask)
    _st(out_ptr + g * M * D, rows, D, cols, rmask, _rnd(lam * u, ROUND) + _rnd(s * t, ROUND))


@triton.jit
def _kv_fwd(up_ptr, s_upg, s_ups, vc_ptr, gain_ptr, bgk_ptr, bgv_ptr,
            kf_ptr, s_kfg, s_kfm, vf_ptr, s_vfg, s_vfm, cos_ptr, sin_ptr, s_cg,
            kout_ptr, vout_ptr, M, D, H, Bsz, dcut, tau, eps,
            BM: tl.constexpr, DH: tl.constexpr, NHG: tl.constexpr, ROUND: tl.constexpr, CLAMP: tl.constexpr):
    """NHG heads per program. K: k_ctx = head_rms(up[0]) * gain; g = sig(up[1] + b);
    kpp = g*kf + (1-g)*k_ctx; K = rope(kpp).  V: g2 = sig(up[2] + b); vpp = g2*vf +
    (1-g2)*v_ctx (raw for cols < dcut); V = vpp * min(1, tau/||vpp||_head)."""
    pid_m = tl.program_id(0)
    hg = tl.program_id(1)
    g = tl.program_id(2)
    HALF: tl.constexpr = DH // 2
    rows = pid_m * BM + tl.arange(0, BM)
    rmask = rows < M
    j = tl.arange(0, HALF)
    w = rows // Bsz
    cs = tl.load(cos_ptr + g * s_cg + w[:, None] * HALF + j[None, :], mask=rmask[:, None], other=0.0)
    sn = tl.load(sin_ptr + g * s_cg + w[:, None] * HALF + j[None, :], mask=rmask[:, None], other=0.0)
    k_p = up_ptr + g * s_upg
    gk_p = k_p + s_ups
    gv_p = k_p + 2 * s_ups
    vc_p = vc_ptr + g * M * D
    kf_p = kf_ptr + g * s_kfg
    vf_p = vf_ptr + g * s_vfg
    ko_p = kout_ptr + g * M * D
    vo_p = vout_ptr + g * M * D
    for i in range(NHG):
        hd = hg * NHG + i
        c1 = hd * DH + j
        c2 = c1 + HALF
        # ---- keys
        k1 = _ld(k_p, rows, D, c1, rmask)
        k2 = _ld(k_p, rows, D, c2, rmask)
        rstd = tl.rsqrt((tl.sum(k1 * k1, axis=1) + tl.sum(k2 * k2, axis=1)) / DH + eps)
        gain = tl.load(gain_ptr + g * H + hd).to(tl.float32)
        kc1 = _rnd((k1 * rstd[:, None]) * gain, ROUND)
        kc2 = _rnd((k2 * rstd[:, None]) * gain, ROUND)
        bb1 = tl.load(bgk_ptr + g * D + c1).to(tl.float32)
        bb2 = tl.load(bgk_ptr + g * D + c2).to(tl.float32)
        gg1 = _rnd(tl.sigmoid(_ld(gk_p, rows, D, c1, rmask) + bb1[None, :]), ROUND)
        gg2 = _rnd(tl.sigmoid(_ld(gk_p, rows, D, c2, rmask) + bb2[None, :]), ROUND)
        kf1 = _ld(kf_p, rows, s_kfm, c1, rmask)
        kf2 = _ld(kf_p, rows, s_kfm, c2, rmask)
        kpp1 = _rnd(_rnd(gg1 * kf1, ROUND) + _rnd(_rnd(1.0 - gg1, ROUND) * kc1, ROUND), ROUND)
        kpp2 = _rnd(_rnd(gg2 * kf2, ROUND) + _rnd(_rnd(1.0 - gg2, ROUND) * kc2, ROUND), ROUND)
        _st(ko_p, rows, D, c1, rmask, kpp1 * cs - kpp2 * sn)
        _st(ko_p, rows, D, c2, rmask, kpp2 * cs + kpp1 * sn)
        # ---- values
        bb1 = tl.load(bgv_ptr + g * D + c1).to(tl.float32)
        bb2 = tl.load(bgv_ptr + g * D + c2).to(tl.float32)
        gg1 = _rnd(tl.sigmoid(_ld(gv_p, rows, D, c1, rmask) + bb1[None, :]), ROUND)
        gg2 = _rnd(tl.sigmoid(_ld(gv_p, rows, D, c2, rmask) + bb2[None, :]), ROUND)
        vf1 = _ld(vf_p, rows, s_vfm, c1, rmask)
        vf2 = _ld(vf_p, rows, s_vfm, c2, rmask)
        vc1 = _ld(vc_p, rows, D, c1, rmask)
        vc2 = _ld(vc_p, rows, D, c2, rmask)
        vpp1 = _rnd(_rnd(gg1 * vf1, ROUND) + _rnd(_rnd(1.0 - gg1, ROUND) * vc1, ROUND), ROUND)
        vpp2 = _rnd(_rnd(gg2 * vf2, ROUND) + _rnd(_rnd(1.0 - gg2, ROUND) * vc2, ROUND), ROUND)
        vpp1 = tl.where((c1 < dcut)[None, :], vf1, vpp1)
        vpp2 = tl.where((c2 < dcut)[None, :], vf2, vpp2)
        if CLAMP:
            n = tl.sqrt(tl.sum(vpp1 * vpp1, axis=1) + tl.sum(vpp2 * vpp2, axis=1))
            scale = tl.minimum(tau / tl.maximum(n, 1e-20), 1.0)
            vpp1 = vpp1 * scale[:, None]
            vpp2 = vpp2 * scale[:, None]
        _st(vo_p, rows, D, c1, rmask, vpp1)
        _st(vo_p, rows, D, c2, rmask, vpp2)


# ============================================================================ backward kernels
@triton.jit
def _kv_bwd(up_ptr, s_upg, s_ups, vc_ptr, gain_ptr, bgk_ptr, bgv_ptr,
            kf_ptr, s_kfg, s_kfm, vf_ptr, s_vfg, s_vfm, cos_ptr, sin_ptr, s_cg,
            dk_ptr, s_dkg, s_dkm, dv_ptr, s_dvg, s_dvm,
            dup_ptr, s_dupg, s_dups, dvc_ptr, dkf_ptr, dvf_ptr, pgain_ptr, pgk_ptr, pgv_ptr,
            M, D, H, NMT, Bsz, dcut, tau, eps,
            BM: tl.constexpr, DH: tl.constexpr, NHG: tl.constexpr, ROUND: tl.constexpr, CLAMP: tl.constexpr):
    """Backward of _kv_fwd. dK -> d up[0] (k-norm bwd), d up[1] (gate), dkf; dV -> d up[2],
    dv_ctx, dvf. Partials: d k_gain per row [G, H, M]; d bias_gk / d bias_gv per row tile
    [G, NMT, D] (summed in torch)."""
    pid_m = tl.program_id(0)
    hg = tl.program_id(1)
    g = tl.program_id(2)
    HALF: tl.constexpr = DH // 2
    rows = pid_m * BM + tl.arange(0, BM)
    rmask = rows < M
    j = tl.arange(0, HALF)
    w = rows // Bsz
    cs = tl.load(cos_ptr + g * s_cg + w[:, None] * HALF + j[None, :], mask=rmask[:, None], other=0.0)
    sn = tl.load(sin_ptr + g * s_cg + w[:, None] * HALF + j[None, :], mask=rmask[:, None], other=0.0)
    k_p = up_ptr + g * s_upg
    gk_p = k_p + s_ups
    gv_p = k_p + 2 * s_ups
    vc_p = vc_ptr + g * M * D
    kf_p = kf_ptr + g * s_kfg
    vf_p = vf_ptr + g * s_vfg
    dk_p = dk_ptr + g * s_dkg
    dv_p = dv_ptr + g * s_dvg
    d0_p = dup_ptr + g * s_dupg
    d1_p = d0_p + s_dups
    d2_p = d0_p + 2 * s_dups
    dvc_p = dvc_ptr + g * M * D
    dkf_p = dkf_ptr + g * M * D
    dvf_p = dvf_ptr + g * M * D
    pgk = pgk_ptr + (g * NMT + pid_m) * D
    pgv = pgv_ptr + (g * NMT + pid_m) * D
    for i in range(NHG):
        hd = hg * NHG + i
        c1 = hd * DH + j
        c2 = c1 + HALF
        # ================= keys
        k1 = _ld(k_p, rows, D, c1, rmask)
        k2 = _ld(k_p, rows, D, c2, rmask)
        rstd = tl.rsqrt((tl.sum(k1 * k1, axis=1) + tl.sum(k2 * k2, axis=1)) / DH + eps)
        gain = tl.load(gain_ptr + g * H + hd).to(tl.float32)
        xn1 = k1 * rstd[:, None]
        xn2 = k2 * rstd[:, None]
        kc1 = _rnd(xn1 * gain, ROUND)
        kc2 = _rnd(xn2 * gain, ROUND)
        bb1 = tl.load(bgk_ptr + g * D + c1).to(tl.float32)
        bb2 = tl.load(bgk_ptr + g * D + c2).to(tl.float32)
        s1 = tl.sigmoid(_ld(gk_p, rows, D, c1, rmask) + bb1[None, :])
        s2 = tl.sigmoid(_ld(gk_p, rows, D, c2, rmask) + bb2[None, :])
        gg1 = _rnd(s1, ROUND)
        gg2 = _rnd(s2, ROUND)
        omg1 = _rnd(1.0 - gg1, ROUND)
        omg2 = _rnd(1.0 - gg2, ROUND)
        kf1 = _ld(kf_p, rows, s_kfm, c1, rmask)
        kf2 = _ld(kf_p, rows, s_kfm, c2, rmask)
        dK1 = _ld(dk_p, rows, s_dkm, c1, rmask)
        dK2 = _ld(dk_p, rows, s_dkm, c2, rmask)
        dkpp1 = _rnd(dK1 * cs + dK2 * sn, ROUND)                 # un-rotate
        dkpp2 = _rnd(dK2 * cs - dK1 * sn, ROUND)
        _st(dkf_p, rows, D, c1, rmask, _rnd(dkpp1 * gg1, ROUND))
        _st(dkf_p, rows, D, c2, rmask, _rnd(dkpp2 * gg2, ROUND))
        dgg1 = dkpp1 * kf1 - dkpp1 * kc1
        dgg2 = dkpp2 * kf2 - dkpp2 * kc2
        dkc1 = _rnd(dkpp1 * omg1, ROUND)
        dkc2 = _rnd(dkpp2 * omg2, ROUND)
        dz1 = dgg1 * (s1 * (1.0 - s1))                            # sigmoid bwd on the fp32 output
        dz2 = dgg2 * (s2 * (1.0 - s2))
        _st(d1_p, rows, D, c1, rmask, dz1)
        _st(d1_p, rows, D, c2, rmask, dz2)
        tl.store(pgk + c1, tl.sum(dz1, axis=0))
        tl.store(pgk + c2, tl.sum(dz2, axis=0))
        tl.store(pgain_ptr + (g * H + hd) * M + rows, tl.sum(dkc1 * xn1, axis=1) + tl.sum(dkc2 * xn2, axis=1),
                 mask=rmask)
        dxn1 = dkc1 * gain
        dxn2 = dkc2 * gain
        mdot = (tl.sum(dxn1 * xn1, axis=1) + tl.sum(dxn2 * xn2, axis=1)) / DH
        _st(d0_p, rows, D, c1, rmask, rstd[:, None] * (dxn1 - xn1 * mdot[:, None]))
        _st(d0_p, rows, D, c2, rmask, rstd[:, None] * (dxn2 - xn2 * mdot[:, None]))
        # ================= values
        bb1 = tl.load(bgv_ptr + g * D + c1).to(tl.float32)
        bb2 = tl.load(bgv_ptr + g * D + c2).to(tl.float32)
        s1 = tl.sigmoid(_ld(gv_p, rows, D, c1, rmask) + bb1[None, :])
        s2 = tl.sigmoid(_ld(gv_p, rows, D, c2, rmask) + bb2[None, :])
        gg1 = _rnd(s1, ROUND)
        gg2 = _rnd(s2, ROUND)
        omg1 = _rnd(1.0 - gg1, ROUND)
        omg2 = _rnd(1.0 - gg2, ROUND)
        vf1 = _ld(vf_p, rows, s_vfm, c1, rmask)
        vf2 = _ld(vf_p, rows, s_vfm, c2, rmask)
        vc1 = _ld(vc_p, rows, D, c1, rmask)
        vc2 = _ld(vc_p, rows, D, c2, rmask)
        vpp1 = _rnd(_rnd(gg1 * vf1, ROUND) + _rnd(omg1 * vc1, ROUND), ROUND)
        vpp2 = _rnd(_rnd(gg2 * vf2, ROUND) + _rnd(omg2 * vc2, ROUND), ROUND)
        raw1 = (c1 < dcut)[None, :]
        raw2 = (c2 < dcut)[None, :]
        vpp1 = tl.where(raw1, vf1, vpp1)
        vpp2 = tl.where(raw2, vf2, vpp2)
        dV1 = _ld(dv_p, rows, s_dvm, c1, rmask)
        dV2 = _ld(dv_p, rows, s_dvm, c2, rmask)
        if CLAMP:
            n = tl.sqrt(tl.sum(vpp1 * vpp1, axis=1) + tl.sum(vpp2 * vpp2, axis=1))
            nn = tl.maximum(n, 1e-20)
            scale = tl.minimum(tau / nn, 1.0)
            dsc = tl.sum(dV1 * vpp1, axis=1) + tl.sum(dV2 * vpp2, axis=1)
            dn = tl.where((tau / nn) <= 1.0, dsc * (-tau / (nn * nn)), 0.0)
            dn = tl.where(n > 0.0, dn / tl.maximum(n, 1e-30), 0.0)          # d n / d v = v / n
            dvpp1 = _rnd(dV1 * scale[:, None] + dn[:, None] * vpp1, ROUND)
            dvpp2 = _rnd(dV2 * scale[:, None] + dn[:, None] * vpp2, ROUND)
        else:
            dvpp1 = dV1
            dvpp2 = dV2
        db1 = tl.where(raw1, 0.0, dvpp1)                                     # blend part (raw cols: none)
        db2 = tl.where(raw2, 0.0, dvpp2)
        _st(dvf_p, rows, D, c1, rmask, tl.where(raw1, dvpp1, _rnd(db1 * gg1, ROUND)))
        _st(dvf_p, rows, D, c2, rmask, tl.where(raw2, dvpp2, _rnd(db2 * gg2, ROUND)))
        _st(dvc_p, rows, D, c1, rmask, _rnd(db1 * omg1, ROUND))
        _st(dvc_p, rows, D, c2, rmask, _rnd(db2 * omg2, ROUND))
        dgg1 = db1 * vf1 - db1 * vc1
        dgg2 = db2 * vf2 - db2 * vc2
        dz1 = dgg1 * (s1 * (1.0 - s1))
        dz2 = dgg2 * (s2 * (1.0 - s2))
        _st(d2_p, rows, D, c1, rmask, dz1)
        _st(d2_p, rows, D, c2, rmask, dz2)
        tl.store(pgv + c1, tl.sum(dz1, axis=0))
        tl.store(pgv + c2, tl.sum(dz2, axis=0))


@triton.jit
def _vin_bwd(up_ptr, s_upg, s_ups, bvg_ptr, dvin_ptr, dup_ptr, s_dupg, s_dups, pbias_ptr, M, D, NMT,
             BM: tl.constexpr, BN: tl.constexpr, ROUND: tl.constexpr):
    """Backward of _vin_fwd: dv_in -> d up[3] (gate logits), d up[4] (tanh input), bias_vg partials."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    g = tl.program_id(2)
    rows = pid_m * BM + tl.arange(0, BM)
    rmask = rows < M
    cols = pid_n * BN + tl.arange(0, BN)
    vg = _ld(up_ptr + g * s_upg + 3 * s_ups, rows, D, cols, rmask)
    vm = _ld(up_ptr + g * s_upg + 4 * s_ups, rows, D, cols, rmask)
    dvin = _ld(dvin_ptr + g * M * D, rows, D, cols, rmask)
    b = tl.load(bvg_ptr + g * D + cols).to(tl.float32)
    sf = tl.sigmoid(vg + b[None, :])
    s = _rnd(sf, ROUND)
    t = _rnd(libdevice.tanh(vm), ROUND)
    ds = _rnd(dvin * t, ROUND)
    dt = _rnd(dvin * s, ROUND)
    dz = ds * (sf * (1.0 - sf))
    _st(dup_ptr + g * s_dupg + 3 * s_dups, rows, D, cols, rmask, dz)
    _st(dup_ptr + g * s_dupg + 4 * s_dups, rows, D, cols, rmask, dt * (1.0 - t * t))
    tl.store(pbias_ptr + (g * NMT + pid_m) * D + cols, tl.sum(dz, axis=0))


@triton.jit
def _u_bwd(o_ptr, s_og, s_om, rstd_ptr, w_ptr, du_ptr, do_ptr, pw_ptr, M, D, NP,
           ROWS: tl.constexpr, BD: tl.constexpr):
    """rms backward of u = rms(o) * w given du (= lam*dv_in + dX(ud2) + dX(ga2)[:, D:], formed
    by one K-concat GEMM): do = rstd * (du*w - xn * mean(du*w*xn)), partial d w [G, NP, D]."""
    pid = tl.program_id(0)
    g = tl.program_id(1)
    rows = pid * ROWS + tl.arange(0, ROWS)
    rmask = rows < M
    rstd = tl.load(rstd_ptr + g * M + rows, mask=rmask, other=0.0)
    base_d = g * M * D + rows[:, None] * D
    acc = tl.zeros([ROWS], dtype=tl.float32)
    for d0 in range(0, D, BD):
        cols = d0 + tl.arange(0, BD)
        cm = cols < D
        m = rmask[:, None] & cm[None, :]
        o = tl.load(o_ptr + g * s_og + rows[:, None] * s_om + cols[None, :], mask=m, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + g * D + cols, mask=cm, other=0.0).to(tl.float32)
        du = tl.load(du_ptr + base_d + cols[None, :], mask=m, other=0.0).to(tl.float32)
        xn = o * rstd[:, None]
        tl.store(pw_ptr + (g * NP + pid) * D + cols, tl.sum(du * xn, axis=0), mask=cm)
        acc += tl.sum((du * w[None, :]) * xn, axis=1)
    mdot = acc / D
    for d0 in range(0, D, BD):
        cols = d0 + tl.arange(0, BD)
        cm = cols < D
        m = rmask[:, None] & cm[None, :]
        o = tl.load(o_ptr + g * s_og + rows[:, None] * s_om + cols[None, :], mask=m, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + g * D + cols, mask=cm, other=0.0).to(tl.float32)
        du = tl.load(du_ptr + base_d + cols[None, :], mask=m, other=0.0).to(tl.float32)
        xn = o * rstd[:, None]
        do = rstd[:, None] * (du * w[None, :] - xn * mdot[:, None])
        tl.store(do_ptr + base_d + cols[None, :], do.to(do_ptr.dtype.element_ty), mask=m)


# ============================================================================ autograd glue
def _s2(t):
    """(batch stride, row stride) of a [G, M, cols] tensor with unit inner stride."""
    assert t.stride(-1) == 1, t.stride()
    return t.stride(0), t.stride(1)


def _colsum(p):
    """[G, n, D] fp32 partials -> [G, D]: a ones-GEMV (torch's middle-dim sum ran at <1 TB/s)."""
    G, n, D = p.shape
    ones = torch.ones(G, 1, n, dtype=p.dtype, device=p.device)
    return torch.bmm(ones, p).view(G, D)


def _rows3(t, G, M, D):
    t = t.reshape(G, M, D)
    return t if t.stride(-1) == 1 else t.contiguous()


class _RefineFused(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xb, o, kf, vf, cos, sin, w_ctx, ga2, ud2, vgd, ups5, b_gk, b_gv, b_vg, k_gain,
                wvd, wvu, h, dh, dcut, tau, lam, rows_dims):
        c = CFG
        G, M, D = o.shape
        R = ud2.shape[1] // 2
        assert D % dh == 0 and dh % 2 == 0 and h % c["NHG"] == 0 and D % c["BN_VIN"] == 0, (D, dh, h, c)
        assert ups5.shape == (G, 5, D, R) and ups5.is_contiguous()
        Wv, Bsz = rows_dims
        assert Wv * Bsz == M and cos.shape == (G, Wv, 1, 1, dh // 2) and cos.is_contiguous() and sin.is_contiguous()
        act = kf.dtype
        rnd = act == torch.bfloat16 and bool(c["ROUND_EMU"])
        dev = o.device
        # 1. g_in = [xb | u], rstd
        g_in = torch.empty(G, M, 2 * D, dtype=act, device=dev)
        rstd = torch.empty(G, M, dtype=torch.float32, device=dev)
        _rms_cat_fwd[(triton.cdiv(M, c["ROWS_RMS"]), G)](
            o, xb, w_ctx, g_in, rstd, M, D, *_s2(o), *_s2(xb), g_in.stride(0), g_in.stride(1),
            X.EPS_NORM, ROWS=c["ROWS_RMS"], BD=c["BD_RMS"], num_warps=c["WARPS_RMS"])
        u = g_in[:, :, D:]
        # 2. LoRA downs + the five ups (cuBLAS; == wave3d_ops.refine)
        gd = X.lin(g_in, ga2)                              # [G, M, 2R] = [gk_d | gv_d]
        ud = X.lin(u, ud2)                                 # [G, M, 2R] = [k_d | vm_d]
        vg_d = X.lin(xb, vgd)                              # [G, M, R]
        assert R == 144, R
        down5 = torch.empty(G, 5, M, R, dtype=act, device=dev)
        _pack_down5[(triton.cdiv(M, 64), G)](ud, *_s2(ud), gd, *_s2(gd), vg_d, *_s2(vg_d),
                                               down5, down5.stride(0), down5.stride(1), M, R, BM=64, num_warps=4)
        up5 = torch.bmm(down5.view(G * 5, M, R), ups5.view(G * 5, D, R).transpose(1, 2)).view(G, 5, M, D)
        # 3. v_in -> vctx
        v_in = torch.empty(G, M, D, dtype=act, device=dev)
        _vin_fwd[(triton.cdiv(M, c["BM_VIN"]), D // c["BN_VIN"], G)](
            up5, up5.stride(0), up5.stride(1), b_vg, u, *_s2(u), v_in, M, D, lam,
            BM=c["BM_VIN"], BN=c["BN_VIN"], ROUND=rnd, num_warps=c["WARPS_VIN"])
        vctx_d = X.lin(v_in, wvd)                          # [G, M, R]
        v_ctx = X.lin(vctx_d, wvu)                         # [G, M, D]
        # 4. K / V
        K = torch.empty(G, M, D, dtype=act, device=dev)
        V = torch.empty(G, M, D, dtype=act, device=dev)
        _kv_fwd[(triton.cdiv(M, c["BM_HEAD"]), h // c["NHG"], G)](
            up5, up5.stride(0), up5.stride(1), v_ctx, k_gain, b_gk, b_gv, kf, *_s2(kf), vf, *_s2(vf),
            cos, sin, cos.stride(0), K, V, M, D, h, Bsz, dcut, tau, X.EPS_HEAD,
            BM=c["BM_HEAD"], DH=dh, NHG=c["NHG"], ROUND=rnd, CLAMP=tau > 0, num_warps=c["WARPS_HEAD"])
        ctx.save_for_backward(xb, o, kf, vf, cos, sin, g_in, rstd, down5, up5, v_in, vctx_d, v_ctx,
                              w_ctx, ga2, ud2, vgd, ups5, b_gk, b_gv, b_vg, k_gain, wvd, wvu)
        ctx.meta = (h, dh, dcut, tau, lam, rows_dims, rnd)
        return K.view(G, Wv, Bsz, h, dh), V.view(G, Wv, Bsz, h, dh)

    @staticmethod
    def backward(ctx, dK, dV):
        (xb, o, kf, vf, cos, sin, g_in, rstd, down5, up5, v_in, vctx_d, v_ctx,
         w_ctx, ga2, ud2, vgd, ups5, b_gk, b_gv, b_vg, k_gain, wvd, wvu) = ctx.saved_tensors
        h, dh, dcut, tau, lam, rows_dims, rnd = ctx.meta
        c = CFG
        G, M, D = o.shape
        R = down5.shape[-1]
        Wv, Bsz = rows_dims
        act = kf.dtype
        dev = o.device
        f32 = dict(dtype=torch.float32, device=dev)
        dK = _rows3(dK, G, M, D)
        dV = _rows3(dV, G, M, D)
        u = g_in[:, :, D:]
        # 1. K/V chain backward -> d up5[0:3], dv_ctx, dkf, dvf
        nmt = triton.cdiv(M, c["BM_HEAD"])
        d_up = torch.empty(G, 5, M, D, dtype=act, device=dev)
        dv_ctx = torch.empty(G, M, D, dtype=act, device=dev)
        dkf = torch.empty(G, M, D, dtype=act, device=dev)
        dvf = torch.empty(G, M, D, dtype=act, device=dev)
        p_gk = torch.empty(G, nmt, D, **f32)
        p_gv = torch.empty(G, nmt, D, **f32)
        p_gain = torch.empty(G, h, M, **f32)
        _kv_bwd[(nmt, h // c["NHG"], G)](
            up5, up5.stride(0), up5.stride(1), v_ctx, k_gain, b_gk, b_gv, kf, *_s2(kf), vf, *_s2(vf),
            cos, sin, cos.stride(0), dK, *_s2(dK), dV, *_s2(dV),
            d_up, d_up.stride(0), d_up.stride(1), dv_ctx, dkf, dvf, p_gain, p_gk, p_gv,
            M, D, h, nmt, Bsz, dcut, tau, X.EPS_HEAD,
            BM=c["BM_HEAD"], DH=dh, NHG=c["NHG"], ROUND=rnd, CLAMP=tau > 0, num_warps=c["WARPS_HEAD"])
        # 2. vctx up / down (cuBLAS)
        dvctx_d = torch.bmm(dv_ctx, wvu)                                   # [G, M, R]
        d_wvu = torch.bmm(dv_ctx.transpose(1, 2), vctx_d)                  # [G, D, R]
        dvin = torch.bmm(dvctx_d, wvd)                                     # [G, M, D]
        d_wvd = torch.bmm(dvctx_d.transpose(1, 2), v_in)                   # [G, R, D]
        # 3. v_in chain backward -> d up5[3:5]
        nmt2 = triton.cdiv(M, c["BM_VIN"])
        p_vg = torch.empty(G, nmt2, D, **f32)
        _vin_bwd[(nmt2, D // c["BN_VIN"], G)](
            up5, up5.stride(0), up5.stride(1), b_vg, dvin, d_up, d_up.stride(0), d_up.stride(1), p_vg, M, D, nmt2,
            BM=c["BM_VIN"], BN=c["BN_VIN"], ROUND=rnd, num_warps=c["WARPS_VIN"])
        # 4. the five ups: ONE dX and ONE dW GEMM over batch 5G; gather dX into [d_ud | d_gd | d_vgx]
        d_up_f = d_up.view(G * 5, M, D)
        d_down5 = torch.bmm(d_up_f, ups5.view(G * 5, D, R))                                   # [5G, M, R]
        d_ups5 = torch.bmm(d_up_f.transpose(1, 2), down5.view(G * 5, M, R)).view(G, 5, D, R)
        dd = torch.empty(G, M, 5 * R, dtype=act, device=dev)
        _gather_ddown[(triton.cdiv(M, 64), G)](d_down5, 5 * M * R, M * R, R, dd, dd.stride(0), dd.stride(1), M, R,
                                                 BM=64, num_warps=4)
        d_ud, d_gd, d_vgx = dd[:, :, :2 * R], dd[:, :, 2 * R:4 * R], dd[:, :, 4 * R:]
        # 5. downs. dX K-concatenated so du / dxb come straight out of cuBLAS:
        #    du  = lam*dv_in + [d_ud | d_gd] @ [ud2 ; ga2_u]   (beta=lam, in place on dv_in)
        #    dxb =             [d_gd | d_vgx] @ [ga2_x ; vgd]
        W_u = torch.empty(G, 4 * R, D, dtype=ud2.dtype, device=dev)
        W_u[:, :2 * R].copy_(ud2)
        W_u[:, 2 * R:].copy_(ga2[:, :, D:])
        W_x = torch.empty(G, 3 * R, D, dtype=ud2.dtype, device=dev)
        W_x[:, :2 * R].copy_(ga2[:, :, :D])
        W_x[:, 2 * R:].copy_(vgd)
        du = dvin.baddbmm_(dd[:, :, :4 * R], W_u, beta=lam)
        dxb = torch.bmm(dd[:, :, 2 * R:], W_x)
        d_ud2 = torch.bmm(d_ud.transpose(1, 2), u)                                             # [G, 2R, D]
        d_ga2 = torch.bmm(d_gd.transpose(1, 2), g_in)                                          # [G, 2R, 2D]
        d_vgd = torch.bmm(d_vgx.transpose(1, 2), xb)                                           # [G, R, D]
        # 6. u = rms(o) backward
        do = torch.empty(G, M, D, dtype=act, device=dev)
        n_p = triton.cdiv(M, c["ROWS_RMS"])
        p_w = torch.empty(G, n_p, D, **f32)
        _u_bwd[(n_p, G)](o, *_s2(o), rstd, w_ctx, du, do, p_w, M, D, n_p,
                         ROWS=c["ROWS_RMS"], BD=c["BD_RMS"], num_warps=c["WARPS_RMS"])
        d_wctx = _colsum(p_w).to(w_ctx.dtype)
        d_bgk = _colsum(p_gk).to(b_gk.dtype)
        d_bgv = _colsum(p_gv).to(b_gv.dtype)
        d_bvg = _colsum(p_vg).to(b_vg.dtype)
        d_gain = torch.bmm(p_gain, torch.ones(G, M, 1, dtype=p_gain.dtype, device=dev)).view(G, h).to(k_gain.dtype)
        return (dxb, do, dkf, dvf, None, None, d_wctx, d_ga2, d_ud2, d_vgd, d_ups5, d_bgk, d_bgv, d_bvg, d_gain,
                d_wvd, d_wvu, None, None, None, None, None, None)


def refine_fused(P, xb, o, kf, vf, cos, sin, h, dh, dcut, tau, lam, rows_dims):
    """Drop-in for wave3d_ops.refine (same signature / outputs / grads)."""
    P = X.combine_params(P)
    return _RefineFused.apply(xb, o, kf, vf, cos, sin, P["attn.norm_ctx.weight"], P["attn.ga2"], P["attn.ud2"],
                              P["attn.v_gate.down.weight"], P["attn.ups5"], P["attn.w_gk.b.bias"],
                              P["attn.w_gv.b.bias"], P["attn.v_gate.up.bias"], P["attn.k_gain"],
                              P["attn.w_vctx.down.weight"], P["attn.w_vctx.up.weight"],
                              h, dh, dcut, float(tau), float(lam), tuple(rows_dims))
