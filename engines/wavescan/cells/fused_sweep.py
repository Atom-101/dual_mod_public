"""Fused sweep path ("Phase D v1 without persistence", megakernel_phaseD_prep §2).

Motivation (flagship bench): at d1024/L24/T4096 the step executes ~49k serial
sweeps; the torch/inductor cell spends ~15 kernels + redundant gate GEMMs per
sweep and lands ~25x off vanilla. This module reduces a sweep to 3 launches:

    K1  _attend_norm_kernel   intra scores + online-softmax merge against the
                              per-cell committed accumulator + raw self column,
                              then RMSNorm -> u rows (o kept in a scratch buf)
    G   one cuBLAS GEMM       u @ [W_kctx | W_vctx | W_gk_u | W_gv_u]^T
                              (the x-hat half of the gate logits is
                              sweep-invariant, hoisted per cell)
    K2  _refine2_kernel       sigmoid(glog_x + glog_u), convex blend, RoPE,
                              strided cache-row write

ctx_mod variants (mult/mult_res/fusion) extend the pack with a 5th row block
(the u-driven variant pre-GEMM) and add exactly one fused elementwise kernel
+ the w_vctx GEMM per sweep; every xhat-only term (sigmoid(v_gate(xhat)),
v_pre xhat-half) is hoisted to the per-layer projection pass, its backward
chain deferred to layer_close. The kmod v-gate pin is a constexpr in K2.

The backward is a hand-orchestrated reverse sweep loop (recompute sweep
states, then per sweep: K2 bwd kernel, GEMM backwards, K1 bwd kernel) — no
autograd inside the loop. Gradients through the online-softmax running max
cancel (scale invariance), exactly as in flash backward.

Numerics: scores/merge/RMSNorm/sigmoid in fp32, storage bf16 — the same
contract as the reference cell; fp32 runs are bit-comparable (tested).
"""

import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import triton.language.extra.libdevice as libdevice

from models.dualmod.rope import apply_rope
from .scan_cell_fwd import _GATE_PIN


def _accum_ctx_dW(grads, mat, dW, factored):
    """Accumulate a full [d_out, d_in] dW into a ctx operator's grad buffer.
    factored (ctx_rank>0): mat is LoRA-factored (W = up@down) — project the
    composed dW onto the factors (d_down = up^T @ dW, d_up = dW @ down^T),
    mirroring the LoRA-gate chain. Else straight into the dense weight buffer
    (byte-identical to the pre-ctx_rank path)."""
    if factored:
        up = mat.up.weight.float()
        down = mat.down.weight.float()
        dW = dW.float()
        grads.buf[id(mat.down.weight)] += up.t() @ dW
        grads.buf[id(mat.up.weight)] += dW @ down.t()
    else:
        grads.buf[id(mat.weight)] += dW


# ---------------------------------------------------------------------------
# K1: intra attend + merge (+ per-head store to o scratch) + RMSNorm -> u
# ---------------------------------------------------------------------------

@triton.jit
def _attend_norm_kernel(
    q_ptr, vraw_ptr,                        # layer bufs [B,h,T,dh], base at pos0
    kin_ptr, vin_ptr,                       # cache rows base at pos0, strided
    sself_ptr,                              # layer buf [B,h,T] fp32, base at pos0
    A_ptr, Z_ptr, M_ptr,                    # [B,h,c,dh] fp32, [B,h,c] fp32 x2
    rmsw_ptr,                               # [d]
    u_ptr, o_ptr,                           # [B,c,d]; o doubles as scratch
    scale, eps,
    stride_cb, stride_ch, stride_ct,
    q_sb, q_sh,                             # strides of q/vraw layer bufs (row=dh)
    ss_sb, ss_sh,                           # strides of s_self layer buf
    h: tl.constexpr, c: tl.constexpr, dh: tl.constexpr, d: tl.constexpr,
    R: tl.constexpr, HAS_COMM: tl.constexpr, OUT_BF16: tl.constexpr,
    IEEE: tl.constexpr, DP: tl.constexpr,
):
    pid = tl.program_id(0)
    n_tiles = c // R
    b = pid // n_tiles
    rt = pid % n_tiles
    rows = rt * R + tl.arange(0, R)
    dd = tl.arange(0, dh)
    cc = tl.arange(0, c)

    for hh in tl.static_range(h):
        qb = q_ptr + b * q_sb + hh * q_sh
        q = tl.load(qb + rows[:, None] * dh + dd[None, :]).to(tl.float32)
        cb = kin_ptr + b * stride_cb + hh * stride_ch
        ki = tl.load(cb + cc[:, None] * stride_ct + dd[None, :]).to(tl.float32)
        if IEEE:
            s = tl.dot(q, tl.trans(ki), input_precision="ieee") * scale
        else:
            s = tl.dot(q, tl.trans(ki)) * scale
        s = tl.where(cc[None, :] < rows[:, None], s, float("-inf"))
        s_self = tl.load(sself_ptr + b * ss_sb + hh * ss_sh + rows)
        m_tail = tl.maximum(tl.max(s, axis=1), s_self)
        if HAS_COMM:
            m_comm = tl.load(M_ptr + (b * h + hh) * c + rows)
            m = tl.maximum(m_tail, m_comm)
            w = tl.exp(m_comm - m)
            Zc = tl.load(Z_ptr + (b * h + hh) * c + rows)
            A = tl.load(A_ptr + ((b * h + hh) * c) * dh
                        + rows[:, None] * dh + dd[None, :])
        else:
            m = m_tail
            w = tl.zeros((R,), dtype=tl.float32)
            Zc = tl.zeros((R,), dtype=tl.float32)
            A = tl.zeros((R, dh), dtype=tl.float32)
        e = tl.exp(s - m[:, None])
        es = tl.exp(s_self - m)
        Z = Zc * w + tl.sum(e, axis=1) + es
        vb = vin_ptr + b * stride_cb + hh * stride_ch
        vi = tl.load(vb + cc[:, None] * stride_ct + dd[None, :]).to(tl.float32)
        vr = tl.load(vraw_ptr + b * q_sb + hh * q_sh
                     + rows[:, None] * dh + dd[None, :]).to(tl.float32)
        if IEEE:
            ev = tl.dot(e, vi, input_precision="ieee")
        else:
            ev = tl.dot(e, vi)
        o_h = (A * w[:, None] + ev + es[:, None] * vr) / Z[:, None]
        # per-head store into the o scratch row segment (fp32 scratch)
        tl.store(o_ptr + (b * c + rows[:, None]) * d + hh * dh + dd[None, :],
                 o_h)
    # reload the full-width rows (same program wrote them; L2-hot) + RMSNorm.
    # The reload crosses warps within the program (per-head stores use a
    # different thread mapping than the full-row load): a CTA barrier is
    # REQUIRED for visibility — without it this races nondeterministically.
    tl.debug_barrier()
    col = tl.arange(0, DP)                # DP = next_power_of_2(d); pad lanes
    cm = col < d                          # masked (other=0.0: sum-neutral,
    o_full = tl.load(o_ptr + (b * c + rows[:, None]) * d + col[None, :],
                     mask=cm[None, :], other=0.0)
    ms = tl.sum(o_full * o_full, axis=1) / d      # explicit /d, NOT /DP
    rs = 1.0 / tl.sqrt(ms + eps)
    wgt = tl.load(rmsw_ptr + col, mask=cm, other=0.0).to(tl.float32)
    u = o_full * rs[:, None] * wgt[None, :]
    if OUT_BF16:
        tl.store(u_ptr + (b * c + rows[:, None]) * d + col[None, :],
                 u.to(tl.bfloat16), mask=cm[None, :])
    else:
        tl.store(u_ptr + (b * c + rows[:, None]) * d + col[None, :], u,
                 mask=cm[None, :])


# ---------------------------------------------------------------------------
# Committed-prefix accumulator, flash-style (no [c,a] materialization).
# fwd: A/Z/m/argmax per (b,h) in one kernel. bwd: the full merge backward —
# dV_gc/dK_gc tiles owned exclusively per (b,h), dq accumulated, dm residual
# (running-max subgradient) exported for a tiny torch-side argmax correction.
# ---------------------------------------------------------------------------

@triton.jit
def _comm_fwd_kernel(
    q_ptr, kc_ptr, vc_ptr,
    A_ptr, Z_ptr, M_ptr, AM_ptr,
    scale, a,
    stride_cb, stride_ch, stride_ct,
    q_sb, q_sh,
    h: tl.constexpr, c: tl.constexpr, dh: tl.constexpr,
    RC: tl.constexpr, TA: tl.constexpr, IEEE: tl.constexpr,
):
    pid = tl.program_id(0)
    nt = c // RC
    b = pid // (h * nt)
    hh = (pid // nt) % h
    rt = pid % nt
    rows = rt * RC + tl.arange(0, RC)
    dd = tl.arange(0, dh)
    ta = tl.arange(0, TA)
    q = tl.load(q_ptr + b * q_sb + hh * q_sh
                + rows[:, None] * dh + dd[None, :]).to(tl.float32)
    kb = kc_ptr + b * stride_cb + hh * stride_ch
    vb = vc_ptr + b * stride_cb + hh * stride_ch
    m_run = tl.full((RC,), float("-inf"), dtype=tl.float32)
    Z_run = tl.zeros((RC,), dtype=tl.float32)
    A_run = tl.zeros((RC, dh), dtype=tl.float32)
    am = tl.zeros((RC,), dtype=tl.int32)
    for t0 in range(0, a, TA):
        msk = (t0 + ta) < a
        k = tl.load(kb + (t0 + ta)[:, None] * stride_ct + dd[None, :],
                    mask=msk[:, None], other=0.0).to(tl.float32)
        if IEEE:
            s = tl.dot(q, tl.trans(k), input_precision="ieee") * scale
        else:
            s = tl.dot(q, tl.trans(k)) * scale
        s = tl.where(msk[None, :], s, float("-inf"))
        m_t = tl.max(s, axis=1)
        am_t = t0 + tl.argmax(s, axis=1)
        am = tl.where(m_t > m_run, am_t.to(tl.int32), am)
        m_new = tl.maximum(m_run, m_t)
        alpha = tl.exp(m_run - m_new)
        e = tl.exp(s - m_new[:, None])
        Z_run = Z_run * alpha + tl.sum(e, axis=1)
        v = tl.load(vb + (t0 + ta)[:, None] * stride_ct + dd[None, :],
                    mask=msk[:, None], other=0.0).to(tl.float32)
        if IEEE:
            ev = tl.dot(e, v, input_precision="ieee")
        else:
            ev = tl.dot(e, v)
        A_run = A_run * alpha[:, None] + ev
        m_run = m_new
    hb = (b * h + hh) * c
    tl.store(A_ptr + hb * dh + rows[:, None] * dh + dd[None, :], A_run)
    tl.store(Z_ptr + hb + rows, Z_run)
    tl.store(M_ptr + hb + rows, m_run)
    tl.store(AM_ptr + hb + rows, am)


@triton.jit
def _comm_bwd_kernel(
    q_ptr, kc_ptr, vc_ptr, M_ptr,
    dA_ptr, dZ_ptr, dM_ptr,
    dq_ptr, dkc_ptr, dvc_ptr, dmt_ptr,     # dq/dkc/dvc ACCUM into layer/grad-cache bufs
    scale, a,
    stride_cb, stride_ch, stride_ct,
    gc_sb, gc_sh,                           # grad-cache strides (dkc/dvc), row stride = dh? no: stride_ct reused
    q_sb, q_sh,
    h: tl.constexpr, c: tl.constexpr, dh: tl.constexpr,
    RC: tl.constexpr, TA: tl.constexpr, IEEE: tl.constexpr,
):
    pid = tl.program_id(0)
    nt = c // RC
    b = pid // (h * nt)
    hh = (pid // nt) % h
    rt = pid % nt
    rows = rt * RC + tl.arange(0, RC)
    dd = tl.arange(0, dh)
    ta = tl.arange(0, TA)
    hb = (b * h + hh) * c
    q = tl.load(q_ptr + b * q_sb + hh * q_sh
                + rows[:, None] * dh + dd[None, :]).to(tl.float32)
    m = tl.load(M_ptr + hb + rows)
    dA = tl.load(dA_ptr + hb * dh + rows[:, None] * dh + dd[None, :])
    dZ = tl.load(dZ_ptr + hb + rows)
    dM = tl.load(dM_ptr + hb + rows)
    kb = kc_ptr + b * stride_cb + hh * stride_ch
    vb = vc_ptr + b * stride_cb + hh * stride_ch
    dkb = dkc_ptr + b * gc_sb + hh * gc_sh
    dvb = dvc_ptr + b * gc_sb + hh * gc_sh
    dq_acc = tl.zeros((RC, dh), dtype=tl.float32)
    dsum = tl.zeros((RC,), dtype=tl.float32)
    for t0 in range(0, a, TA):
        msk = (t0 + ta) < a
        k = tl.load(kb + (t0 + ta)[:, None] * stride_ct + dd[None, :],
                    mask=msk[:, None], other=0.0).to(tl.float32)
        v = tl.load(vb + (t0 + ta)[:, None] * stride_ct + dd[None, :],
                    mask=msk[:, None], other=0.0).to(tl.float32)
        if IEEE:
            s = tl.dot(q, tl.trans(k), input_precision="ieee") * scale
            deV = tl.dot(dA, tl.trans(v), input_precision="ieee")
        else:
            s = tl.dot(q, tl.trans(k)) * scale
            deV = tl.dot(dA, tl.trans(v))
        e = tl.exp(s - m[:, None])
        e = tl.where(msk[None, :], e, 0.0)
        de = deV + dZ[:, None]
        ds = e * de
        dsum += tl.sum(ds, axis=1)
        if IEEE:
            dq_acc += tl.dot(ds, k, input_precision="ieee") * scale
            dK_t = tl.dot(tl.trans(ds), q, input_precision="ieee") * scale
            dV_t = tl.dot(tl.trans(e), dA, input_precision="ieee")
        else:
            dq_acc += tl.dot(ds, k) * scale
            dK_t = tl.dot(tl.trans(ds), q) * scale
            dV_t = tl.dot(tl.trans(e), dA)
        kaddr = dkb + (t0 + ta)[:, None] * stride_ct + dd[None, :]
        vaddr = dvb + (t0 + ta)[:, None] * stride_ct + dd[None, :]
        if nt == 1:
            prev = tl.load(kaddr, mask=msk[:, None], other=0.0)
            tl.store(kaddr, prev + dK_t, mask=msk[:, None])
            prev = tl.load(vaddr, mask=msk[:, None], other=0.0)
            tl.store(vaddr, prev + dV_t, mask=msk[:, None])
        else:
            tl.atomic_add(kaddr, dK_t, mask=msk[:, None])
            tl.atomic_add(vaddr, dV_t, mask=msk[:, None])
    ob = b * q_sb + hh * q_sh + rows[:, None] * dh + dd[None, :]
    prev = tl.load(dq_ptr + ob)
    tl.store(dq_ptr + ob, prev + dq_acc)
    tl.store(dmt_ptr + hb + rows, dM - dsum)


@triton.jit
def _comm_bwd_det_kernel(
    q_ptr, kc_ptr, vc_ptr, M_ptr,
    dA_ptr, dZ_ptr, dM_ptr,
    dq_ptr, dkc_ptr, dvc_ptr, dmt_ptr,     # dkc/dvc are COMPACT [NT, B, h, a, dh]
    scale, a,
    stride_cb, stride_ch, stride_ct,       # cache (kc/vc) strides
    gc_sb, gc_sh, gc_st, plane,            # det-buffer b/h/key strides + plane stride
    q_sb, q_sh,
    h: tl.constexpr, c: tl.constexpr, dh: tl.constexpr,
    RC: tl.constexpr, TA: tl.constexpr, IEEE: tl.constexpr,
):
    """DETERMINISTIC twin of _comm_bwd_kernel for the split-chunk case (NT>1).
    dq/dmt are row-owned (one program per query-row tile) -> plain load+add.
    The dK/dV cache-grad tiles are the raced output: each row tile rt writes
    its full contribution (over all committed key positions, disjoint key
    tiles within the program) to its OWN compact plane dkc[rt] via plain
    STORE (no read-modify-write, no atomic); the caller reduces over rt in
    fixed order and adds into the real grad cache."""
    pid = tl.program_id(0)
    nt = c // RC
    b = pid // (h * nt)
    hh = (pid // nt) % h
    rt = pid % nt
    rows = rt * RC + tl.arange(0, RC)
    dd = tl.arange(0, dh)
    ta = tl.arange(0, TA)
    hb = (b * h + hh) * c
    q = tl.load(q_ptr + b * q_sb + hh * q_sh
                + rows[:, None] * dh + dd[None, :]).to(tl.float32)
    m = tl.load(M_ptr + hb + rows)
    dA = tl.load(dA_ptr + hb * dh + rows[:, None] * dh + dd[None, :])
    dZ = tl.load(dZ_ptr + hb + rows)
    dM = tl.load(dM_ptr + hb + rows)
    kb = kc_ptr + b * stride_cb + hh * stride_ch
    vb = vc_ptr + b * stride_cb + hh * stride_ch
    dkb = dkc_ptr + rt * plane + b * gc_sb + hh * gc_sh
    dvb = dvc_ptr + rt * plane + b * gc_sb + hh * gc_sh
    dq_acc = tl.zeros((RC, dh), dtype=tl.float32)
    dsum = tl.zeros((RC,), dtype=tl.float32)
    for t0 in range(0, a, TA):
        msk = (t0 + ta) < a
        k = tl.load(kb + (t0 + ta)[:, None] * stride_ct + dd[None, :],
                    mask=msk[:, None], other=0.0).to(tl.float32)
        v = tl.load(vb + (t0 + ta)[:, None] * stride_ct + dd[None, :],
                    mask=msk[:, None], other=0.0).to(tl.float32)
        if IEEE:
            s = tl.dot(q, tl.trans(k), input_precision="ieee") * scale
            deV = tl.dot(dA, tl.trans(v), input_precision="ieee")
        else:
            s = tl.dot(q, tl.trans(k)) * scale
            deV = tl.dot(dA, tl.trans(v))
        e = tl.exp(s - m[:, None])
        e = tl.where(msk[None, :], e, 0.0)
        de = deV + dZ[:, None]
        ds = e * de
        dsum += tl.sum(ds, axis=1)
        if IEEE:
            dq_acc += tl.dot(ds, k, input_precision="ieee") * scale
            dK_t = tl.dot(tl.trans(ds), q, input_precision="ieee") * scale
            dV_t = tl.dot(tl.trans(e), dA, input_precision="ieee")
        else:
            dq_acc += tl.dot(ds, k) * scale
            dK_t = tl.dot(tl.trans(ds), q) * scale
            dV_t = tl.dot(tl.trans(e), dA)
        kaddr = dkb + (t0 + ta)[:, None] * gc_st + dd[None, :]
        vaddr = dvb + (t0 + ta)[:, None] * gc_st + dd[None, :]
        tl.store(kaddr, dK_t, mask=msk[:, None])       # own plane, no atomic
        tl.store(vaddr, dV_t, mask=msk[:, None])
    ob = b * q_sb + hh * q_sh + rows[:, None] * dh + dd[None, :]
    prev = tl.load(dq_ptr + ob)
    tl.store(dq_ptr + ob, prev + dq_acc)
    tl.store(dmt_ptr + hb + rows, dM - dsum)


# ---------------------------------------------------------------------------
# K1 split variant: per-(b,head) attend (square tensor-core dots, 16x the
# program count of the fused kernel) + row-local RMSNorm. Faster than the
# monolithic _attend_norm_kernel because the head loop there serializes 16
# small dots inside 128 programs (latency-bound at ~0.5% of peak).
# ---------------------------------------------------------------------------

@triton.jit
def _attend_head_kernel(
    q_ptr, vraw_ptr, kin_ptr, vin_ptr, sself_ptr,
    A_ptr, Z_ptr, M_ptr, o_ptr,
    scale,
    stride_cb, stride_ch, stride_ct,
    q_sb, q_sh, ss_sb, ss_sh,
    h: tl.constexpr, c: tl.constexpr, dh: tl.constexpr, d: tl.constexpr,
    RC: tl.constexpr, HAS_COMM: tl.constexpr, IEEE: tl.constexpr,
):
    pid = tl.program_id(0)
    nt = c // RC
    b = pid // (h * nt)
    hh = (pid // nt) % h
    rt = pid % nt
    rows = rt * RC + tl.arange(0, RC)
    cols = tl.arange(0, c)
    dd = tl.arange(0, dh)
    lb = b * q_sb + hh * q_sh
    sb = b * ss_sb + hh * ss_sh
    q = tl.load(q_ptr + lb + rows[:, None] * dh + dd[None, :]).to(tl.float32)
    cb = kin_ptr + b * stride_cb + hh * stride_ch
    ki = tl.load(cb + cols[:, None] * stride_ct + dd[None, :]).to(tl.float32)
    if IEEE:
        s = tl.dot(q, tl.trans(ki), input_precision="ieee") * scale
    else:
        s = tl.dot(q, tl.trans(ki)) * scale
    s = tl.where(cols[None, :] < rows[:, None], s, float("-inf"))
    s_self = tl.load(sself_ptr + sb + rows)
    m_tail = tl.maximum(tl.max(s, axis=1), s_self)
    if HAS_COMM:
        hb = (b * h + hh) * c
        m_comm = tl.load(M_ptr + hb + rows)
        m = tl.maximum(m_tail, m_comm)
        w = tl.exp(m_comm - m)
        Zc = tl.load(Z_ptr + hb + rows)
        A = tl.load(A_ptr + hb * dh + rows[:, None] * dh + dd[None, :])
    else:
        m = m_tail
        w = tl.zeros((RC,), dtype=tl.float32)
        Zc = tl.zeros((RC,), dtype=tl.float32)
        A = tl.zeros((RC, dh), dtype=tl.float32)
    e = tl.exp(s - m[:, None])
    es = tl.exp(s_self - m)
    Z = Zc * w + tl.sum(e, axis=1) + es
    vb = vin_ptr + b * stride_cb + hh * stride_ch
    vi = tl.load(vb + cols[:, None] * stride_ct + dd[None, :]).to(tl.float32)
    vr = tl.load(vraw_ptr + lb + rows[:, None] * dh + dd[None, :]).to(tl.float32)
    if IEEE:
        ev = tl.dot(e, vi, input_precision="ieee")
    else:
        ev = tl.dot(e, vi)
    o_h = (A * w[:, None] + ev + es[:, None] * vr) / Z[:, None]
    tl.store(o_ptr + (b * c + rows[:, None]) * d + hh * dh + dd[None, :], o_h)


@triton.jit
def _rms_u_kernel(
    o_ptr, rmsw_ptr, u_ptr, eps,
    c: tl.constexpr, d: tl.constexpr, R: tl.constexpr, OUT_BF16: tl.constexpr,
    DP: tl.constexpr,
):
    pid = tl.program_id(0)
    n_tiles = c // R
    b = pid // n_tiles
    rt = pid % n_tiles
    rows = rt * R + tl.arange(0, R)
    col = tl.arange(0, DP)                # DP = next_power_of_2(d)
    cm = col < d
    base = (b * c + rows[:, None]) * d + col[None, :]
    o = tl.load(o_ptr + base, mask=cm[None, :], other=0.0)  # 0 = sum-neutral
    ms = tl.sum(o * o, axis=1) / d                # explicit /d, NOT /DP
    rs = 1.0 / tl.sqrt(ms + eps)
    wgt = tl.load(rmsw_ptr + col, mask=cm, other=0.0).to(tl.float32)
    u = o * rs[:, None] * wgt[None, :]
    if OUT_BF16:
        tl.store(u_ptr + base, u.to(tl.bfloat16), mask=cm[None, :])
    else:
        tl.store(u_ptr + base, u, mask=cm[None, :])


# ---------------------------------------------------------------------------
# Fused per-head RMS-norm + per-head gain (k_ctx / v_ctx norm), fwd + VJP.
# One program per (row, head); replaces the eager reshape/.float/.to/copy_
# flood (the K/V-norm 2x tax — see analysis/knorm_prof). fp32 accumulate,
# bf16 store, explicit /Dh. Parity twin of attention.norm_kctx / norm_vctx.
# ---------------------------------------------------------------------------
@triton.jit
def _ctxnorm_fwd_kernel(
    kc_ptr, gain_ptr, eps,
    SD: tl.constexpr, H: tl.constexpr, Dh: tl.constexpr,
    OUT_BF16: tl.constexpr, DP: tl.constexpr,
):
    """In-place per-head RMS+gain on the [N, d] block at cols [0, d) of a
    row-stride-SD tensor (d = H*Dh). y = gain * x * rsqrt(mean(x^2)+eps)."""
    pid = tl.program_id(0)
    row = pid // H
    head = pid % H
    col = tl.arange(0, DP)
    cm = col < Dh
    base = row * SD + head * Dh + col
    x = tl.load(kc_ptr + base, mask=cm, other=0.0).to(tl.float32)
    r = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / Dh + eps)
    g = tl.load(gain_ptr + head).to(tl.float32)
    y = x * r * g
    if OUT_BF16:
        tl.store(kc_ptr + base, y.to(tl.bfloat16), mask=cm)
    else:
        tl.store(kc_ptr + base, y, mask=cm)


@triton.jit
def _ctxnorm_bwd_kernel(
    x_ptr, dy_ptr, gain_ptr, dx_ptr, dg_ptr, eps,
    XSD: tl.constexpr, DSD: tl.constexpr,
    H: tl.constexpr, Dh: tl.constexpr,
    OUT_BF16: tl.constexpr, DP: tl.constexpr,
):
    """Closed-form VJP of the per-head RMS+gain. x = pre-norm candidate (row
    stride XSD); dy = grad wrt normed out (row stride DSD). dx -> dx_ptr (may
    alias dy_ptr, in place); per-(row,head) gain-grad partial r*<dy,x> -> dg_ptr.
      dx = gain * ( r*dy - r^3 * x * <dy,x>/Dh )"""
    pid = tl.program_id(0)
    row = pid // H
    head = pid % H
    col = tl.arange(0, DP)
    cm = col < Dh
    xb = row * XSD + head * Dh + col
    db = row * DSD + head * Dh + col
    x = tl.load(x_ptr + xb, mask=cm, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + db, mask=cm, other=0.0).to(tl.float32)
    g = tl.load(gain_ptr + head).to(tl.float32)
    r = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / Dh + eps)
    S = tl.sum(dy * x, axis=0)
    dx = g * (r * dy - (r * r * r) * x * S / Dh)
    tl.store(dg_ptr + pid, r * S)
    if OUT_BF16:
        tl.store(dx_ptr + db, dx.to(tl.bfloat16), mask=cm)
    else:
        tl.store(dx_ptr + db, dx, mask=cm)


# ---------------------------------------------------------------------------
# K2: sigmoid(glog_x + glog_u) + blend + rope + strided cache write
# (extension of triton_kernels._fused_refine_kernel with the split gate logit)
# ---------------------------------------------------------------------------

@triton.jit
def _refine2_kernel(
    k_ptr, v_ptr,                           # k/v layer bufs, base at pos0 (row stride kv_srow per b)
    ctxg_ptr,                               # [N, SD] = (kctx | vctx | glogu_k | glogu_v [| vmap_pre])
    vc_ptr,                                 # [N, d] variant v_ctx (VSEP; else unused, vctx read from ctxg)
    glogx_ptr,                              # glogx layer buf, base at pos0
    cos_ptr, sin_ptr, kout_ptr, vout_ptr,
    kv_sb, gx_sb,                           # batch strides of k/v and glogx layer bufs
    gate_pin, tau,
    d: tl.constexpr, dh: tl.constexpr, c: tl.constexpr,
    stride_out_b, stride_out_h, stride_out_t,
    OUT_BF16: tl.constexpr, HALF: tl.constexpr, SD: tl.constexpr,
    VSEP: tl.constexpr, NPIN: tl.constexpr, HALFP: tl.constexpr,
    CLAMP: tl.constexpr = False,
):
    n = tl.program_id(0)
    p = n % c
    b_idx = n // c
    krow = b_idx * kv_sb + p * d            # k/v row base
    gxrow = b_idx * gx_sb + p * 2 * d       # glogx row base
    lane = tl.arange(0, HALFP)              # HALFP = next_power_of_2(HALF)
    msk = lane < HALF                       # elementwise: other=0.0 safe
    hh = lane // (dh // 2)
    off = lane % (dh // 2)
    a_dim = hh * dh + off
    b_dim = a_dim + dh // 2

    k_a = tl.load(k_ptr + krow + a_dim, mask=msk, other=0.0).to(tl.float32)
    k_b = tl.load(k_ptr + krow + b_dim, mask=msk, other=0.0).to(tl.float32)
    kc_a = tl.load(ctxg_ptr + n * SD + a_dim, mask=msk, other=0.0).to(tl.float32)
    kc_b = tl.load(ctxg_ptr + n * SD + b_dim, mask=msk, other=0.0).to(tl.float32)
    gl_a = tl.load(ctxg_ptr + n * SD + 2 * d + a_dim, mask=msk,
                   other=0.0).to(tl.float32) \
        + tl.load(glogx_ptr + gxrow + a_dim, mask=msk, other=0.0).to(tl.float32)
    gl_b = tl.load(ctxg_ptr + n * SD + 2 * d + b_dim, mask=msk,
                   other=0.0).to(tl.float32) \
        + tl.load(glogx_ptr + gxrow + b_dim, mask=msk, other=0.0).to(tl.float32)
    g_a = tl.sigmoid(gl_a)
    g_b = tl.sigmoid(gl_b)
    kpp_a = g_a * k_a + (1.0 - g_a) * kc_a
    kpp_b = g_b * k_b + (1.0 - g_b) * kc_b
    cos = tl.load(cos_ptr + p * (dh // 2) + off, mask=msk, other=0.0)
    sin = tl.load(sin_ptr + p * (dh // 2) + off, mask=msk, other=0.0)
    rot_a = kpp_a * cos - kpp_b * sin
    rot_b = kpp_b * cos + kpp_a * sin

    v_a = tl.load(v_ptr + krow + a_dim, mask=msk, other=0.0).to(tl.float32)
    v_b = tl.load(v_ptr + krow + b_dim, mask=msk, other=0.0).to(tl.float32)
    if VSEP:
        vc_a = tl.load(vc_ptr + n * d + a_dim, mask=msk, other=0.0).to(tl.float32)
        vc_b = tl.load(vc_ptr + n * d + b_dim, mask=msk, other=0.0).to(tl.float32)
    else:
        vc_a = tl.load(ctxg_ptr + n * SD + d + a_dim, mask=msk,
                       other=0.0).to(tl.float32)
        vc_b = tl.load(ctxg_ptr + n * SD + d + b_dim, mask=msk,
                       other=0.0).to(tl.float32)
    gu_a = tl.load(ctxg_ptr + n * SD + 3 * d + a_dim, mask=msk,
                   other=0.0).to(tl.float32)
    gu_b = tl.load(ctxg_ptr + n * SD + 3 * d + b_dim, mask=msk,
                   other=0.0).to(tl.float32)
    if NPIN > 0:
        # kmod gate pin: v-gate u-logit -> +30 => g_v~1 => v'' == v_raw
        gu_a = tl.where(a_dim < NPIN, gate_pin, gu_a)
        gu_b = tl.where(b_dim < NPIN, gate_pin, gu_b)
    gvl_a = gu_a + tl.load(glogx_ptr + gxrow + d + a_dim, mask=msk,
                           other=0.0).to(tl.float32)
    gvl_b = gu_b + tl.load(glogx_ptr + gxrow + d + b_dim, mask=msk,
                           other=0.0).to(tl.float32)
    gv_a = tl.sigmoid(gvl_a)
    gv_b = tl.sigmoid(gvl_b)
    vpp_a = gv_a * v_a + (1.0 - gv_a) * vc_a
    vpp_b = gv_b * v_b + (1.0 - gv_b) * vc_b

    if CLAMP:
        # per-(position,head) entry-norm clamp: v'' <- v'' * min(1, tau/||v''||)
        # head lanes are the dh//2-block hh occupies; reduce over that block.
        HD2: tl.constexpr = dh // 2
        NHS: tl.constexpr = HALFP // HD2
        hn2 = tl.sum(tl.reshape(vpp_a * vpp_a + vpp_b * vpp_b, (NHS, HD2)),
                     axis=1)                                  # [NHS] head ||.||^2
        sc = tl.minimum(1.0, tau * tl.rsqrt(hn2 + 1e-20))     # [NHS]
        sc = tl.reshape(tl.broadcast_to(sc[:, None], (NHS, HD2)), (HALFP,))
        vpp_a = vpp_a * sc
        vpp_b = vpp_b * sc

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


# ---------------------------------------------------------------------------
# orchestration. Projections are computed ONCE PER LAYER into engine-owned
# buffers (batched GEMMs over full T; kernels read strided slices) — per-cell
# projection recompute + .contiguous() copies were ~30% of the flagship
# backward. Gradient accumulators are layer-shaped; ONE autograd region per
# layer closes attn_norm/W_Q/W_K/W_V/rope/s_self/glog_x. The committed-
# accumulator backward is hand-written (bf16 GEMMs, single exp) instead of
# autograd. Per-cell work that remains: sweep recompute, the reverse sweep
# kernel loop, the (position-wise) W_O/MLP region, and the merge-setup bwd.
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# K1 backward: (a) RMSNorm backward (row-local, full width), (b) head-local
# attention-merge backward. Gradients through the running max cancel (scale
# invariance), so m is treated as constant; dm_comm flows only through
# w = exp(m_comm - m) and is closed by the hand-written merge-setup backward.
# ---------------------------------------------------------------------------

@triton.jit
def _k1b_rms_kernel(
    du_ptr, o_ptr, rmsw_ptr, do_ptr, dwgt_ptr,
    eps, c: tl.constexpr, d: tl.constexpr, R: tl.constexpr, DP: tl.constexpr,
):
    """u = o * rs * wgt with rs = 1/sqrt(mean(o^2)+eps).
    do = rs*(du*wgt) - o * rs^3/d * sum(du*wgt*o);  dwgt += sum_rows(du*o*rs)."""
    pid = tl.program_id(0)
    n_tiles = c // R
    b = pid // n_tiles
    rt = pid % n_tiles
    rows = rt * R + tl.arange(0, R)
    col = tl.arange(0, DP)                # DP = next_power_of_2(d)
    cm = col < d
    base = (b * c + rows[:, None]) * d + col[None, :]
    du = tl.load(du_ptr + base, mask=cm[None, :], other=0.0).to(tl.float32)
    o = tl.load(o_ptr + base, mask=cm[None, :], other=0.0)  # 0 = sum-neutral
    wgt = tl.load(rmsw_ptr + col, mask=cm, other=0.0).to(tl.float32)
    ms = tl.sum(o * o, axis=1) / d                # explicit /d, NOT /DP
    rs = 1.0 / tl.sqrt(ms + eps)
    duw = du * wgt[None, :]
    dot = tl.sum(duw * o, axis=1)
    do = rs[:, None] * duw - o * (rs * rs * rs / d * dot)[:, None]
    tl.store(do_ptr + base, do, mask=cm[None, :])
    dw = tl.sum(du * o * rs[:, None], axis=0)            # [d] partial per tile
    tl.atomic_add(dwgt_ptr + col, dw, mask=cm)


@triton.jit
def _k1b_rms_det_kernel(
    du_ptr, o_ptr, rmsw_ptr, do_ptr, dwpart_ptr,
    eps, c: tl.constexpr, d: tl.constexpr, R: tl.constexpr, DP: tl.constexpr,
):
    """DETERMINISTIC twin of _k1b_rms_kernel: identical `do` math (row-local,
    already deterministic), but the RMS-weight partial dw is written to this
    program's OWN slot dwpart[pid, :] (plain store, no cross-program race)
    instead of tl.atomic_add into a shared [d]. The caller reduces the
    [n_programs, d] partial with a fixed-order torch sum. Selected only when
    cfg.deterministic_bwd; the non-det _k1b_rms_kernel stays byte-identical."""
    pid = tl.program_id(0)
    n_tiles = c // R
    b = pid // n_tiles
    rt = pid % n_tiles
    rows = rt * R + tl.arange(0, R)
    col = tl.arange(0, DP)                # DP = next_power_of_2(d)
    cm = col < d
    base = (b * c + rows[:, None]) * d + col[None, :]
    du = tl.load(du_ptr + base, mask=cm[None, :], other=0.0).to(tl.float32)
    o = tl.load(o_ptr + base, mask=cm[None, :], other=0.0)  # 0 = sum-neutral
    wgt = tl.load(rmsw_ptr + col, mask=cm, other=0.0).to(tl.float32)
    ms = tl.sum(o * o, axis=1) / d                # explicit /d, NOT /DP
    rs = 1.0 / tl.sqrt(ms + eps)
    duw = du * wgt[None, :]
    dot = tl.sum(duw * o, axis=1)
    do = rs[:, None] * duw - o * (rs * rs * rs / d * dot)[:, None]
    tl.store(do_ptr + base, do, mask=cm[None, :])
    dw = tl.sum(du * o * rs[:, None], axis=0)            # [d] partial per tile
    tl.store(dwpart_ptr + pid * d + col, dw, mask=cm)    # own slot, no atomic


@triton.jit
def _k1b_attn_kernel(
    do_ptr, o_ptr,                          # [B,c,d] fp32 (do from rms bwd)
    q_ptr, vraw_ptr, kin_ptr, vin_ptr,      # q/vraw LAYER bufs (strided, base pos0); kin/vin contig
    sself_ptr, A_ptr, Z_ptr, M_ptr,         # sself LAYER buf (strided); A/Z/M per-cell
    dq_ptr, dvraw_ptr, dkin_ptr, dvin_ptr,  # dq/dvraw LAYER bufs ACCUM; dkin/dvin [B*c,d] MERGED
    dss_ptr, dA_ptr, dZ_ptr, dM_ptr,        # dss LAYER buf ACCUM; dA/dZ/dM per-cell ACCUM
    scale,
    q_sb, q_sh, ss_sb, ss_sh,               # layer-buffer strides
    h: tl.constexpr, c: tl.constexpr, dh: tl.constexpr, d: tl.constexpr,
    RC: tl.constexpr, HAS_COMM: tl.constexpr, IEEE: tl.constexpr,
):
    """Row-tiled per (b, head, row-tile). All row-indexed outputs are
    program-local; the column-indexed dK_in/dV_in use atomics when the chunk
    is split across row tiles (NT > 1), plain stores otherwise."""
    pid = tl.program_id(0)
    nt = c // RC
    b = pid // (h * nt)
    hh = (pid // nt) % h
    rt = pid % nt
    rows = rt * RC + tl.arange(0, RC)
    cols = tl.arange(0, c)
    dd = tl.arange(0, dh)
    hb = ((b * h + hh) * c)
    lb = b * q_sb + hh * q_sh
    sb = b * ss_sb + hh * ss_sh

    q = tl.load(q_ptr + lb + rows[:, None] * dh + dd[None, :]).to(tl.float32)
    ki = tl.load(kin_ptr + hb * dh + cols[:, None] * dh + dd[None, :]).to(tl.float32)
    vi = tl.load(vin_ptr + hb * dh + cols[:, None] * dh + dd[None, :]).to(tl.float32)
    vr = tl.load(vraw_ptr + lb + rows[:, None] * dh + dd[None, :]).to(tl.float32)
    s_self = tl.load(sself_ptr + sb + rows)
    if IEEE:
        s = tl.dot(q, tl.trans(ki), input_precision="ieee") * scale
    else:
        s = tl.dot(q, tl.trans(ki)) * scale
    s = tl.where(cols[None, :] < rows[:, None], s, float("-inf"))
    m_tail = tl.maximum(tl.max(s, axis=1), s_self)
    if HAS_COMM:
        m_comm = tl.load(M_ptr + hb + rows)
        m = tl.maximum(m_tail, m_comm)
        w = tl.exp(m_comm - m)
        Zc = tl.load(Z_ptr + hb + rows)
        A = tl.load(A_ptr + hb * dh + rows[:, None] * dh + dd[None, :])
    else:
        m = m_tail
        w = tl.zeros((RC,), dtype=tl.float32)
        Zc = tl.zeros((RC,), dtype=tl.float32)
        A = tl.zeros((RC, dh), dtype=tl.float32)
    e = tl.exp(s - m[:, None])
    es = tl.exp(s_self - m)
    Z = Zc * w + tl.sum(e, axis=1) + es

    do = tl.load(do_ptr + (b * c + rows[:, None]) * d
                 + hh * dh + dd[None, :]).to(tl.float32)
    o_h = tl.load(o_ptr + (b * c + rows[:, None]) * d + hh * dh + dd[None, :])
    dN = do / Z[:, None]
    dZ = -tl.sum(o_h * do, axis=1) / Z
    if IEEE:
        deV = tl.dot(dN, tl.trans(vi), input_precision="ieee")
    else:
        deV = tl.dot(dN, tl.trans(vi))
    de = deV + dZ[:, None]
    ds = e * de
    des = tl.sum(vr * dN, axis=1) + dZ
    dss = es * des
    dvr = es[:, None] * dN
    if IEEE:
        dq = tl.dot(ds, ki, input_precision="ieee") * scale
        dki = tl.dot(tl.trans(ds), q, input_precision="ieee") * scale
        dvi = tl.dot(tl.trans(e), dN, input_precision="ieee")
    else:
        dq = tl.dot(ds, ki) * scale
        dki = tl.dot(tl.trans(ds), q) * scale
        dvi = tl.dot(tl.trans(e), dN)
    ob = lb + rows[:, None] * dh + dd[None, :]
    prev = tl.load(dq_ptr + ob)
    tl.store(dq_ptr + ob, prev + dq)
    prev = tl.load(dvraw_ptr + ob)
    tl.store(dvraw_ptr + ob, prev + dvr)
    mb = (b * c + cols[:, None]) * d + hh * dh + dd[None, :]
    if nt == 1:
        tl.store(dkin_ptr + mb, dki)
        tl.store(dvin_ptr + mb, dvi)
    else:
        tl.atomic_add(dkin_ptr + mb, dki)
        tl.atomic_add(dvin_ptr + mb, dvi)
    prev = tl.load(dss_ptr + sb + rows)
    tl.store(dss_ptr + sb + rows, prev + dss)
    if HAS_COMM:
        dA_l = dN * w[:, None]
        dw = tl.sum(A * dN, axis=1) + Zc * dZ
        ab = hb * dh + rows[:, None] * dh + dd[None, :]
        prev = tl.load(dA_ptr + ab)
        tl.store(dA_ptr + ab, prev + dA_l)
        prev = tl.load(dZ_ptr + hb + rows)
        tl.store(dZ_ptr + hb + rows, prev + w * dZ)
        prev = tl.load(dM_ptr + hb + rows)
        tl.store(dM_ptr + hb + rows, prev + w * dw)


@triton.jit
def _k1b_attn_det_kernel(
    do_ptr, o_ptr,
    q_ptr, vraw_ptr, kin_ptr, vin_ptr,
    sself_ptr, A_ptr, Z_ptr, M_ptr,
    dq_ptr, dvraw_ptr, dkin_ptr, dvin_ptr,  # dkin/dvin are [NT, B*c, d]
    dss_ptr, dA_ptr, dZ_ptr, dM_ptr,
    scale,
    q_sb, q_sh, ss_sb, ss_sh,
    ND,                                     # = B*c*d, the row-tile plane stride
    h: tl.constexpr, c: tl.constexpr, dh: tl.constexpr, d: tl.constexpr,
    RC: tl.constexpr, HAS_COMM: tl.constexpr, IEEE: tl.constexpr,
):
    """DETERMINISTIC twin of _k1b_attn_kernel for the split-chunk case (NT>1).
    Every row-indexed output (dq/dvraw/dss/dA/dZ/dM) is already written by a
    single owning program (partitioned by the query-row tile) and so stays a
    plain load+add. ONLY dK_in/dV_in are column-indexed (each row tile writes
    ALL c key columns), which is where the non-det kernel races via atomics —
    here each row tile rt writes its contribution to its OWN plane
    dkin[rt, :, :] (plain store); the caller sums over rt in fixed order."""
    pid = tl.program_id(0)
    nt = c // RC
    b = pid // (h * nt)
    hh = (pid // nt) % h
    rt = pid % nt
    rows = rt * RC + tl.arange(0, RC)
    cols = tl.arange(0, c)
    dd = tl.arange(0, dh)
    hb = ((b * h + hh) * c)
    lb = b * q_sb + hh * q_sh
    sb = b * ss_sb + hh * ss_sh

    q = tl.load(q_ptr + lb + rows[:, None] * dh + dd[None, :]).to(tl.float32)
    ki = tl.load(kin_ptr + hb * dh + cols[:, None] * dh + dd[None, :]).to(tl.float32)
    vi = tl.load(vin_ptr + hb * dh + cols[:, None] * dh + dd[None, :]).to(tl.float32)
    vr = tl.load(vraw_ptr + lb + rows[:, None] * dh + dd[None, :]).to(tl.float32)
    s_self = tl.load(sself_ptr + sb + rows)
    if IEEE:
        s = tl.dot(q, tl.trans(ki), input_precision="ieee") * scale
    else:
        s = tl.dot(q, tl.trans(ki)) * scale
    s = tl.where(cols[None, :] < rows[:, None], s, float("-inf"))
    m_tail = tl.maximum(tl.max(s, axis=1), s_self)
    if HAS_COMM:
        m_comm = tl.load(M_ptr + hb + rows)
        m = tl.maximum(m_tail, m_comm)
        w = tl.exp(m_comm - m)
        Zc = tl.load(Z_ptr + hb + rows)
        A = tl.load(A_ptr + hb * dh + rows[:, None] * dh + dd[None, :])
    else:
        m = m_tail
        w = tl.zeros((RC,), dtype=tl.float32)
        Zc = tl.zeros((RC,), dtype=tl.float32)
        A = tl.zeros((RC, dh), dtype=tl.float32)
    e = tl.exp(s - m[:, None])
    es = tl.exp(s_self - m)
    Z = Zc * w + tl.sum(e, axis=1) + es

    do = tl.load(do_ptr + (b * c + rows[:, None]) * d
                 + hh * dh + dd[None, :]).to(tl.float32)
    o_h = tl.load(o_ptr + (b * c + rows[:, None]) * d + hh * dh + dd[None, :])
    dN = do / Z[:, None]
    dZ = -tl.sum(o_h * do, axis=1) / Z
    if IEEE:
        deV = tl.dot(dN, tl.trans(vi), input_precision="ieee")
    else:
        deV = tl.dot(dN, tl.trans(vi))
    de = deV + dZ[:, None]
    ds = e * de
    des = tl.sum(vr * dN, axis=1) + dZ
    dss = es * des
    dvr = es[:, None] * dN
    if IEEE:
        dq = tl.dot(ds, ki, input_precision="ieee") * scale
        dki = tl.dot(tl.trans(ds), q, input_precision="ieee") * scale
        dvi = tl.dot(tl.trans(e), dN, input_precision="ieee")
    else:
        dq = tl.dot(ds, ki) * scale
        dki = tl.dot(tl.trans(ds), q) * scale
        dvi = tl.dot(tl.trans(e), dN)
    ob = lb + rows[:, None] * dh + dd[None, :]
    prev = tl.load(dq_ptr + ob)
    tl.store(dq_ptr + ob, prev + dq)
    prev = tl.load(dvraw_ptr + ob)
    tl.store(dvraw_ptr + ob, prev + dvr)
    mb = rt * ND + (b * c + cols[:, None]) * d + hh * dh + dd[None, :]
    tl.store(dkin_ptr + mb, dki)          # own row-tile plane, no atomic
    tl.store(dvin_ptr + mb, dvi)
    prev = tl.load(dss_ptr + sb + rows)
    tl.store(dss_ptr + sb + rows, prev + dss)
    if HAS_COMM:
        dA_l = dN * w[:, None]
        dw = tl.sum(A * dN, axis=1) + Zc * dZ
        ab = hb * dh + rows[:, None] * dh + dd[None, :]
        prev = tl.load(dA_ptr + ab)
        tl.store(dA_ptr + ab, prev + dA_l)
        prev = tl.load(dZ_ptr + hb + rows)
        tl.store(dZ_ptr + hb + rows, prev + w * dZ)
        prev = tl.load(dM_ptr + hb + rows)
        tl.store(dM_ptr + hb + rows, prev + w * dw)


@triton.jit
def _refine2_bwd_kernel(
    dkout_ptr, dvout_ptr,                   # incoming grads, merged [N,d]
    k_ptr, v_ptr, ctxg_ptr, vc_ptr, glogx_ptr, cos_ptr, sin_ptr,
    dk_ptr, dv_ptr, dctxg_ptr, dvc_ptr, dglogx_ptr,  # dk/dv/dglogx: LAYER bufs ACCUM; dctxg/dvc per-sweep
    kv_sb, gx_sb,                           # batch strides (k/v/dk/dv, glogx/dglogx)
    gate_pin, tau,
    d: tl.constexpr, dh: tl.constexpr, c: tl.constexpr,
    N, HALF: tl.constexpr, CTXG_BF16: tl.constexpr, SD: tl.constexpr,
    VSEP: tl.constexpr, NPIN: tl.constexpr, HALFP: tl.constexpr,
    CLAMP: tl.constexpr = False,
):
    n = tl.program_id(0)
    p = n % c
    b_idx = n // c
    krow = b_idx * kv_sb + p * d
    gxrow = b_idx * gx_sb + p * 2 * d
    lane = tl.arange(0, HALFP)              # HALFP = next_power_of_2(HALF)
    msk = lane < HALF                       # elementwise: other=0.0 safe;
    hh = lane // (dh // 2)                  # accum loads masked too (their
    off = lane % (dh // 2)                  # stores are masked, value unused)
    a_dim = hh * dh + off
    b_dim = a_dim + dh // 2
    in_base = n * d

    cos = tl.load(cos_ptr + p * (dh // 2) + off, mask=msk, other=0.0)
    sin = tl.load(sin_ptr + p * (dh // 2) + off, mask=msk, other=0.0)
    dK1 = tl.load(dkout_ptr + in_base + a_dim, mask=msk, other=0.0).to(tl.float32)
    dK2 = tl.load(dkout_ptr + in_base + b_dim, mask=msk, other=0.0).to(tl.float32)
    da = dK1 * cos + dK2 * sin
    db = -dK1 * sin + dK2 * cos

    k_a = tl.load(k_ptr + krow + a_dim, mask=msk, other=0.0).to(tl.float32)
    k_b = tl.load(k_ptr + krow + b_dim, mask=msk, other=0.0).to(tl.float32)
    kc_a = tl.load(ctxg_ptr + n * SD + a_dim, mask=msk, other=0.0).to(tl.float32)
    kc_b = tl.load(ctxg_ptr + n * SD + b_dim, mask=msk, other=0.0).to(tl.float32)
    gl_a = tl.load(ctxg_ptr + n * SD + 2 * d + a_dim, mask=msk,
                   other=0.0).to(tl.float32) \
        + tl.load(glogx_ptr + gxrow + a_dim, mask=msk, other=0.0).to(tl.float32)
    gl_b = tl.load(ctxg_ptr + n * SD + 2 * d + b_dim, mask=msk,
                   other=0.0).to(tl.float32) \
        + tl.load(glogx_ptr + gxrow + b_dim, mask=msk, other=0.0).to(tl.float32)
    g_a = tl.sigmoid(gl_a)
    g_b = tl.sigmoid(gl_b)
    dgl_a = da * (k_a - kc_a) * g_a * (1.0 - g_a)
    dgl_b = db * (k_b - kc_b) * g_b * (1.0 - g_b)
    prev = tl.load(dk_ptr + krow + a_dim, mask=msk, other=0.0)
    tl.store(dk_ptr + krow + a_dim, prev + da * g_a, mask=msk)
    prev = tl.load(dk_ptr + krow + b_dim, mask=msk, other=0.0)
    tl.store(dk_ptr + krow + b_dim, prev + db * g_b, mask=msk)
    if CTXG_BF16:
        tl.store(dctxg_ptr + n * SD + a_dim, (da * (1.0 - g_a)).to(tl.bfloat16),
                 mask=msk)
        tl.store(dctxg_ptr + n * SD + b_dim, (db * (1.0 - g_b)).to(tl.bfloat16),
                 mask=msk)
        tl.store(dctxg_ptr + n * SD + 2 * d + a_dim, dgl_a.to(tl.bfloat16),
                 mask=msk)
        tl.store(dctxg_ptr + n * SD + 2 * d + b_dim, dgl_b.to(tl.bfloat16),
                 mask=msk)
    else:
        tl.store(dctxg_ptr + n * SD + a_dim, da * (1.0 - g_a), mask=msk)
        tl.store(dctxg_ptr + n * SD + b_dim, db * (1.0 - g_b), mask=msk)
        tl.store(dctxg_ptr + n * SD + 2 * d + a_dim, dgl_a, mask=msk)
        tl.store(dctxg_ptr + n * SD + 2 * d + b_dim, dgl_b, mask=msk)
    prev = tl.load(dglogx_ptr + gxrow + a_dim, mask=msk, other=0.0)
    tl.store(dglogx_ptr + gxrow + a_dim, prev + dgl_a, mask=msk)
    prev = tl.load(dglogx_ptr + gxrow + b_dim, mask=msk, other=0.0)
    tl.store(dglogx_ptr + gxrow + b_dim, prev + dgl_b, mask=msk)

    dV1 = tl.load(dvout_ptr + in_base + a_dim, mask=msk, other=0.0).to(tl.float32)
    dV2 = tl.load(dvout_ptr + in_base + b_dim, mask=msk, other=0.0).to(tl.float32)
    v_a = tl.load(v_ptr + krow + a_dim, mask=msk, other=0.0).to(tl.float32)
    v_b = tl.load(v_ptr + krow + b_dim, mask=msk, other=0.0).to(tl.float32)
    if VSEP:
        vc_a = tl.load(vc_ptr + n * d + a_dim, mask=msk, other=0.0).to(tl.float32)
        vc_b = tl.load(vc_ptr + n * d + b_dim, mask=msk, other=0.0).to(tl.float32)
    else:
        vc_a = tl.load(ctxg_ptr + n * SD + d + a_dim, mask=msk,
                       other=0.0).to(tl.float32)
        vc_b = tl.load(ctxg_ptr + n * SD + d + b_dim, mask=msk,
                       other=0.0).to(tl.float32)
    gu_a = tl.load(ctxg_ptr + n * SD + 3 * d + a_dim, mask=msk,
                   other=0.0).to(tl.float32)
    gu_b = tl.load(ctxg_ptr + n * SD + 3 * d + b_dim, mask=msk,
                   other=0.0).to(tl.float32)
    if NPIN > 0:
        gu_a = tl.where(a_dim < NPIN, gate_pin, gu_a)
        gu_b = tl.where(b_dim < NPIN, gate_pin, gu_b)
    gvl_a = gu_a + tl.load(glogx_ptr + gxrow + d + a_dim, mask=msk,
                           other=0.0).to(tl.float32)
    gvl_b = gu_b + tl.load(glogx_ptr + gxrow + d + b_dim, mask=msk,
                           other=0.0).to(tl.float32)
    gv_a = tl.sigmoid(gvl_a)
    gv_b = tl.sigmoid(gvl_b)
    if CLAMP:
        # backward through v'' = u * min(1, tau/||u||), u the pre-clamp blend.
        # Projection Jacobian (symmetric): clamped head =>
        #   dV <- (tau/||u||)(dV - u (u.dV)/||u||^2); in-range head => identity.
        HD2: tl.constexpr = dh // 2
        NHS: tl.constexpr = HALFP // HD2
        u_a = gv_a * v_a + (1.0 - gv_a) * vc_a
        u_b = gv_b * v_b + (1.0 - gv_b) * vc_b
        hn2 = tl.sum(tl.reshape(u_a * u_a + u_b * u_b, (NHS, HD2)), axis=1)
        udv = tl.sum(tl.reshape(u_a * dV1 + u_b * dV2, (NHS, HD2)), axis=1)
        hn2 = tl.reshape(tl.broadcast_to(hn2[:, None], (NHS, HD2)), (HALFP,))
        udv = tl.reshape(tl.broadcast_to(udv[:, None], (NHS, HD2)), (HALFP,))
        over = hn2 > tau * tau
        s = tau * tl.rsqrt(hn2 + 1e-20)
        coef = udv / (hn2 + 1e-20)
        dV1 = tl.where(over, s * (dV1 - u_a * coef), dV1)
        dV2 = tl.where(over, s * (dV2 - u_b * coef), dV2)
    dgvl_a = dV1 * (v_a - vc_a) * gv_a * (1.0 - gv_a)
    dgvl_b = dV2 * (v_b - vc_b) * gv_b * (1.0 - gv_b)
    prev = tl.load(dv_ptr + krow + a_dim, mask=msk, other=0.0)
    tl.store(dv_ptr + krow + a_dim, prev + dV1 * gv_a, mask=msk)
    prev = tl.load(dv_ptr + krow + b_dim, mask=msk, other=0.0)
    tl.store(dv_ptr + krow + b_dim, prev + dV2 * gv_b, mask=msk)
    if CTXG_BF16:
        if VSEP:
            # v_ctx grad exits via dvc (variant chain); the packed v_ctx
            # block is dead for variants — zero it so du/dW_pack[d:2d]
            # see no contribution
            tl.store(dvc_ptr + n * d + a_dim, (dV1 * (1.0 - gv_a)).to(tl.bfloat16),
                     mask=msk)
            tl.store(dvc_ptr + n * d + b_dim, (dV2 * (1.0 - gv_b)).to(tl.bfloat16),
                     mask=msk)
            z = tl.zeros((HALFP,), dtype=tl.bfloat16)
            tl.store(dctxg_ptr + n * SD + d + a_dim, z, mask=msk)
            tl.store(dctxg_ptr + n * SD + d + b_dim, z, mask=msk)
        else:
            tl.store(dctxg_ptr + n * SD + d + a_dim,
                     (dV1 * (1.0 - gv_a)).to(tl.bfloat16), mask=msk)
            tl.store(dctxg_ptr + n * SD + d + b_dim,
                     (dV2 * (1.0 - gv_b)).to(tl.bfloat16), mask=msk)
        tl.store(dctxg_ptr + n * SD + 3 * d + a_dim, dgvl_a.to(tl.bfloat16),
                 mask=msk)
        tl.store(dctxg_ptr + n * SD + 3 * d + b_dim, dgvl_b.to(tl.bfloat16),
                 mask=msk)
    else:
        if VSEP:
            tl.store(dvc_ptr + n * d + a_dim, dV1 * (1.0 - gv_a), mask=msk)
            tl.store(dvc_ptr + n * d + b_dim, dV2 * (1.0 - gv_b), mask=msk)
            z = tl.zeros((HALFP,), dtype=tl.float32)
            tl.store(dctxg_ptr + n * SD + d + a_dim, z, mask=msk)
            tl.store(dctxg_ptr + n * SD + d + b_dim, z, mask=msk)
        else:
            tl.store(dctxg_ptr + n * SD + d + a_dim, dV1 * (1.0 - gv_a),
                     mask=msk)
            tl.store(dctxg_ptr + n * SD + d + b_dim, dV2 * (1.0 - gv_b),
                     mask=msk)
        tl.store(dctxg_ptr + n * SD + 3 * d + a_dim, dgvl_a, mask=msk)
        tl.store(dctxg_ptr + n * SD + 3 * d + b_dim, dgvl_b, mask=msk)
    prev = tl.load(dglogx_ptr + gxrow + d + a_dim, mask=msk, other=0.0)
    tl.store(dglogx_ptr + gxrow + d + a_dim, prev + dgvl_a, mask=msk)
    prev = tl.load(dglogx_ptr + gxrow + d + b_dim, mask=msk, other=0.0)
    tl.store(dglogx_ptr + gxrow + d + b_dim, prev + dgvl_b, mask=msk)


# ---------------------------------------------------------------------------
# ctx_mod variant v_in, fused (fwd + bwd chain). MODE: 0=mult, 1=mult_res,
# 2=fusion. hx = the sweep-invariant hoisted term (sigmoid(v_gate(xhat)) for
# the mult family, the v_pre xhat-half pre-activation for fusion), a layer
# buf [B,T,d] read strided at pos0. The u-driven pre-GEMM comes packed in
# ctxg[:, 4d:5d]. fp32 compute, storage dtype of the target buffers — the
# same contract as the rest of the file.
# ---------------------------------------------------------------------------

@triton.jit
def _vin_fwd_kernel(
    u_ptr, hx_ptr, ctxg_ptr, vin_ptr,
    hx_sb,                                  # batch stride of hx (base at pos0)
    lam,                                    # mult_res-lambda carry leak
    d: tl.constexpr, c: tl.constexpr, SD: tl.constexpr, MODE: tl.constexpr,
    DP: tl.constexpr,
):
    n = tl.program_id(0)
    p = n % c
    b_idx = n // c
    col = tl.arange(0, DP)                  # DP = next_power_of_2(d)
    cm = col < d                            # elementwise: other=0.0 safe
    hx = tl.load(hx_ptr + b_idx * hx_sb + p * d + col, mask=cm,
                 other=0.0).to(tl.float32)
    pre_u = tl.load(ctxg_ptr + n * SD + 4 * d + col, mask=cm,
                    other=0.0).to(tl.float32)
    if MODE == 2:
        pre = hx + pre_u
        vin = pre * tl.sigmoid(pre)         # silu
    else:
        vin = hx * libdevice.tanh(pre_u)
    if MODE != 0:                           # the +lam*u shortcut (res mods)
        vin += lam * tl.load(u_ptr + n * d + col, mask=cm,
                             other=0.0).to(tl.float32)
    tl.store(vin_ptr + n * d + col, vin.to(vin_ptr.dtype.element_ty), mask=cm)


@triton.jit
def _vin_bwd_kernel(
    dvin_ptr,                               # [N,d] = d_vctx @ w_vctx
    u_ptr, hx_ptr, ctxg_ptr,
    dctxg_ptr, vin_ptr, dhx_ptr,            # dctxg[4d:5d] store; v_in recompute; d_hoist ACCUM [N,d] fp32
    hx_sb, lam,
    d: tl.constexpr, c: tl.constexpr, SD: tl.constexpr, MODE: tl.constexpr,
    DP: tl.constexpr,
):
    n = tl.program_id(0)
    p = n % c
    b_idx = n // c
    col = tl.arange(0, DP)                  # DP = next_power_of_2(d)
    cm = col < d                            # elementwise: other=0.0 safe
    hx = tl.load(hx_ptr + b_idx * hx_sb + p * d + col, mask=cm,
                 other=0.0).to(tl.float32)
    pre_u = tl.load(ctxg_ptr + n * SD + 4 * d + col, mask=cm,
                    other=0.0).to(tl.float32)
    dvin = tl.load(dvin_ptr + n * d + col, mask=cm, other=0.0).to(tl.float32)
    if MODE == 2:
        pre = hx + pre_u
        sg = tl.sigmoid(pre)
        vin = pre * sg
        # silu'(pre) = sg*(1 + pre*(1-sg)); d_pre feeds BOTH the packed
        # u-half (dctxg) and the deferred xhat-half (dhx)
        d_pre = dvin * (sg * (1.0 + pre * (1.0 - sg)))
        dhc = d_pre
    else:
        t = libdevice.tanh(pre_u)
        vin = hx * t
        dhc = dvin * t                      # d_sig contribution
        d_pre = (dvin * hx) * (1.0 - t * t)
    if MODE != 0:
        vin += lam * tl.load(u_ptr + n * d + col, mask=cm,
                             other=0.0).to(tl.float32)
    tl.store(dctxg_ptr + n * SD + 4 * d + col,
             d_pre.to(dctxg_ptr.dtype.element_ty), mask=cm)
    tl.store(vin_ptr + n * d + col, vin.to(vin_ptr.dtype.element_ty), mask=cm)
    prev = tl.load(dhx_ptr + n * d + col, mask=cm, other=0.0)
    tl.store(dhx_ptr + n * d + col, prev + dhc, mask=cm)


_MODE_ID = {"mult": 0, "mult_res": 1, "fusion": 2}


def _det_scatter_add_(dst, dim, index, src):
    """scatter_add_ with a run-to-run bit-identical (deterministic) reduction.
    CUDA scatter_add_ is atomic (non-deterministic) by default; scoping
    torch.use_deterministic_algorithms selects the deterministic kernel for
    just this call and restores the previous global setting."""
    prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        dst.scatter_add_(dim, index, src)
    finally:
        torch.use_deterministic_algorithms(prev)


def _merged(x_bhcd):
    """[B,h,c,dh] -> [B*c, h*dh] (merged-head row layout)."""
    B, h, c, dh = x_bhcd.shape
    return x_bhcd.transpose(1, 2).reshape(B * c, h * dh).contiguous()


class _CellBwdScratch:
    """Per-cell fp32 scratch (committed-accum grads, chain ping-pong, do)."""

    def __init__(self, B, h, c_max, dh, d, device):
        f32 = dict(device=device, dtype=torch.float32)
        self.dA = torch.zeros(B, h, c_max, dh, **f32)
        self.dZ = torch.zeros(B, h, c_max, **f32)
        self.dM = torch.zeros(B, h, c_max, **f32)
        self.dkin = [torch.zeros(B * c_max, d, **f32) for _ in range(2)]
        self.dvin = [torch.zeros(B * c_max, d, **f32) for _ in range(2)]
        self.do_buf = torch.zeros(B, c_max, d, **f32)
        self.c_max = c_max

    def views(self, B, h, c, dh, d):
        N = B * c
        dkin = [t.reshape(-1, self.c_max, d)[:, :c].reshape(N, d)
                for t in self.dkin]
        dvin = [t.reshape(-1, self.c_max, d)[:, :c].reshape(N, d)
                for t in self.dvin]
        return (self.dA[:, :, :c], self.dZ[:, :, :c], self.dM[:, :, :c],
                dkin, dvin, self.do_buf[:, :c])


class LayerBufs:
    """Engine-owned per-layer projection buffers + backward accumulators."""

    def __init__(self, B, h, T, dh, d, act, device, ctx_mod="linear"):
        f32 = dict(device=device, dtype=torch.float32)
        a = dict(device=device, dtype=act)
        self.qr = torch.zeros(B, h, T, dh, **a)
        self.kself = torch.zeros(B, h, T, dh, **a)
        self.vraw = torch.zeros(B, h, T, dh, **a)
        self.kflat = torch.zeros(B, T, d, **a)
        self.vflat = torch.zeros(B, T, d, **a)
        self.sself = torch.zeros(B, h, T, **f32)
        self.glogx = torch.zeros(B, T, 2 * d, **a)
        # backward accumulators (zeroed per layer in the proj pass)
        self.dq = torch.zeros(B, h, T, dh, **f32)
        self.dvraw = torch.zeros(B, h, T, dh, **f32)
        self.dkself = torch.zeros(B, h, T, dh, **f32)
        self.dss = torch.zeros(B, h, T, **f32)
        self.dkflat = torch.zeros(B, T, d, **f32)
        self.dvflat = torch.zeros(B, T, d, **f32)
        self.dglogx = torch.zeros(B, T, 2 * d, **f32)
        self.do_T = torch.zeros(B, T, d, **f32)
        # ctx_mod variants (mult/mult_res/fusion): grad wrt xhat that flows
        # ONLY through the v-branch (v_gate(xhat) / v_pre([xhat;u])) — folded
        # into the layer_close autograd region (which owns attn_norm) as an
        # extra grad_output on xhat. Zero for linear (byte-identical path).
        self.d_xhat = torch.zeros(B, T, d, **f32)
        # sweep-invariant hoists (function of xhat only, computed once per
        # layer in layer_proj; act dtype = the dtype refine_kv's .to(u.dtype)
        # produces) + per-layer accumulators for their deferred backward
        # chain, applied ONCE per layer in layer_close.
        if ctx_mod in ("mult", "mult_res"):
            self.sig = torch.zeros(B, T, d, **a)      # sigmoid(v_gate(xhat))
            self.d_sig = torch.zeros(B, T, d, **f32)
        elif ctx_mod == "fusion":
            self.prex = torch.zeros(B, T, d, **a)     # v_pre xhat-half (no bias)
            self.d_prex = torch.zeros(B, T, d, **f32)


class FusedSweepCell:
    def __init__(self, model):
        cfg = model.cfg
        assert cfg.gate_type == "convex_sigmoid" and cfg.gate_input == "xu" \
            and cfg.enable_key_mod and cfg.enable_value_mod and cfg.rmsnorm_on_o \
            and getattr(cfg, "gate_type_v", "convex") == "convex", \
            "FusedSweepCell bakes the convex value blend; independent-v runs " \
            "on the verbatim/compiled cell path"
        self.model = model
        self.cfg = cfg
        d = cfg.d_model
        dev = next(model.parameters()).device
        dt = model.blocks[0].attn.w_kctx.weight.dtype
        L = cfg.n_layers
        self.mod = getattr(cfg, "ctx_mod", "linear")
        # v_ctx_norm: the norm is applied torch-side at the v_ctx formation
        # site (kernels untouched). Linear mode moves its v-block onto the
        # VSEP path (separate vc tensor) so the packed ctxg stays PRE-norm —
        # the backward dW_pack/du GEMMs need the pre-norm chain, and the
        # norm backward is closed per sweep via a tiny autograd region over
        # attn.norm_vctx (ctx_gain joins the param grads like v_gain did).
        self.vnorm_ctx = getattr(cfg, "v_ctx_norm", "none") == "rms"
        self.vsep = self.mod != "linear" or self.vnorm_ctx
        # K-norm: per-head RMS+gain on the ctx KEY block ctxg[:, :d] before
        # the blend. Applied in-place torch-side after the packed GEMM (the
        # k-block is only read by _refine2); backward corrects dctxg[:, :d]
        # through the norm before dW_pack/du. Identity at g=1.
        self.knorm_ctx = getattr(cfg, "k_ctx_norm", "none") == "rms"
        # v'' entry-norm clamp (storm governor): per-(position,head) cap at
        # radius tau, applied in the _refine2 store (every sweep -> matches
        # eager clamp_vpp; at K=c the Jacobi sweep is exact so parity holds).
        self.v_clamp_tau = float(getattr(cfg, "v_clamp_tau", 0.0) or 0.0)
        # mult_res-lambda: leak on the identity carry (v_in = lam*u + branch);
        # 1.0 = byte-identical current mult_res.
        self.mult_res_lambda = float(getattr(cfg, "mult_res_lambda", 1.0))
        # deterministic backward: route the float atomic_add accumulators
        # (dnormw always; dK/dV cache-grad + dK_in/dV_in when a chunk splits
        # across row tiles) through per-program partials + fixed-order torch
        # sums. False = byte-identical to the historical non-det path.
        self.deterministic_bwd = bool(getattr(cfg, "deterministic_bwd", False))
        # variants: rows 4d:5d = the u-driven half of the variant pre-GEMM
        # (v_map for mult family, v_pre u-columns for fusion), folded into the
        # per-sweep packed GEMM. Linear stays [4d, d] (byte-identical path).
        pk = 5 * d if self.mod != "linear" else 4 * d
        self.W_pack = [torch.zeros(pk, d, device=dev, dtype=dt) for _ in range(L)]
        self.W_gx = [torch.zeros(2 * d, d, device=dev, dtype=dt) for _ in range(L)]
        self.b_g = [torch.zeros(2 * d, device=dev, dtype=dt) for _ in range(L)]

    @torch.no_grad()
    def pack(self):
        d = self.cfg.d_model

        def gate_dense(g):
            # LoRAGate (gate_rank>0): materialize the effective dense [d, 2d]
            # weight b@a for the packed GEMMs; grads chain back to the
            # factors in _bwd_epilogue_body / layer_close.
            w = g.weight
            return w if w.shape[1] == 2 * d else g.b.weight @ g.a.weight

        for li, blk in enumerate(self.model.blocks):
            a = blk.attn
            self.W_pack[li][:d].copy_(a.w_kctx.weight)
            self.W_pack[li][d:2 * d].copy_(a.w_vctx.weight)
            wgk, wgv = gate_dense(a.w_gk), gate_dense(a.w_gv)
            self.W_pack[li][2 * d:3 * d].copy_(wgk[:, d:])
            self.W_pack[li][3 * d:4 * d].copy_(wgv[:, d:])
            if self.mod in ("mult", "mult_res"):
                self.W_pack[li][4 * d:].copy_(a.v_map.weight)
            elif self.mod == "fusion":
                self.W_pack[li][4 * d:].copy_(a.v_pre.weight[:, d:])
            self.W_gx[li][:d].copy_(wgk[:, :d])
            self.W_gx[li][d:].copy_(wgv[:, :d])
            self.b_g[li][:d].copy_(a.w_gk.bias)
            self.b_g[li][d:].copy_(a.w_gv.bias)

    # ---- ctx_mod variant helpers -------------------------------------------

    def _variant_vctx(self, attn, P, pos0, ctxg, u_bcd, vin_buf):
        """v_ctx = w_vctx(v_in) for the supported non-linear ctx_mod: one
        fused kernel builds v_in from the hoisted xhat term (P.sig / P.prex,
        computed once per layer in layer_proj) and the packed u-driven
        pre-GEMM (ctxg[:, 4d:5d]), then the one unavoidable per-sweep GEMM
        (the nonlinearity sits between u and w_vctx). Mirrors
        DualModAttention.refine_kv; applied in BOTH the forward cell and the
        backward recompute so fwd/bwd see the same function. The stale
        w_vctx@u block ctxg[:, d:2d] is computed-and-ignored — slicing it out
        would fragment the packed GEMM; its dctxg block is zeroed in the
        backward kernel so no grad leaks through it."""
        B, c, d = u_bcd.shape
        hx = P.sig if self.mod != "fusion" else P.prex
        _vin_fwd_kernel[(B * c,)](
            u_bcd, hx[:, pos0:], ctxg, vin_buf, hx_sb=hx.shape[1] * d,
            lam=self.mult_res_lambda,
            d=d, c=c, SD=ctxg.shape[1], MODE=_MODE_ID[self.mod],
            DP=triton.next_power_of_2(d), num_warps=4)
        return attn.w_vctx(vin_buf)          # PRE-norm candidate (v_ctx_norm
        #                                      applied at the call sites)

    @staticmethod
    def _rmsnorm_ctx_vjp(y, dout, gain, n_heads, head_dim, eps=1e-6):
        """Closed-form VJP of the per-head RMS-norm + per-head gain shared by
        norm_kctx / norm_vctx — replaces the per-sweep-per-layer
        torch.autograd.grad (the K/V-norm 2x fused-engine tax) with the analytic
        gradient so the norm rides the fused path. fp32 throughout, mirroring the
        forward's fp32 definition (parity: tests/test_ctxnorm_vjp.py).

          fwd (per head, x = pre-norm):  y = gain * x * r,   r = rsqrt(mean(x^2)+eps)
          dx    = gain * ( r*dout - r^3 * x * <dout,x>/D )
          dgain = sum_pos  r * <dout,x>                         (per head)
        """
        shp = y.shape
        x  = y.reshape(*shp[:-1], n_heads, head_dim).float()            # [.., H, D]
        do = dout.to(torch.float32).reshape(*shp[:-1], n_heads, head_dim)
        r  = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)         # [.., H, 1]
        S  = (do * x).sum(-1, keepdim=True)                             # [.., H, 1] = <dout,x>
        g  = gain.float().unsqueeze(-1)                                 # [H, 1]
        dx = g * (r * do - (r * r * r) * x * S / head_dim)             # [.., H, D]
        dgain = (r * S).reshape(-1, n_heads).sum(0)                     # [H]
        return dx.reshape(shp).to(y.dtype), dgain.to(gain.dtype)

    def _vctx_norm_bwd(self, attn, y, dvc, grads):
        """Backward through the v_ctx norm for one sweep: y is the recomputed
        PRE-norm candidate, dvc the grad wrt the normed v_ctx. Closed-form VJP
        (was a per-sweep autograd region); ctx_gain joins the param grads here."""
        dy, dgain = self._rmsnorm_ctx_vjp(
            y, dvc, attn.ctx_gain, attn.n_heads, attn.head_dim)
        grads.add(attn.ctx_gain, dgain)
        return dy

    def _ctxnorm_fwd_launch(self, gain, buf, n_heads, head_dim):
        """In-place fused per-head RMS+gain on the [:, :d] k/v block of `buf`
        ([N, SD], d = n_heads*head_dim). Replaces the eager norm_kctx/norm_vctx
        (kills the reshape/.float/.to/copy_ flood — analysis/knorm_prof)."""
        N, SD = buf.shape
        _ctxnorm_fwd_kernel[(N * n_heads,)](
            buf, gain, 1e-6, SD=SD, H=n_heads, Dh=head_dim,
            OUT_BF16=buf.dtype == torch.bfloat16,
            DP=triton.next_power_of_2(head_dim), num_warps=1)
        return buf

    def _kctx_norm_bwd(self, attn, y, dkc, grads):
        """Backward through the K-norm for one sweep: y is the recomputed
        PRE-norm k_ctx (u @ W_pack[:d]^T), dkc the grad wrt the NORMED k_ctx
        (dctxg[:, :d], a strided view). Fused closed-form VJP kernel: writes dx
        IN PLACE into dkc + accumulates the k_gain grad. Eager reference stays
        in _rmsnorm_ctx_vjp (parity: analysis/ctxnorm_kernel_test.py)."""
        N, d = y.shape
        H, Dh = attn.n_heads, attn.head_dim
        dgp = torch.empty(N * H, device=y.device, dtype=torch.float32)
        _ctxnorm_bwd_kernel[(N * H,)](
            y, dkc, attn.k_gain, dkc, dgp, 1e-6,
            XSD=y.stride(0), DSD=dkc.stride(0), H=H, Dh=Dh,
            OUT_BF16=dkc.dtype == torch.bfloat16,
            DP=triton.next_power_of_2(Dh), num_warps=1)
        grads.add(attn.k_gain, dgp.reshape(N, H).sum(0).to(attn.k_gain.dtype))
        return dkc

    def _variant_bwd_sweep(self, attn, P, pos0, ctxg_s, dctxg, dvc, u_bcd,
                           vin_buf, d_hoist, dW_vctx_buf):
        """Per-sweep backprop of the ctx_mod v-branch. dvc = grad wrt
        v_ctx = w_vctx(v_in) (written by _refine2_bwd_kernel, which also
        zeroes the dead dctxg v_ctx block). One GEMM back through w_vctx,
        one fused kernel (recompute v_in, fill dctxg[:, 4d:5d] so du and
        dW_pack rows 4d:5d flow through the existing packed machinery,
        accumulate the hoisted-term grad into d_hoist — d_sig for the mult
        family, d_pre for fusion, applied once per layer in layer_close),
        one GEMM for dW_vctx. Returns d_v_in (the +u shortcut) for the res
        mods, None for mult."""
        B, c, d = u_bcd.shape
        N = B * c
        hx = P.sig if self.mod != "fusion" else P.prex
        d_v_in = dvc @ attn.w_vctx.weight
        _vin_bwd_kernel[(N,)](
            d_v_in, u_bcd, hx[:, pos0:], ctxg_s, dctxg, vin_buf, d_hoist,
            hx_sb=hx.shape[1] * d, lam=self.mult_res_lambda,
            d=d, c=c, SD=ctxg_s.shape[1],
            MODE=_MODE_ID[self.mod], DP=triton.next_power_of_2(d),
            num_warps=4)
        dW_vctx_buf += dvc.transpose(0, 1) @ vin_buf
        # the identity-carry grad (d_v_in) is scaled by lambda at the du addmm
        # (beta=lam) in the caller; return unscaled here.
        return d_v_in if self.mod != "mult" else None

    # ---- per-layer projection pass -----------------------------------------

    @torch.no_grad()
    def layer_proj(self, li, x_l, P, zero_grads=False):
        blk = self.model.blocks[li]
        attn = blk.attn
        B, T, d = x_l.shape
        cos = self.model.rope_cos[:T]
        sin = self.model.rope_sin[:T]
        xhat = blk.attn_norm(x_l)
        P.qr.copy_(apply_rope(attn._split(attn.wq(xhat)), cos, sin))
        kf = attn.wk(xhat)
        # v_norm applied at source: P.vflat/P.vraw are the ONLY v the kernels
        # ever see (fwd, bwd recompute, cache rows), so norming here keeps
        # kernels untouched; the norm's backward runs in layer_close, whose
        # recompute must call the SAME function (project_v)
        vf = attn.project_v(xhat)
        P.kflat.copy_(kf)
        P.vflat.copy_(vf)
        P.kself.copy_(apply_rope(attn._split(kf), cos, sin))
        P.vraw.copy_(attn._split(vf))
        P.sself.copy_((P.qr.float() * P.kself.float()).sum(-1) * attn.scale)
        P.glogx.copy_(torch.addmm(self.b_g[li], xhat.reshape(B * T, d),
                                  self.W_gx[li].transpose(0, 1)).reshape(B, T, 2 * d))
        if self.mod in ("mult", "mult_res"):
            P.sig.copy_(torch.sigmoid(attn.v_gate(xhat).float()))
        elif self.mod == "fusion":
            P.prex.copy_(F.linear(xhat, attn.v_pre.weight[:, :d]))
        if zero_grads:
            for t in (P.dq, P.dvraw, P.dkself, P.dss, P.dkflat, P.dvflat,
                      P.dglogx, P.d_xhat):
                t.zero_()
            if self.mod in ("mult", "mult_res"):
                P.d_sig.zero_()
            elif self.mod == "fusion":
                P.d_prex.zero_()

    # ---- committed accumulator (per cell) -----------------------------------

    @torch.no_grad()
    def _comm_setup(self, attn, q_r, K_comm, V_comm, want_idx=False):
        s_comm = (q_r @ K_comm.transpose(-1, -2)).float() * attn.scale
        if want_idx:
            m_comm, am = s_comm.max(dim=-1)
        else:
            m_comm, am = s_comm.amax(dim=-1), None
        e_comm = torch.exp(s_comm - m_comm.unsqueeze(-1))
        Z_comm = e_comm.sum(dim=-1)
        A_comm = (e_comm.to(V_comm.dtype) @ V_comm).float().contiguous()
        return A_comm, Z_comm.contiguous(), m_comm.contiguous(), e_comm, am

    # ---- forward -------------------------------------------------------------

    @torch.no_grad()
    def cell_forward(self, li, P, cos_c, sin_c, K_cache, V_cache, pos0,
                     n_sweeps, u_buf, o_buf, ocat_out=None, x_c=None):
        attn = self.model.blocks[li].attn
        cfg = self.cfg
        B = P.qr.shape[0]
        c = cos_c.shape[0]
        h, dh, d = cfg.n_heads, cfg.head_dim, cfg.d_model
        N = B * c
        T = P.qr.shape[2]
        mod = self.mod
        nvr = getattr(cfg, "kmod_vraw_heads", 0)
        # gated_norm kmod: no gate-logit pin (NPIN=0, logits live); the
        # head's v_ctx slice is zeroed instead -> blend == pure write gate
        npin = nvr * dh if getattr(cfg, "kmod_mode", "raw") == "raw" else 0
        q_r = P.qr[:, :, pos0:pos0 + c]
        ieee_ = P.qr.dtype == torch.float32
        if pos0 > 0:
            A_comm = torch.empty(B, h, c, dh, device=q_r.device,
                                 dtype=torch.float32)
            Z_comm = torch.empty(B, h, c, device=q_r.device, dtype=torch.float32)
            m_comm = torch.empty_like(Z_comm)
            am = torch.empty(B, h, c, device=q_r.device, dtype=torch.int32)
            _comm_fwd_kernel[(B * h * (c // min(64, c)),)](
                P.qr[:, :, pos0:], K_cache, V_cache, A_comm, Z_comm, m_comm,
                am, attn.scale, pos0,
                K_cache.stride(0), K_cache.stride(1), K_cache.stride(2),
                q_sb=h * T * dh, q_sh=T * dh,
                h=h, c=c, dh=dh, RC=min(64, c), TA=64, IEEE=ieee_,
                num_warps=4)
        else:
            A_comm = Z_comm = m_comm = P.sself     # dummies (HAS_COMM=False)
        K_cache[:, :, pos0:pos0 + c] = P.kself[:, :, pos0:pos0 + c]
        V_cache[:, :, pos0:pos0 + c] = P.vraw[:, :, pos0:pos0 + c]
        kin = K_cache[:, :, pos0:]
        vin = V_cache[:, :, pos0:]
        R = min(16, c)
        cosc, sinc = cos_c.contiguous(), sin_c.contiguous()
        args_k1 = dict(
            scale=attn.scale, eps=1e-5,
            stride_cb=K_cache.stride(0), stride_ch=K_cache.stride(1),
            stride_ct=K_cache.stride(2),
            q_sb=h * T * dh, q_sh=T * dh, ss_sb=h * T, ss_sh=T,
            h=h, c=c, dh=dh, d=d, R=R, HAS_COMM=pos0 > 0,
            OUT_BF16=u_buf.dtype == torch.bfloat16,
            IEEE=P.qr.dtype == torch.float32)
        kf = P.kflat[:, pos0:]
        vf = P.vflat[:, pos0:]
        gx = P.glogx[:, pos0:]
        RC = min(64, c)
        vin_buf = None
        if mod != "linear":
            vin_buf = torch.empty(N, d, device=u_buf.device,
                                  dtype=u_buf.dtype)
        for _ in range(n_sweeps):
            _attend_head_kernel[(B * h * (c // RC),)](
                q_r, P.vraw[:, :, pos0:], kin, vin, P.sself[:, :, pos0:],
                A_comm, Z_comm, m_comm, o_buf,
                attn.scale,
                K_cache.stride(0), K_cache.stride(1), K_cache.stride(2),
                q_sb=h * T * dh, q_sh=T * dh, ss_sb=h * T, ss_sh=T,
                h=h, c=c, dh=dh, d=d, RC=RC, HAS_COMM=pos0 > 0,
                IEEE=P.qr.dtype == torch.float32, num_warps=4)
            _rms_u_kernel[(B * (c // R),)](
                o_buf, attn.norm_ctx.weight, u_buf, 1e-5,
                c=c, d=d, R=R, OUT_BF16=u_buf.dtype == torch.bfloat16,
                DP=triton.next_power_of_2(d), num_warps=8)
            ctxg = u_buf.reshape(N, d) @ self.W_pack[li].transpose(0, 1)
            if self.knorm_ctx:
                self._ctxnorm_fwd_launch(                    # k-block, in place
                    attn.k_gain, ctxg, attn.n_heads, attn.head_dim)
            vctx = ctxg
            if mod != "linear":
                vctx = self._variant_vctx(attn, P, pos0, ctxg, u_buf, vin_buf)
            if self.vnorm_ctx:
                # v_ctx_norm at the formation site (torch-side); linear's
                # v-block moves onto the VSEP path (ctxg stays pre-norm)
                vctx = attn.norm_vctx(vctx if mod != "linear"
                                      else ctxg[:, d:2 * d])
            if nvr > 0 and npin == 0:
                # gated_norm: zero the kmod head's v_ctx slice (AFTER the
                # norm — eager order, zero last)
                if self.vsep:
                    vctx[:, :nvr * dh] = 0.0
                else:
                    ctxg[:, d:d + nvr * dh] = 0.0
            _refine2_kernel[(N,)](
                kf, vf, ctxg, vctx, gx, cosc, sinc, kin, vin,
                kv_sb=T * d, gx_sb=T * 2 * d, gate_pin=_GATE_PIN,
                tau=self.v_clamp_tau,
                d=d, dh=dh, c=c,
                stride_out_b=K_cache.stride(0), stride_out_h=K_cache.stride(1),
                stride_out_t=K_cache.stride(2),
                OUT_BF16=K_cache.dtype == torch.bfloat16, HALF=d // 2,
                SD=ctxg.shape[1], VSEP=self.vsep, NPIN=npin,
                HALFP=triton.next_power_of_2(d // 2),
                CLAMP=self.v_clamp_tau > 0, num_warps=4)
        if ocat_out is not None:
            ocat_out.copy_(o_buf)
        return o_buf

    # ---- backward ------------------------------------------------------------

    @torch.no_grad()
    def cell_backward(self, li, P, x_c, cos_c, sin_c, K_cache, V_cache, pos0,
                      n_sweeps, dK_rows, dV_rows, grads,
                      dK_gc, dV_gc, scratch, dW_pack_buf, dnormw_buf,
                      dW_vctx_buf=None):
        model = self.model
        blk = model.blocks[li]
        attn = blk.attn
        cfg = self.cfg
        B, c, d = x_c.shape
        h, dh = cfg.n_heads, cfg.head_dim
        N = B * c
        T = P.qr.shape[2]
        ieee = x_c.dtype == torch.float32
        mod = self.mod
        nvr = getattr(cfg, "kmod_vraw_heads", 0)
        npin = nvr * dh if getattr(cfg, "kmod_mode", "raw") == "raw" else 0
        lp = torch.float32 if ieee else torch.bfloat16
        dA, dZ, dM, dkin, dvin, do_buf = scratch.views(B, h, c, dh, d)
        dA.zero_(); dZ.zero_(); dM.zero_()
        q_r = P.qr[:, :, pos0:pos0 + c]
        cosc, sinc = cos_c.contiguous(), sin_c.contiguous()

        # recompute sweep states + u/o/ctxg
        if pos0 > 0:
            A_comm = torch.empty(B, h, c, dh, device=x_c.device,
                                 dtype=torch.float32)
            Z_comm = torch.empty(B, h, c, device=x_c.device, dtype=torch.float32)
            m_comm = torch.empty_like(Z_comm)
            am = torch.empty(B, h, c, device=x_c.device, dtype=torch.int32)
            _comm_fwd_kernel[(B * h * (c // min(64, c)),)](
                P.qr[:, :, pos0:], K_cache, V_cache, A_comm, Z_comm, m_comm,
                am, attn.scale, pos0,
                K_cache.stride(0), K_cache.stride(1), K_cache.stride(2),
                q_sb=h * T * dh, q_sh=T * dh,
                h=h, c=c, dh=dh, RC=min(64, c), TA=64, IEEE=ieee,
                num_warps=4)
        else:
            A_comm = Z_comm = m_comm = P.sself
        K_in = [P.kself[:, :, pos0:pos0 + c].contiguous()]
        V_in = [P.vraw[:, :, pos0:pos0 + c].contiguous()]
        u_list, o_list, ctxg_list, vctx_list = [], [], [], []
        vin_buf = None
        if mod != "linear":
            vin_buf = torch.empty(N, d, device=x_c.device, dtype=x_c.dtype)
        R = min(16, c)
        args_k1 = dict(
            scale=attn.scale, eps=1e-5,
            q_sb=h * T * dh, q_sh=T * dh, ss_sb=h * T, ss_sh=T,
            h=h, c=c, dh=dh, d=d, R=R, HAS_COMM=pos0 > 0,
            OUT_BF16=x_c.dtype == torch.bfloat16, IEEE=ieee)
        kf = P.kflat[:, pos0:]
        vf = P.vflat[:, pos0:]
        gx = P.glogx[:, pos0:]
        for s in range(n_sweeps):
            u_s = torch.empty(B, c, d, device=x_c.device, dtype=x_c.dtype)
            o_s = torch.empty(B, c, d, device=x_c.device, dtype=torch.float32)
            _attend_head_kernel[(B * h * (c // min(64, c)),)](
                q_r, P.vraw[:, :, pos0:], K_in[s], V_in[s],
                P.sself[:, :, pos0:], A_comm, Z_comm, m_comm, o_s,
                attn.scale, h * c * dh, c * dh, dh,
                q_sb=h * T * dh, q_sh=T * dh, ss_sb=h * T, ss_sh=T,
                h=h, c=c, dh=dh, d=d, RC=min(64, c), HAS_COMM=pos0 > 0,
                IEEE=ieee, num_warps=4)
            _rms_u_kernel[(B * (c // R),)](
                o_s, attn.norm_ctx.weight, u_s, 1e-5,
                c=c, d=d, R=R, OUT_BF16=u_s.dtype == torch.bfloat16,
                DP=triton.next_power_of_2(d), num_warps=8)
            ctxg = u_s.reshape(N, d) @ self.W_pack[li].transpose(0, 1)
            if self.knorm_ctx:
                self._ctxnorm_fwd_launch(                    # k-block, in place
                    attn.k_gain, ctxg, attn.n_heads, attn.head_dim)
            vctx = ctxg
            if mod != "linear":
                vctx = self._variant_vctx(attn, P, pos0, ctxg, u_s, vin_buf)
            if self.vnorm_ctx:
                # v_ctx_norm recompute — same function as the forward cell
                vctx = attn.norm_vctx(vctx if mod != "linear"
                                      else ctxg[:, d:2 * d])
            if nvr > 0 and npin == 0:
                # gated_norm: zero the kmod head's v_ctx slice (same as fwd)
                if self.vsep:
                    vctx[:, :nvr * dh] = 0.0
                else:
                    ctxg[:, d:d + nvr * dh] = 0.0
            Kn = torch.empty(B, h, c, dh, device=x_c.device, dtype=x_c.dtype)
            Vn = torch.empty_like(Kn)
            _refine2_kernel[(N,)](
                kf, vf, ctxg, vctx, gx, cosc, sinc, Kn, Vn,
                kv_sb=T * d, gx_sb=T * 2 * d, gate_pin=_GATE_PIN,
                tau=self.v_clamp_tau,
                d=d, dh=dh, c=c,
                stride_out_b=h * c * dh, stride_out_h=c * dh, stride_out_t=dh,
                OUT_BF16=Kn.dtype == torch.bfloat16, HALF=d // 2,
                SD=ctxg.shape[1], VSEP=self.vsep, NPIN=npin,
                HALFP=triton.next_power_of_2(d // 2),
                CLAMP=self.v_clamp_tau > 0, num_warps=4)
            K_in.append(Kn)
            V_in.append(Vn)
            u_list.append(u_s)
            o_list.append(o_s)
            ctxg_list.append(ctxg)
            vctx_list.append(vctx)

        do_last = P.do_T[:, pos0:pos0 + c]   # from the batched W_O/MLP region

        # reverse sweep loop
        dq_l = P.dq[:, :, pos0:pos0 + c]
        dvraw_l = P.dvraw[:, :, pos0:pos0 + c]
        dss_l = P.dss[:, :, pos0:]
        dkf_l = P.dkflat[:, pos0:]
        dvf_l = P.dvflat[:, pos0:]
        dgx_l = P.dglogx[:, pos0:]
        dK_next, dV_next = _merged(dK_rows), _merged(dV_rows)
        pp = 0
        Wp = self.W_pack[li] if not ieee else self.W_pack[li].float()
        SDv = self.W_pack[li].shape[0]
        d_hoist = dvc = None
        if mod != "linear":
            d_hoist = torch.zeros(B, c, d, device=x_c.device,
                                  dtype=torch.float32)
        if self.vsep:
            dvc = torch.empty(N, d, device=x_c.device, dtype=lp)
        for s in range(n_sweeps - 1, -1, -1):
            dctxg = torch.empty(N, SDv, device=x_c.device, dtype=lp)
            _refine2_bwd_kernel[(N,)](
                dK_next, dV_next, kf, vf, ctxg_list[s], vctx_list[s],
                gx, cosc, sinc,
                dkf_l, dvf_l, dctxg, dvc if dvc is not None else dctxg,
                dgx_l,
                kv_sb=T * d, gx_sb=T * 2 * d, gate_pin=_GATE_PIN,
                tau=self.v_clamp_tau,
                d=d, dh=dh, c=c, N=N, HALF=d // 2,
                CTXG_BF16=lp == torch.bfloat16, SD=SDv,
                VSEP=self.vsep, NPIN=npin,
                HALFP=triton.next_power_of_2(d // 2),
                CLAMP=self.v_clamp_tau > 0, num_warps=4)
            if nvr > 0 and npin == 0:
                # gated_norm: the head's v_ctx is structurally zero (not a
                # function of u) — kill its grad before it reaches dW_pack /
                # du / the variant w_vctx chain (mirrors eager masked-fill)
                if self.vsep:
                    dvc[:, :nvr * dh] = 0.0
                else:
                    dctxg[:, d:d + nvr * dh] = 0.0
            if self.vnorm_ctx:
                # v_ctx_norm backward: dvc is the grad wrt the NORMED v_ctx
                # (the blend backward above used the post-norm candidate);
                # recompute the pre-norm y and push dvc through the norm.
                # Linear: dy re-enters the packed machinery via the dctxg
                # v-block (VSEP zeroed it), closing dW_pack[d:2d] and du.
                if mod != "linear":
                    y = self._variant_vctx(attn, P, pos0, ctxg_list[s],
                                           u_list[s], vin_buf)
                else:
                    y = ctxg_list[s][:, d:2 * d]
                dy = self._vctx_norm_bwd(attn, y, dvc, grads)
                if mod != "linear":
                    dvc = dy.to(lp)
                else:
                    dctxg[:, d:2 * d] = dy.to(lp)
            if self.knorm_ctx:
                # K-norm backward: dctxg[:, :d] is grad wrt the NORMED k_ctx;
                # recompute pre-norm k_ctx = u @ W_pack[:d]^T and push through
                # norm_kctx so dW_pack[:d] / du see the pre-norm grad + k_gain.
                ykc = u_list[s].reshape(N, d) @ Wp[:d].transpose(0, 1)
                # fused VJP writes dx IN PLACE into dctxg[:, :d] (lp dtype) +
                # accumulates the k_gain grad — no reassign/cast copy
                self._kctx_norm_bwd(attn, ykc, dctxg[:, :d], grads)
            if mod != "linear":
                # route the v_ctx grad through the variant chain instead of
                # the packed w_vctx GEMM (the kernel zeroed the dead v_ctx
                # block, so dW_pack[d:2d] stays 0 and du sees no leak); the
                # +u shortcut folds into the du GEMM as the addmm bias.
                d_u_extra = self._variant_bwd_sweep(
                    attn, P, pos0, ctxg_list[s], dctxg, dvc, u_list[s],
                    vin_buf, d_hoist, dW_vctx_buf)
                if d_u_extra is not None:
                    # identity carry is lam*u -> its grad into du scales by lam
                    du = torch.addmm(d_u_extra, dctxg, Wp,
                                     beta=self.mult_res_lambda)
                else:
                    du = dctxg @ Wp
            else:
                du = (dctxg @ Wp).reshape(N, d)
            dW_pack_buf += dctxg.transpose(0, 1) @ u_list[s].reshape(N, d)
            if self.deterministic_bwd:
                # per-program RMS-weight partials -> fixed-order torch sum
                nprog = B * (c // R)
                dwpart = torch.zeros(nprog, d, device=x_c.device,
                                     dtype=torch.float32)
                _k1b_rms_det_kernel[(nprog,)](
                    du, o_list[s], attn.norm_ctx.weight,
                    do_buf, dwpart, 1e-5, c=c, d=d, R=R,
                    DP=triton.next_power_of_2(d), num_warps=8)
                dnormw_buf += dwpart.sum(0)
            else:
                _k1b_rms_kernel[(B * (c // R),)](
                    du, o_list[s], attn.norm_ctx.weight,
                    do_buf, dnormw_buf, 1e-5, c=c, d=d, R=R,
                    DP=triton.next_power_of_2(d), num_warps=8)
            if s == n_sweeps - 1:
                do_buf += do_last
            RCb = min(64, c)
            nt_b = c // RCb
            if self.deterministic_bwd and nt_b > 1:
                # split chunk: each row tile writes its own plane, fixed sum
                dkin_d = torch.zeros(nt_b, N, d, device=x_c.device,
                                     dtype=torch.float32)
                dvin_d = torch.zeros(nt_b, N, d, device=x_c.device,
                                     dtype=torch.float32)
                _k1b_attn_det_kernel[(B * h * nt_b,)](
                    do_buf, o_list[s], q_r, P.vraw[:, :, pos0:], K_in[s],
                    V_in[s], P.sself[:, :, pos0:], A_comm, Z_comm, m_comm,
                    P.dq[:, :, pos0:], P.dvraw[:, :, pos0:], dkin_d, dvin_d,
                    P.dss[:, :, pos0:], dA, dZ, dM, attn.scale,
                    q_sb=h * T * dh, q_sh=T * dh, ss_sb=h * T, ss_sh=T,
                    ND=N * d,
                    h=h, c=c, dh=dh, d=d, RC=RCb, HAS_COMM=pos0 > 0,
                    IEEE=ieee, num_warps=8)
                dkin[pp].copy_(dkin_d.sum(0))
                dvin[pp].copy_(dvin_d.sum(0))
            else:
                # nt_b == 1 => the non-det kernel already takes its plain-store
                # (deterministic) branch; otherwise this IS the non-det path
                if c > RCb:
                    dkin[pp].zero_()      # atomic accumulation across row tiles
                    dvin[pp].zero_()
                _k1b_attn_kernel[(B * h * (c // RCb),)](
                    do_buf, o_list[s], q_r, P.vraw[:, :, pos0:], K_in[s],
                    V_in[s], P.sself[:, :, pos0:], A_comm, Z_comm, m_comm,
                    P.dq[:, :, pos0:], P.dvraw[:, :, pos0:], dkin[pp], dvin[pp],
                    P.dss[:, :, pos0:], dA, dZ, dM, attn.scale,
                    q_sb=h * T * dh, q_sh=T * dh, ss_sb=h * T, ss_sh=T,
                    h=h, c=c, dh=dh, d=d, RC=RCb, HAS_COMM=pos0 > 0, IEEE=ieee,
                    num_warps=8)
            dK_next, dV_next = dkin[pp], dvin[pp]
            pp ^= 1

        if mod in ("mult", "mult_res"):
            P.d_sig[:, pos0:pos0 + c] += d_hoist
        elif mod == "fusion":
            P.d_prex[:, pos0:pos0 + c] += d_hoist

        # sweep-0 rows are (rotated self key, raw value): route into layer accums
        P.dkself[:, :, pos0:pos0 + c] += \
            dK_next.reshape(B, c, h, dh).transpose(1, 2)
        P.dvraw[:, :, pos0:pos0 + c] += \
            dV_next.reshape(B, c, h, dh).transpose(1, 2)

        # committed-accumulator backward, flash-style (no [c,a] tensors)
        if pos0 > 0:
            dm_t = torch.empty(B, h, c, device=x_c.device, dtype=torch.float32)
            RCc = min(64, c)
            nt_c = c // RCc
            if self.deterministic_bwd and nt_c > 1:
                # split chunk: per-row-tile compact planes, fixed-order sum
                a = pos0
                dkc_d = torch.zeros(nt_c, B, h, a, dh, device=x_c.device,
                                    dtype=torch.float32)
                dvc_d = torch.zeros(nt_c, B, h, a, dh, device=x_c.device,
                                    dtype=torch.float32)
                _comm_bwd_det_kernel[(B * h * nt_c,)](
                    P.qr[:, :, pos0:], K_cache, V_cache, m_comm, dA, dZ, dM,
                    P.dq[:, :, pos0:], dkc_d, dvc_d, dm_t, attn.scale, pos0,
                    K_cache.stride(0), K_cache.stride(1), K_cache.stride(2),
                    gc_sb=h * a * dh, gc_sh=a * dh, gc_st=dh, plane=B * h * a * dh,
                    q_sb=h * T * dh, q_sh=T * dh,
                    h=h, c=c, dh=dh, RC=RCc, TA=64, IEEE=ieee, num_warps=4)
                dK_gc[:, :, :pos0] += dkc_d.sum(0)
                dV_gc[:, :, :pos0] += dvc_d.sum(0)
            else:
                _comm_bwd_kernel[(B * h * (c // RCc),)](
                    P.qr[:, :, pos0:], K_cache, V_cache, m_comm, dA, dZ, dM,
                    P.dq[:, :, pos0:], dK_gc, dV_gc, dm_t, attn.scale, pos0,
                    K_cache.stride(0), K_cache.stride(1), K_cache.stride(2),
                    gc_sb=dK_gc.stride(0), gc_sh=dK_gc.stride(1),
                    q_sb=h * T * dh, q_sh=T * dh,
                    h=h, c=c, dh=dh, RC=RCc, TA=64, IEEE=ieee,
                    num_warps=4)
            # running-max subgradient correction at the argmax positions
            am_l = am.long().unsqueeze(-1)
            Kg = K_cache[:, :, :pos0].gather(
                2, am_l.expand(-1, -1, -1, dh)).float()
            P.dq[:, :, pos0:pos0 + c] += \
                (dm_t * attn.scale).unsqueeze(-1) * Kg
            scatter_src = (dm_t * attn.scale).unsqueeze(-1) * q_r.float()
            if self.deterministic_bwd:
                # CUDA scatter_add_ is atomic (non-det); use the det kernel
                _det_scatter_add_(dK_gc[:, :, :pos0], 2,
                                  am_l.expand(-1, -1, -1, dh), scatter_src)
            else:
                dK_gc[:, :, :pos0].scatter_add_(
                    2, am_l.expand(-1, -1, -1, dh), scatter_src)

    # ---- per-layer W_O/MLP backward (batched, uses fwd-saved o_cat) ---------

    def layer_blockmlp_bwd(self, li, x_l, ocat_l, dx_next, dx_out, P, grads):
        blk = self.model.blocks[li]
        attn = blk.attn
        params_blk = [attn.wo.weight, blk.mlp_norm.weight, blk.mlp.w1.weight,
                      blk.mlp.w2.weight, blk.mlp.w3.weight]
        x_leaf = x_l.detach().requires_grad_(True)
        o_leaf = ocat_l.detach().requires_grad_(True)
        with torch.enable_grad():
            x_mid = x_leaf + attn.wo(o_leaf)
            x_next = x_mid + blk.mlp(blk.mlp_norm(x_mid))
        gs = torch.autograd.grad([x_next], [x_leaf, o_leaf] + params_blk,
                                 [dx_next.to(x_next.dtype)])
        dx_out.copy_(gs[0].float())
        P.do_T.copy_(gs[1].float())
        for p, g in zip(params_blk, gs[2:]):
            grads.add(p, g)

    # ---- per-layer closing region -------------------------------------------

    def layer_close(self, li, x_l, P, dx_add, grads):
        """One autograd region over full T: attn_norm, W_Q/K/V, rope, s_self,
        glog_x. Adds into dx_add (the dx[li] buffer) and param grads."""
        model = self.model
        blk = model.blocks[li]
        attn = blk.attn
        B, T, d = x_l.shape
        cos = model.rope_cos[:T]
        sin = model.rope_sin[:T]
        lora = attn.w_gk.weight.shape[1] != 2 * d   # LoRAGate (gate_rank>0)
        if lora:
            gate_params = [attn.w_gk.a.weight, attn.w_gk.b.weight,
                           attn.w_gv.a.weight, attn.w_gv.b.weight,
                           attn.w_gk.b.bias, attn.w_gv.b.bias]
        else:
            gate_params = [attn.w_gk.weight, attn.w_gv.weight,
                           attn.w_gk.bias, attn.w_gv.bias]
        attn_params = [blk.attn_norm.weight, attn.wq.weight, attn.wk.weight,
                       attn.wv.weight] + gate_params
        if getattr(attn.cfg, "v_norm", "none") == "rms":
            attn_params.append(attn.v_gain)
        x_leaf = x_l.detach().requires_grad_(True)
        with torch.enable_grad():
            xhat = blk.attn_norm(x_leaf)
            q_ = apply_rope(attn._split(attn.wq(xhat)), cos, sin)
            kf_ = attn.wk(xhat)
            # project_v == layer_proj's vf: dvf_total (dvflat + merged dvraw)
            # is the grad wrt the NORMED v, so autograd here runs the norm's
            # backward (v_gain, wv, xhat) as part of the recomputed graph
            vf_ = attn.project_v(xhat)
            ks_ = apply_rope(attn._split(kf_), cos, sin)
            ss_ = ((q_ * ks_).sum(-1) * attn.scale).float()
            if lora:
                # x-half of the gate pre-activations through the live
                # factors: autograd delivers a-weight (x-columns) and
                # b-weight/bias grads here; the u-half chains through
                # dW_pack in _bwd_epilogue_body.
                xf_ = xhat.reshape(B * T, d)
                gk_ = attn.w_gk.b(xf_ @ attn.w_gk.a.weight[:, :d]
                                  .transpose(0, 1))
                gv_ = attn.w_gv.b(xf_ @ attn.w_gv.a.weight[:, :d]
                                  .transpose(0, 1))
                gx_ = torch.cat([gk_, gv_], 1)
            else:
                Wgx_ = torch.cat([attn.w_gk.weight[:, :d],
                                  attn.w_gv.weight[:, :d]], 0)
                bg_ = torch.cat([attn.w_gk.bias, attn.w_gv.bias], 0)
                gx_ = torch.addmm(bg_, xhat.reshape(B * T, d),
                                  Wgx_.transpose(0, 1))
        if self.mod != "linear":
            # deferred hoisted-term chain: the per-sweep d_sig/d_pre grads
            # were summed into P.d_sig/P.d_prex by cell_backward; the sigmoid
            # chain + GEMMs to v_gate / v_pre-xhat-half / xhat run ONCE per
            # layer here. Must precede grad_outs (feeds P.d_xhat).
            xf = xhat.detach().reshape(B * T, d)
            if self.mod in ("mult", "mult_res"):
                sg = torch.sigmoid(attn.v_gate(xf).float())
                d_pre = P.d_sig.reshape(B * T, d) * sg * (1.0 - sg)
                # v_gate weight grad: dense buffer, or projected onto the LoRA
                # factors (ctx_rank>0). The bias (attn.v_gate.bias == up.bias
                # for the factored case, a real Parameter) is unchanged.
                _accum_ctx_dW(grads, attn.v_gate,
                              d_pre.transpose(0, 1) @ xf.float(),
                              getattr(attn.cfg, "ctx_rank", 0) > 0)
                grads.buf[id(attn.v_gate.bias)] += d_pre.sum(0)
                P.d_xhat += (d_pre @ attn.v_gate.weight.float()
                             ).reshape(B, T, d)
            else:  # fusion
                d_pre = P.d_prex.reshape(B * T, d)
                grads.buf[id(attn.v_pre.weight)][:, :d] += \
                    d_pre.transpose(0, 1) @ xf.float()
                P.d_xhat += (d_pre @ attn.v_pre.weight[:, :d].float()
                             ).reshape(B, T, d)
        dvf_total = P.dvflat + _merged(P.dvraw).reshape(B, T, d)
        outputs = [q_, kf_, vf_, ks_, ss_, gx_]
        grad_outs = [P.dq.to(q_.dtype), P.dkflat.to(kf_.dtype),
                     dvf_total.to(vf_.dtype), P.dkself.to(ks_.dtype), P.dss,
                     P.dglogx.reshape(B * T, 2 * d).to(gx_.dtype)]
        if self.mod != "linear":
            # ctx_mod v-branch grad wrt xhat (accumulated by cell_backward)
            # re-enters here, so attn_norm / attn_norm.weight see it too.
            outputs = outputs + [xhat]
            grad_outs = grad_outs + [P.d_xhat.to(xhat.dtype)]
        gs = torch.autograd.grad(outputs, [x_leaf] + attn_params, grad_outs)
        dx_add += gs[0].float()
        for p, g in zip(attn_params, gs[1:]):
            grads.add(p, g)
