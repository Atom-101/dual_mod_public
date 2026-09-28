"""Attention kernels for the Wave3DX window attention (engines/wave3d/wave3d_x.py).

Operator: the Lq window queries q [B, h, Lq, dh] attend the Lk = [committed
prefix ++ window] keys k/v [B, h, Lk, dh] causally at ABSOLUTE positions: query
i sees keys j <= i + (Lk - Lq) ("lower-right" causal). Inputs are strided views
of the time-major caches (q: [Lq, B, h, dh] permuted; k/v: Kbuf[:Lk] permuted).

Each kernel exposes
    fwd(q, k, v, scale)                     -> (o, lse, extra)
    bwd(dout, q, k, v, o, lse, extra, scale) -> (dq, dk, dv)   [B, h, L, dh] layouts
so the engine's _AttendGroup stashes (o, lse, extra) per (step, layer) and runs
the backward off the stash for any kernel.

  * FlashKernel  : aten._scaled_dot_product_flash_attention (FA2-generation
                   sm80 code on Blackwell, ~250 TFLOP/s here). is_causal with
                   Lq < Lk IS lower-right aligned for this op (verified vs math).
  * CudnnKernel  : cuDNN's Blackwell-native SDPA (sm100 fprop/bprop kernels,
                   ~2.5x the flash kernels' throughput). torch's cuDNN SDPA
                   binding only exposes the TOP-LEFT causal mask (is_causal with
                   Lq < Lk silently computes the wrong operator: verified rel err
                   0.94 vs lower-right, 2e-3 vs top-left), so the operator is
                   run as prefix-DENSE [Lq x P] + window-CAUSAL [Lq x Lq] (both
                   cuDNN-supported) and merged by an online-softmax (LSE) merge
                   in one Triton kernel that also emits the merged lse.
                   BACKWARD: both halves' cuDNN backward are called with the
                   MERGED (o, lse) -- the kernel recomputes P = exp(S - lse) and
                   delta = rowsum(dO * O) from them, which is exactly the
                   full-row softmax backward split over two key ranges (the
                   flash-decoding / ring-attention identity). This is what the
                   previous _AttendCacheCudnn got wrong: it differentiated
                   through the merge with autograd, and aten's cuDNN backward
                   DROPS the gradient w.r.t. its logsumexp output => dq/dk were
                   biased (rel err 3-5e-2 vs 2e-3; the "noisier grads").
                   Kernel-dispatch note: with Lq <= 64 cuDNN picks an sm80 wmma
                   kernel (small), fine.
"""

import torch
import triton
import triton.language as tl

try:                                   # nvidia-cudnn-frontend (pure binding onto torch's bundled libcudnn)
    import cudnn as _cudnn_fe
except ImportError:                    # pragma: no cover
    _cudnn_fe = None

aten = torch.ops.aten


# ----------------------------------------------------------------------------- flash
class FlashKernel:
    name = "flash"
    out_ok = False                         # cannot write o / dq into caller buffers

    @staticmethod
    def fwd(q, k, v, scale, out=None):
        out, lse, cq, ck, mq, mk, seed, off, _ = aten._scaled_dot_product_flash_attention(
            q, k, v, 0.0, True, False, scale=scale)
        return out, lse, (cq, ck, mq, mk, seed, off)

    @staticmethod
    def bwd(dout, q, k, v, out, lse, extra, scale, dq_out=None):
        cq, ck, mq, mk, seed, off = extra
        dq, dk, dv = aten._scaled_dot_product_flash_attention_backward(
            dout.contiguous(), q, k, v, out, lse, cq, ck, mq, mk, 0.0, True, seed, off, scale=scale)
        return dq, [(0, dk.permute(2, 0, 1, 3))], [(0, dv.permute(2, 0, 1, 3))]


def accum_kv(pieces, gbuf, row0, r1, commit):
    """Reverse-time prefix/commit bookkeeping of _AttendCache (see wave3d.py) over a
    time-major key/value grad given as PIECES [(start_row, g_tm [rows, B, h, dh])]
    covering [0, r1) in order (flash: one piece; cuDNN split: prefix + window).
        gbuf[:row0] += g[:row0]                 (owed to the committed prefix)
        slab = g[row0:r1]; slab[:commit] += gbuf[row0:row0+commit]
    Returns the slab grad [r1-row0, B, h, dh] (a view when one piece spans it)."""
    if len(pieces) == 1:
        g = pieces[0][1]
        if row0 > 0:
            gbuf[:row0] += g[:row0]
        slab = g[row0:r1]
    else:
        slab = None
        for s0, g in pieces:
            s1 = s0 + g.shape[0]
            if s0 < row0:                          # prefix part of this piece
                gbuf[s0:min(s1, row0)] += g[:min(s1, row0) - s0]
            if s1 > row0:                          # slab part of this piece
                if slab is None:
                    slab = torch.empty((r1 - row0,) + tuple(g.shape[1:]), device=g.device, dtype=g.dtype)
                a = max(s0, row0)
                slab[a - row0:s1 - row0].copy_(g[a - s0:])
    if commit > 0:
        slab[:commit] += gbuf[row0:row0 + commit]
    return slab


# ----------------------------------------------------------------------------- LSE merge
@triton.jit
def _merge_kernel(oA, oB, lA, lB, O, LSE, Lq,
                  sAb, sAh, sAl, sBb, sBh, sBl, sOb, sOh, sOl,
                  H: tl.constexpr, DH: tl.constexpr, R: tl.constexpr):
    """Rows [r0, r0+R) of head (b, h): o = (eA*oA + eB*oB)/(eA+eB), lse = m + log(eA+eB).
    lse tensors are [B, h, Lq] contiguous (fp32); o tensors have arbitrary row strides."""
    pid = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    rows = pid * R + tl.arange(0, R)
    cols = tl.arange(0, DH)
    m_ok = rows < Lq
    lrow = bh * Lq + rows
    la = tl.load(lA + lrow, mask=m_ok, other=0.0)
    lb = tl.load(lB + lrow, mask=m_ok, other=0.0)
    m = tl.maximum(la, lb)
    ea = tl.exp(la - m)
    eb = tl.exp(lb - m)
    den = ea + eb
    tl.store(LSE + lrow, m + tl.log(den), mask=m_ok)
    offA = b * sAb + h * sAh + rows[:, None] * sAl + cols[None, :]
    offB = b * sBb + h * sBh + rows[:, None] * sBl + cols[None, :]
    offO = b * sOb + h * sOh + rows[:, None] * sOl + cols[None, :]
    mk = m_ok[:, None] & (cols[None, :] < DH)
    a = tl.load(oA + offA, mask=mk, other=0.0).to(tl.float32)
    c = tl.load(oB + offB, mask=mk, other=0.0).to(tl.float32)
    o = (a * ea[:, None] + c * eb[:, None]) / den[:, None]
    tl.store(O + offO, o.to(O.dtype.element_ty), mask=mk)


def lse_merge(oA, lseA, oB, lseB, out_like, out=None):
    """Online-softmax merge of two attention partials over disjoint key ranges.
    oA/oB [B, h, Lq, dh] (dh-contiguous), lseA/lseB [B, h, Lq, 1] fp32.
    Returns (o, lse): o with the strides of out_like (the query view => o inherits
    the time-major layout the engine reshapes for free), lse [B, h, Lq, 1] fp32.
    out=: write o there (may alias oA: each program reads its rows before writing them)."""
    B, h, Lq, dh = oA.shape
    o = torch.empty_like(out_like) if out is None else out
    lse = torch.empty(B, h, Lq, 1, device=oA.device, dtype=torch.float32)
    lA = lseA.reshape(B, h, Lq).contiguous()
    lB = lseB.reshape(B, h, Lq).contiguous()
    R = 16
    grid = (triton.cdiv(Lq, R), B * h)
    assert oA.stride(3) == 1 and oB.stride(3) == 1 and o.stride(3) == 1
    _merge_kernel[grid](oA, oB, lA, lB, o, lse, Lq,
                        oA.stride(0), oA.stride(1), oA.stride(2),
                        oB.stride(0), oB.stride(1), oB.stride(2),
                        o.stride(0), o.stride(1), o.stride(2),
                        H=h, DH=dh, R=R, num_warps=4)
    return o, lse


# ----------------------------------------------------------------------------- raw-self partial
def self_merge(o_main, lse_main, q, k_self, v_self, scale, out=None):
    """RAW-DIAGONAL operator (2026-09-08): the window attention runs over keys j < pos
    (slab minus its last row => bottom-right causal excludes the diagonal), and the
    query's OWN raw (rotated key, value) enters as a one-key partial merged here:
        s_self = <q, k_self> * scale ; o = (e^lse o_main + e^s v_self) / (e^lse + e^s).
    q/k_self/v_self [B, h, Lq, dh] (dh-contiguous). Returns (o, lse) exactly like the
    kernel would for the full row -> the kernel backward runs unchanged on (o, lse)."""
    s_self = (q.float() * k_self.float()).sum(-1, keepdim=True) * scale         # [B,h,Lq,1] fp32
    return lse_merge(o_main, lse_main, v_self, s_self, o_main if out is None else out, out=out), s_self


def self_bwd(dout, q, k_self, v_self, o, lse, s_self, scale):
    """Gradient of the raw-self partial given the MERGED (o, lse): the same split-softmax
    identity the prefix/window halves use -- P = exp(s - lse), delta = rowsum(dO*O),
    dS = P (rowsum(dO*v) - delta); dq += dS k scale; dk = dS q scale; dv = P dO."""
    P = torch.exp(s_self - lse)                                                   # [B,h,Lq,1] fp32
    do = dout.float()
    delta = (do * o.float()).sum(-1, keepdim=True)
    dP = (do * v_self.float()).sum(-1, keepdim=True)
    dS = P * (dP - delta) * scale
    dq = (dS * k_self.float()).to(q.dtype)
    dk = (dS * q.float()).to(q.dtype)
    dv = (P * do).to(q.dtype)
    return dq, dk, dv


# ----------------------------------------------------------------------------- fused raw-self (Triton)
# One kernel per direction for the raw-diagonal operator's glue (self_merge/self_bwd + rope of the
# raw key), bf16 in / fp32 math. Unfused this was ~30 full-window elementwise passes per layer per
# engine step (~1.3 GB) and cost +65% step time at K16 (2026-09-09 A/B); fused: ~6 tensor reads.
@triton.jit
def _self_fwd_kernel(Q, KF, VF, COS, SIN, O, LSE, SS, Wv, scale,
                     sq_b, sq_h, sq_t, sk_b, sk_h, sk_t, sv_b, sv_h, sv_t, so_b, so_h, so_t,
                     sl_b, sl_h, sl_t, sc_t, H: tl.constexpr, DH: tl.constexpr, R: tl.constexpr):
    pid_t = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    HD: tl.constexpr = DH // 2
    d1 = tl.arange(0, HD)
    d2 = d1 + HD
    for r in range(R):
        t = pid_t * R + r
        if t < Wv:
            qb = Q + b * sq_b + h * sq_h + t * sq_t
            kb = KF + b * sk_b + h * sk_h + t * sk_t
            vb = VF + b * sv_b + h * sv_h + t * sv_t
            ob = O + b * so_b + h * so_h + t * so_t
            lb = LSE + b * sl_b + h * sl_h + t * sl_t
            c = tl.load(COS + t * sc_t + d1).to(tl.float32)
            sn = tl.load(SIN + t * sc_t + d1).to(tl.float32)
            k1 = tl.load(kb + d1).to(tl.float32)
            k2 = tl.load(kb + d2).to(tl.float32)
            ks1 = k1 * c - k2 * sn
            ks2 = k2 * c + k1 * sn
            q1 = tl.load(qb + d1).to(tl.float32)
            q2 = tl.load(qb + d2).to(tl.float32)
            s_self = (tl.sum(q1 * ks1, 0) + tl.sum(q2 * ks2, 0)) * scale
            lm = tl.load(lb).to(tl.float32)
            m = tl.maximum(lm, s_self)
            wA = tl.exp(lm - m)
            wB = tl.exp(s_self - m)
            den = wA + wB
            o1 = tl.load(ob + d1).to(tl.float32)
            o2 = tl.load(ob + d2).to(tl.float32)
            v1 = tl.load(vb + d1).to(tl.float32)
            v2 = tl.load(vb + d2).to(tl.float32)
            tl.store(ob + d1, ((wA * o1 + wB * v1) / den).to(O.dtype.element_ty))
            tl.store(ob + d2, ((wA * o2 + wB * v2) / den).to(O.dtype.element_ty))
            tl.store(lb, m + tl.log(den))
            tl.store(SS + b * sl_b + h * sl_h + t * sl_t, s_self)


@triton.jit
def _self_bwd_kernel(DO, Q, KF, VF, COS, SIN, O, LSE, SS, DQ, DKF, DVF, Wv, scale,
                     sdo_b, sdo_h, sdo_t, sq_b, sq_h, sq_t, sk_b, sk_h, sk_t, sv_b, sv_h, sv_t,
                     so_b, so_h, so_t, sl_b, sl_h, sl_t, sdq_b, sdq_h, sdq_t, sdk_b, sdk_h, sdk_t,
                     sc_t, H: tl.constexpr, DH: tl.constexpr, R: tl.constexpr):
    pid_t = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    HD: tl.constexpr = DH // 2
    d1 = tl.arange(0, HD)
    d2 = d1 + HD
    for r in range(R):
        t = pid_t * R + r
        if t < Wv:
            dob = DO + b * sdo_b + h * sdo_h + t * sdo_t
            qb = Q + b * sq_b + h * sq_h + t * sq_t
            kb = KF + b * sk_b + h * sk_h + t * sk_t
            vb = VF + b * sv_b + h * sv_h + t * sv_t
            ob = O + b * so_b + h * so_h + t * so_t
            lb = LSE + b * sl_b + h * sl_h + t * sl_t
            dqb = DQ + b * sdq_b + h * sdq_h + t * sdq_t
            dkb = DKF + b * sdk_b + h * sdk_h + t * sdk_t
            dvb = DVF + b * sdk_b + h * sdk_h + t * sdk_t
            c = tl.load(COS + t * sc_t + d1).to(tl.float32)
            sn = tl.load(SIN + t * sc_t + d1).to(tl.float32)
            k1 = tl.load(kb + d1).to(tl.float32)
            k2 = tl.load(kb + d2).to(tl.float32)
            ks1 = k1 * c - k2 * sn
            ks2 = k2 * c + k1 * sn
            do1 = tl.load(dob + d1).to(tl.float32)
            do2 = tl.load(dob + d2).to(tl.float32)
            o1 = tl.load(ob + d1).to(tl.float32)
            o2 = tl.load(ob + d2).to(tl.float32)
            v1 = tl.load(vb + d1).to(tl.float32)
            v2 = tl.load(vb + d2).to(tl.float32)
            P = tl.exp(tl.load(SS + b * sl_b + h * sl_h + t * sl_t) - tl.load(lb))
            delta = tl.sum(do1 * o1, 0) + tl.sum(do2 * o2, 0)
            dP = tl.sum(do1 * v1, 0) + tl.sum(do2 * v2, 0)
            dS = P * (dP - delta) * scale
            q1 = tl.load(qb + d1).to(tl.float32)
            q2 = tl.load(qb + d2).to(tl.float32)
            # dq += dS * k_self (rotated); dk_rot = dS * q -> back-rotate (rope^T = rope with -sin)
            tl.store(dqb + d1, (tl.load(dqb + d1).to(tl.float32) + dS * ks1).to(DQ.dtype.element_ty))
            tl.store(dqb + d2, (tl.load(dqb + d2).to(tl.float32) + dS * ks2).to(DQ.dtype.element_ty))
            dk1 = dS * q1
            dk2 = dS * q2
            tl.store(dkb + d1, (dk1 * c + dk2 * sn).to(DKF.dtype.element_ty))
            tl.store(dkb + d2, (dk2 * c - dk1 * sn).to(DKF.dtype.element_ty))
            tl.store(dvb + d1, (P * do1).to(DVF.dtype.element_ty))
            tl.store(dvb + d2, (P * do2).to(DVF.dtype.element_ty))


def _bhts(x):
    """strides (b, h, t) of a [B, h, Wv, dh] view (dh must be contiguous)."""
    assert x.stride(3) == 1, x.stride()
    return x.stride(0), x.stride(1), x.stride(2)


def self_merge_fused(o, lse, q, kf, vf, cos, sin, scale):
    """In-place raw-self merge: o [B,h,Wv,dh] (the window-part output, MERGED in place), lse
    [B,h,Wv,1] fp32 (merged in place); q/kf/vf [B,h,Wv,dh] views (kf = RAW UNROTATED key; the
    rope by cos/sin [Wv, dh/2] happens inside). Returns s_self [B,h,Wv,1] fp32."""
    B, h, Wv, dh = q.shape
    ss = torch.empty_like(lse)
    cos = cos.reshape(Wv, dh // 2); sin = sin.reshape(Wv, dh // 2)
    R = 8
    grid = (triton.cdiv(Wv, R), B * h)
    _self_fwd_kernel[grid](q, kf, vf, cos, sin, o, lse, ss, Wv, scale,
                           *_bhts(q), *_bhts(kf), *_bhts(vf), *_bhts(o),
                           lse.stride(0), lse.stride(1), lse.stride(2), cos.stride(0),
                           H=h, DH=dh, R=R, num_warps=2)
    return ss


def self_bwd_fused(dout, q, kf, vf, cos, sin, o, lse, s_self, scale, dq, dkf, dvf):
    """dq [B,h,Wv,dh] += dq_self (in place); dkf/dvf [B,h,Wv,dh] views WRITTEN (raw-space key grad
    = back-rotated, raw value grad)."""
    B, h, Wv, dh = q.shape
    cos = cos.reshape(Wv, dh // 2); sin = sin.reshape(Wv, dh // 2)
    assert dkf.stride() == dvf.stride()
    R = 8
    grid = (triton.cdiv(Wv, R), B * h)
    _self_bwd_kernel[grid](dout, q, kf, vf, cos, sin, o, lse, s_self, dq, dkf, dvf, Wv, scale,
                           *_bhts(dout), *_bhts(q), *_bhts(kf), *_bhts(vf), *_bhts(o),
                           lse.stride(0), lse.stride(1), lse.stride(2), *_bhts(dq), *_bhts(dkf),
                           cos.stride(0), H=h, DH=dh, R=R, num_warps=2)


# ----------------------------------------------------------------------------- cuDNN
def _cudnn_fwd(q, k, v, causal, scale):
    r = aten._scaled_dot_product_cudnn_attention(q, k, v, None, True, 0.0, causal, False, scale=scale)
    return r[0], r[1], r[6], r[7]            # out, lse [B,h,Lq,1] fp32, philox seed/offset


def _cudnn_bwd(dout, q, k, v, out, lse, seed, off, causal, scale):
    Lq, Lk = q.shape[2], k.shape[2]
    return aten._scaled_dot_product_cudnn_attention_backward(
        dout, q, k, v, out, lse, seed, off, None, None, None, Lq, Lk, 0.0, causal, scale=scale)


class CudnnKernel:
    name = "cudnn"
    out_ok = False

    @staticmethod
    def fwd(q, k, v, scale, out=None):
        Lq, Lk = q.shape[2], k.shape[2]
        P = Lk - Lq
        oB, lseB, seed, off = _cudnn_fwd(q, k[:, :, P:], v[:, :, P:], True, scale)
        if P == 0:
            return oB, lseB, (seed, off)
        oA, lseA, _, _ = _cudnn_fwd(q, k[:, :, :P], v[:, :, :P], False, scale)
        o, lse = lse_merge(oA, lseA, oB, lseB, q)
        return o, lse, (seed, off)

    @staticmethod
    def bwd(dout, q, k, v, out, lse, extra, scale, dq_out=None):
        seed, off = extra
        Lq, Lk = q.shape[2], k.shape[2]
        P = Lk - Lq
        if dout.stride() != out.stride():
            dout = torch.empty_like(out).copy_(dout)     # cuDNN wants dO in O's layout
        dqB, dkB, dvB = _cudnn_bwd(dout, q, k[:, :, P:], v[:, :, P:], out, lse, seed, off, True, scale)
        tm = lambda g: g.permute(2, 0, 1, 3)
        if P == 0:
            return dqB, [(0, tm(dkB))], [(0, tm(dvB))]
        dqA, dkA, dvA = _cudnn_bwd(dout, q, k[:, :, :P], v[:, :, :P], out, lse, seed, off, False, scale)
        dqA += dqB
        return dqA, [(0, tm(dkA)), (P, tm(dkB))], [(0, tm(dvA)), (P, tm(dvB))]


# ----------------------------------------------------------------------------- cuDNN frontend (direct)
class CudnnFEKernel:
    """cuDNN SDPA through the cuDNN *frontend* python bindings (nvidia-cudnn-frontend;
    same libcudnn 9.20 torch bundles) with the BOTTOM-RIGHT causal mask torch's
    binding does not expose: ONE graph for the whole lower-right window
    attention (no prefix/window split, no LSE merge, no grad cat) => the lowest
    GPU time (fwd 144 / bwd 450 us at 1024x4096 vs 175/502 split, 407/1217
    flash) AND the lowest CPU cost per call (one execute() each way) -- the
    engine's forward is launch-bound, so CPU per call is what decides the
    in-engine number. Execution plans are cached per (shape, strides) key (~100
    per K; ~0.1 s build each on first sight); ONE workspace per device that only
    ever grows -- superseded (smaller) buffers are kept ALIVE in _ws_old because
    captured CUDA graphs bake the workspace address they were captured with (the
    K8 graphs replayed after a K16 capture grew and freed the buffer -> illegal
    memory access), growth inside a capture is refused (it would allocate from
    the graph pool), and the engine pre-reserves an upper bound (reserve()) at
    buffer allocation so no growth happens after the first capture at all; the
    handle's stream is set to the current torch stream at every call (CUDA
    graph capture safe). dK/dV are written straight into time-major
    [Lk, B, h, dh] buffers (the layout accum_kv / the engine's grad caches want)."""
    name = "cudnn_fe"
    out_ok = True                          # fwd(out=) / bwd(dq_out=): o and dq land in caller buffers
    _handles = {}
    _plans = {}
    _ws = {}                               # dev -> current (largest) workspace
    _ws_old = {}                           # dev -> superseded workspaces, kept alive (graphs bake addresses)
    WS_BYTES_PER_ROW_HEAD = 260            # measured plan workspace: 18.3/36.6/73.1 MiB at K8/16/32 (B4, 36 heads)

    @classmethod
    def reserve(cls, dev, nbytes):
        """Pre-size the workspace (call BEFORE any graph capture; see class doc)."""
        return cls._workspace(dev, int(nbytes))

    @classmethod
    def ws_bound(cls, rows, heads):
        """Upper bound of any plan's workspace for windows of <= rows query rows (B*Lq) and heads."""
        return int(1.25 * rows * heads * cls.WS_BYTES_PER_ROW_HEAD) + (1 << 20)

    @classmethod
    def _handle(cls, dev):
        cudnn = _cudnn_fe
        if cudnn is None:
            raise ImportError("attn_impl='cudnn_fe' needs the nvidia-cudnn-frontend package "
                              "(uv pip install nvidia-cudnn-frontend); use attn_impl='cudnn' or 'flash'")
        h = cls._handles.get(dev)
        if h is None:
            with torch.cuda.device(dev):
                h = cls._handles[dev] = cudnn.create_handle()
        cudnn.set_stream(handle=h, stream=torch.cuda.current_stream(dev).cuda_stream)
        return h

    @classmethod
    def _workspace(cls, dev, n):
        ws = cls._ws.get(dev)
        if ws is None or ws.numel() < n:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    f"cudnn_fe workspace would grow to {n} bytes INSIDE a CUDA graph capture (have "
                    f"{0 if ws is None else ws.numel()}): reserve it first (CudnnFEKernel.reserve / the engine's "
                    f"_alloc bound)")
            if ws is not None:
                cls._ws_old.setdefault(dev, []).append(ws)       # keep alive: older graphs use this address
            ws = cls._ws[dev] = torch.empty(max(n, 1 << 20), device=dev, dtype=torch.uint8)
        return ws

    @staticmethod
    def _sig(*ts):
        return tuple((tuple(t.shape), tuple(t.stride())) for t in ts)

    @classmethod
    def _fwd_plan(cls, q, k, v, o, stats, scale, handle, window=None):
        cudnn = _cudnn_fe
        key = ("f", q.device.index, scale, window) + cls._sig(q, k, v, o)
        p = cls._plans.get(key)
        if p is None:
            g = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16 if q.dtype == torch.bfloat16 else cudnn.data_type.HALF,
                              intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT,
                              handle=handle)
            tq, tk, tv = g.tensor_like(q), g.tensor_like(k), g.tensor_like(v)
            kw = {"sliding_window_length": int(window)} if window else {}   # W keys incl. the aligned diagonal
            to, ts = g.sdpa(name="sdpa", q=tq, k=tk, v=tv, is_inference=False, attn_scale=scale,
                            use_causal_mask_bottom_right=True, **kw)
            to.set_output(True).set_dim(list(o.shape)).set_stride(list(o.stride()))
            ts.set_output(True).set_data_type(cudnn.data_type.FLOAT).set_dim(list(stats.shape)).set_stride(list(stats.stride()))
            g.validate(); g.build_operation_graph()
            g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK]); g.check_support(); g.build_plans()
            p = cls._plans[key] = (g, (tq, tk, tv, to, ts), g.get_workspace_size())
        return p

    @classmethod
    def _bwd_plan(cls, q, k, v, o, do, stats, dq, dk, dv, scale, handle, window=None):
        cudnn = _cudnn_fe
        key = ("b", q.device.index, scale, window) + cls._sig(q, k, v, o, do, stats, dq, dk, dv)
        p = cls._plans.get(key)
        if p is None:
            g = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16 if q.dtype == torch.bfloat16 else cudnn.data_type.HALF,
                              intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT,
                              handle=handle)
            tq, tk, tv, to, tdo = (g.tensor_like(x) for x in (q, k, v, o, do))
            tst = g.tensor_like(stats)
            kw = {"sliding_window_length": int(window)} if window else {}
            tdq, tdk, tdv = g.sdpa_backward(name="sdpa_bwd", q=tq, k=tk, v=tv, o=to, dO=tdo, stats=tst,
                                            attn_scale=scale, use_causal_mask_bottom_right=True, **kw)
            for t, x in ((tdq, dq), (tdk, dk), (tdv, dv)):
                t.set_output(True).set_dim(list(x.shape)).set_stride(list(x.stride()))
            g.validate(); g.build_operation_graph()
            g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK]); g.check_support(); g.build_plans()
            p = cls._plans[key] = (g, (tq, tk, tv, to, tdo, tst, tdq, tdk, tdv), g.get_workspace_size())
        return p

    @classmethod
    def fwd(cls, q, k, v, scale, out=None, window=None):
        dev = q.device
        handle = cls._handle(dev)
        B, h, Lq, dh = q.shape
        o = torch.empty_like(q) if out is None else out
        stats = torch.empty(B, h, Lq, 1, device=dev, dtype=torch.float32)
        g, (tq, tk, tv, to, ts), wsz = cls._fwd_plan(q, k, v, o, stats, scale, handle, window)
        g.execute({tq: q, tk: k, tv: v, to: o, ts: stats}, cls._workspace(dev, wsz), handle=handle)
        return o, stats, None

    @classmethod
    def bwd(cls, dout, q, k, v, out, lse, extra, scale, dq_out=None, window=None, pad_rows=0):
        dev = q.device
        handle = cls._handle(dev)
        B, h, Lk, dh = k.shape
        dq = torch.empty_like(q) if dq_out is None else dq_out
        dkb_f = torch.empty(Lk + pad_rows, B, h, dh, device=dev, dtype=q.dtype)
        dvb_f = torch.empty(Lk + pad_rows, B, h, dh, device=dev, dtype=q.dtype)
        if pad_rows:                       # rows nobody attended (raw_diag: the slab's last row)
            dkb_f[Lk:].zero_(); dvb_f[Lk:].zero_()
        dkb, dvb = dkb_f[:Lk], dvb_f[:Lk]
        dk, dv = dkb.permute(1, 2, 0, 3), dvb.permute(1, 2, 0, 3)
        g, (tq, tk, tv, to, tdo, tst, tdq, tdk, tdv), wsz = cls._bwd_plan(q, k, v, out, dout, lse, dq, dk, dv, scale, handle, window)
        g.execute({tq: q, tk: k, tv: v, to: out, tdo: dout, tst: lse, tdq: dq, tdk: dk, tdv: dv},
                  cls._workspace(dev, wsz), handle=handle)
        return dq, [(0, dkb_f)], [(0, dvb_f)]


class MathKernel:
    """fp32 dense twin of CudnnFEKernel (validation only): bottom-right causal, optional
    sliding window (W keys incl. the aligned diagonal), same (o, lse, extra) / pieces API."""
    name = "mathk"
    out_ok = False

    @staticmethod
    def _mask(Lq, Lk, window, dev):
        i = torch.arange(Lq, device=dev)[:, None] + (Lk - Lq)      # bottom-right aligned key index
        j = torch.arange(Lk, device=dev)[None, :]
        m = j <= i
        if window:
            m &= j > i - int(window)
        return m

    @classmethod
    def fwd(cls, q, k, v, scale, out=None, window=None):
        s = (q.float() @ k.float().transpose(-1, -2)) * scale
        s = s.masked_fill(~cls._mask(q.shape[2], k.shape[2], window, q.device), float("-inf"))
        lse = torch.logsumexp(s, -1, keepdim=True)
        o = (torch.exp(s - lse) @ v.float()).to(q.dtype)
        if out is not None:
            out.copy_(o); o = out
        return o, lse, None

    @classmethod
    def bwd(cls, dout, q, k, v, out, lse, extra, scale, dq_out=None, window=None, pad_rows=0):
        # split-softmax backward of THIS partial given the (possibly merged) row (out, lse):
        # P = exp(s - lse), delta = rowsum(dO * O), dS = P (dO v^T - delta)  -- like cuDNN
        qf, kf, vf, do, of = q.float(), k.float(), v.float(), dout.float(), out.float()
        s = (qf @ kf.transpose(-1, -2)) * scale
        m = cls._mask(q.shape[2], k.shape[2], window, q.device)
        P = torch.exp(s - lse.reshape(*lse.shape[:3], 1)).masked_fill(~m, 0.0)
        delta = (do * of).sum(-1, keepdim=True)
        dS = P * (do @ vf.transpose(-1, -2) - delta)
        dq = (dS @ kf) * scale
        dk = (dS.transpose(-1, -2) @ qf) * scale
        dv = P.transpose(-1, -2) @ do
        dq = dq.to(q.dtype)
        if dq_out is not None:
            dq_out.copy_(dq); dq = dq_out
        dk_tm, dv_tm = dk.permute(2, 0, 1, 3).to(q.dtype), dv.permute(2, 0, 1, 3).to(q.dtype)
        if pad_rows:
            z = torch.zeros(pad_rows, *dk_tm.shape[1:], device=q.device, dtype=q.dtype)
            dk_tm, dv_tm = torch.cat([dk_tm, z]), torch.cat([dv_tm, z])
        return dq, [(0, dk_tm)], [(0, dv_tm)]


KERNELS = {"flash": FlashKernel, "cudnn": CudnnKernel, "cudnn_fe": CudnnFEKernel, "mathk": MathKernel}
