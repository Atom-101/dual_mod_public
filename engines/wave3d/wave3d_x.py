"""Wave3D-X: the copy-free, exact-window, explicit-bmm 3D wavefront engine.

Same operator and schedule as engines/wave3d/wave3d.py (Wave3D, the vmap engine,
kept for A/B via make_engine(engine="vmap")): layer l is at intra-layer step
t_l = t - l*K; per global step every active layer preps its new chunk, attends
its window over [committed prefix ++ window], tails its oldest chunk and refines
its window. Differences are ALL plumbing:

  * TIME-MAJOR write-once caches xh/kf/vf/qr [L, T, B, ...]: prep writes chunk
    rows ONCE (_CacheWrite); every window is a zero-copy strided ALIAS of the
    cache (_CacheRead: Tensor.set_ on the storage => own version counter, so
    later chunk writes never trip autograd's saved-tensor check). Grads of the
    reads accumulate (fp32) into grad caches; the writer's backward hands them
    to prep. A per-layer TOKEN chain (write -> token -> reads) gives autograd
    the write-before-read dependency across steps/segments.
  * The window of the G active layers with equal valid length is ONE strided
    view [G, Wv*B, d] (layer stride = T*B*d - K*C*B*d) -> torch.bmm over the
    stacked bf16 weights (wave3d_ops). No stack/shift/permute copies.
  * EXACT windows: partial windows at a layer's first/last K-1 steps are
    processed at their true length (<=2 extra 1-layer groups per step), not
    zero-padded to W -- removes the +23%/+48% refine FLOPs at K16/K32
    (rows/step 16594 -> 13443 at K16).
  * K/V cache in place at absolute positions via the validated _AttendCache
    (slab = previous refined window ++ raw new chunk, restored in backward);
    flash's output inherits the time-major query strides, so o feeds the
    refine GEMMs with no permute copy.
  * Bodies (prep / per-group refine / tail, or one fused refine+tail region)
    torch.compile'd with STATIC shapes (set_compile): ~2K+A_max graphs per K
    (29/39/68 at K16/32/64; the fused region has 139+ patterns -> per-group is
    the default). dynamic=True kernels measured 2.7x slower. The compiled
    forward body is at its GPU floor (K16: 2.35 ms/step = flash 1.07 + GEMM
    0.50 + fused pointwise 0.76, vs 4.1 ms for the vmap engine). The refine's
    12 LoRA GEMMs are 6 launches (wave3d_ops.combine_params: N-concat of the
    same-input downs, one bmm for the five ups) -- launch count, not FLOPs,
    was the limiter (~570 -> ~400 aten launches per step).
  * Launch-count plumbing: ONE _CacheWrite / _CacheRead / _AttendGroup node per
    step (or per window group) instead of per layer; strided single-kernel
    chunk writes; weight grads accumulate through _PSlice straight into fp32
    grad stacks (no SliceBackward zero-fills / per-segment stack+cast+
    AccumulateGrad), handed to the params' .grad by _GradSink at the end.
  * Backward: time-segment checkpoint via our own reentrant _SegCheckpoint
    (torch.utils.checkpoint is dynamo-disabled inside => compiled bodies ran
    eager there; its non-reentrant form also costs ~2.7 ms/step of pack hooks).
    The forward runs under no_grad; the recompute skips the flash forward via a
    per-forward STASH of (o, lse) keyed by (step, layer) (stash_attn; ~35 GB at
    K16, ~73 GB at K32). sac='flash'|'gemm'|'lean'|'store' with store_frac f
    are the selective-checkpoint variants that were measured and rejected here
    (policy-SAC's dispatch mode doubles eager forward CPU; 'gemm' stores
    ~440 MB/step, 'store' ~1 GB/step -> neither fits at f=1; see the report).
  * Attention kernels live in engines/wave3d/wave3d_attn.py (KERNELS), selected by
    attn_impl: 'cudnn_fe' (DEFAULT: cuDNN's Blackwell-native SDPA via the cuDNN
    frontend with the bottom-right causal mask, ONE call fwd / ONE bwd; attention
    GPU time at K16 872+2427 -> 346+1028 us/step), 'cudnn' (same kernels through
    aten as prefix-dense + window-causal + Triton LSE merge; backward = both
    halves' cuDNN backward on the MERGED (o, lse) = exact split-softmax
    gradient -- the old autograd-through-the-merge twin was WRONG because aten's
    cuDNN backward drops the lse gradient -> biased dq/dk, the "noisier grads"),
    'flash' (aten FA2-generation kernels), 'math' (fp32 validation path). All
    validated (validate_wave3d.py --engine x --attn ...) and CUDA-graph
    capturable. NOTE the step is launch/CPU-bound (fwd wall 2.6 ms vs GPU 1.6
    ms/step at K16), so the kernel win shows in wall only with fewer launches
    (CUDA-graph mode / one varlen attention call per step).

  * CUDA-GRAPH MODE (graphs=True; the production training path): the whole
    pipeline runs as graph replays through ONE autograd Function (_GraphedPipe):
    prologue graph (zero grad buffers, restack the bf16 weights from the live
    params) + per-segment forward graphs, then per-segment recompute+backward
    graphs in reverse time; tok_emb and the head stay eager autograd. Captured per
    K on first use into ONE shared mempool (capture order == replay order; every
    cross-graph tensor -- boundary states, attention stash, boundary grads -- is
    alive exactly as long as in replay; refs dropped afterwards so another K's
    capture may alias the memory: one K per training step). The SAME closures run
    eagerly first (warm-up: every compiled body exists in exactly the capture's
    global state), then under torch.compiler.set_stance("fail_on_recompile").
    The recompute+backward runs on the MAIN thread here (torch.autograd.backward
    inside the capture; the engine thread launches on the capture stream via the
    nodes' recorded streams). Measured K16: 3.51 -> 2.06 s/step, wall == GPU.
    Capture cost ~30-50 s per K (one-time); eager no_grad forwards (validation)
    are unaffected and use the fused prep/tail bodies. GOTCHAS: (1) an eager
    forward between a graphed forward and its backward corrupts the shared caches
    the bwd graphs read; (2) never return a STASH tensor object from a Function
    (autograd sets grad_fn on it -> the stash pins whole recompute graphs);
    (3) stash copies of grad-requiring tensors must run under no_grad (an in-place
    copy_ makes the buffer autograd-tracked: CopySlices chains, cross-stream
    capture errors); (4) memory: graph replays bypass the allocator -> report
    memory_reserved / the capture-time peak (bench does), not max_allocated;
    (5) the eager no_grad (val) forward does NOT fill the attention stash (nothing
    recomputes it; it held ~110 GB after a K64 val) and _alloc pre-reserves the
    cuDNN-FE workspace so it never grows after a capture (baked addresses).
  * FUSED REFINE (fused_refine=True, default): the per-group refine chain runs as
    engines/wave3d/wave3d_fused.py (cuBLAS bmm's + hand-written Triton pointwise
    kernels, one per stage in fwd and bwd) instead of torch.compile(X.refine):
    K16 graph-mode pointwise 1.66 -> 0.88 ms/step (see that module's header).
  * DEFERRED WEIGHT GRADS (defer_dw=True) for the 7 big weights (wq/wk/wv, wo,
    w1/w3, w2): prep/tail are split at their GEMMs (compiled pointwise pieces
    prep_a/prep_b/tail_a/tail_b + _DLin bmm's); _DLin's backward returns dX only
    and stashes dY (X stashed at the forward) per (layer, step-in-segment); after
    the segment's backward ONE fp32-output GEMM per (weight, layer) over the
    segment's rows (aten.mm.dtype) lands in the grad stack -- replaces ~350
    skinny per-step dW GEMMs + autograd's bf16 accumulation of them (-0.45 ms/step
    at K16). Flushed in _SegCheckpoint.backward / the graph bseg, and (torch
    checkpoint paths) at the next segment's recompute start + _GradSink.
  * Grad caches gxh/gkf/gvf/gqr in the CACHE dtype (bf16): the reference's own
    accumulation dtype for its k/v/q slices; vectorized same-dtype adds.
  * Plumbing: per-layer state views via unbind (no select_backward zero-fills),
    _CacheWrite.backward hands out grad-cache views (rows final, zeroed at the
    next pipeline start), the attention group writes o / dq straight into the
    stacked buffers (cudnn_fe) and skips the slab writes on a stash hit (the
    backward rewrites them).

Residual stream x stays fp32 (as in the module code); xhat is cached in the
activation dtype (every consumer casts it there anyway); 1-D gains fp32.
"""

import functools

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import (CheckpointPolicy, checkpoint,
                                    create_selective_checkpoint_contexts)

from engines.wave3d import wave3d_ops as X
from engines.wave3d.wave3d import C_DEFAULT, _AttendCache, _math_lr
from engines.wave3d.wave3d_attn import KERNELS as _KERNELS, accum_kv as _accum_kv_pieces, self_merge_fused as _self_merge_fused, self_bwd_fused as _self_bwd_fused

aten = torch.ops.aten


def _raise_dynamo_limits():
    """Static-shape compile => one graph per (shape, grad mode); the defaults
    (recompile_limit 8, accumulated 256) silently fall back to EAGER once hit
    (measured: the whole recompute ran eager at K16 -> bwd 5.5 s vs 3.4 s).
    torch._dynamo.config user overrides are THREAD-LOCAL: the recompute runs on
    the autograd engine thread, so this must also be called from inside
    _SegCheckpoint.backward (cheap), not just at import."""
    import torch._dynamo
    c = torch._dynamo.config
    for attr, val in (("recompile_limit", 4096), ("cache_size_limit", 4096),
                      ("accumulated_recompile_limit", 65536),
                      ("accumulated_cache_size_limit", 65536)):
        if hasattr(c, attr):
            setattr(c, attr, val)


_raise_dynamo_limits()


# ----------------------------------------------------------------------------- SAC
# identity marker so a policy can MUST_SAVE the refined K/V (pointwise outputs)
@torch.library.custom_op("wave3d::keep", mutates_args=())
def _keep(x: torch.Tensor) -> torch.Tensor:
    return x.clone()


@_keep.register_fake
def _keep_fake(x):
    return torch.empty_like(x)


def _keep_bwd(ctx, g):
    return g


_keep.register_autograd(_keep_bwd)

# NOTE: never add the .out overloads (inductor's extern GEMM calls): SAC's recompute
# would hand back the stored result while the generated code reads its (unfilled)
# out= buffer -> silent garbage. Policy-SAC over GEMMs is for EAGER bodies only.
_FLASH_OPS = {aten._scaled_dot_product_flash_attention.default}
_GEMM_OPS = {aten.mm.default, aten.bmm.default, aten.addmm.default, aten.baddbmm.default} | _FLASH_OPS
_LEAN_OPS = _FLASH_OPS | {torch.ops.wave3d.keep.default}


def _mk_policy(save_ops, lean):
    def policy(ctx, op, *args, **kwargs):
        if op in save_ops:
            return CheckpointPolicy.MUST_SAVE
        if lean and op is aten.bmm.default and args[1].shape[-1] <= 512:
            return CheckpointPolicy.MUST_SAVE           # LoRA down-projections (144-wide)
        return CheckpointPolicy.PREFER_RECOMPUTE
    return policy


# ----------------------------------------------------------------------------- caches
def _alias(buf, offset, size, stride):
    """Strided alias of `buf`'s storage with its OWN version counter."""
    t = torch.empty(0, dtype=buf.dtype, device=buf.device)
    t.set_(buf.untyped_storage(), offset, size, stride)
    return t


class _CacheWrite(torch.autograd.Function):
    """prep -> caches for ALL Gp prepped layers of a step in one node. Writes chunk
    tl_g of layer l_g (xh/kf/vf [Gp, C*B, d], qr [Gp, C, B, h, dh]) into the
    write-once caches; returns the layers' next tokens. backward: returns the
    grads that all later reads accumulated into the grad caches (then zeroes
    those rows)."""

    @staticmethod
    def _views(eng, lcs, bufs):
        """[Gp, C, B, ...] strided views of the chunk rows (layer l_g, chunk c_g):
        c_g = t - l_g*K is affine in g, so ONE as_strided view covers all layers."""
        C, B, d, T = eng.C, eng.B, eng.d, eng.T
        Gp = len(lcs)
        (l0, c0), (l1, c1) = lcs[0], lcs[-1]
        step = (l1 - l0) * T * B * d + (c1 - c0) * C * B * d if Gp > 1 else 0
        step //= max(Gp - 1, 1)
        off = l0 * T * B * d + c0 * C * B * d
        outs = []
        for buf in bufs:
            if buf.dim() == 4:                              # [L, T, B, d]
                outs.append(buf.as_strided((Gp, C, B, d), (step, B * d, d, 1), off))
            else:                                           # [L, T, B, h, dh]
                h, dh = buf.shape[3], buf.shape[4]
                outs.append(buf.as_strided((Gp, C, B, h, dh), (step, B * d, d, dh, 1), off))
        return outs

    @staticmethod
    def forward(ctx, xh, kf, vf, qr, eng, lcs, *toks):
        C, B, d = eng.C, eng.B, eng.d
        Gp = len(lcs)
        vx, vk, vv, vq = _CacheWrite._views(eng, lcs, (eng.xh, eng.kf, eng.vf, eng.qr))
        vx.copy_(xh.view(Gp, C, B, d))
        vk.copy_(kf.view(Gp, C, B, d))
        vv.copy_(vf.view(Gp, C, B, d))
        vq.copy_(qr)
        ctx.meta = (eng, lcs)
        return tuple(torch.empty((), device=xh.device) for _ in lcs)

    @staticmethod
    def backward(ctx, *dtoks):
        eng, lcs = ctx.meta
        C, B, d = eng.C, eng.B, eng.d
        Gp = len(lcs)
        # The grad-cache rows of these chunks are FINAL here (every later read's
        # backward has run: token chain) and nothing touches them again before the
        # next pipeline call zeroes the caches -> hand out the views (no clone/zero).
        gx, gk, gv, gq = _CacheWrite._views(eng, lcs, (eng.gxh, eng.gkf, eng.gvf, eng.gqr))
        return (gx.view(Gp, C * B, d), gk.view(Gp, C * B, d), gv.view(Gp, C * B, d), gq,
                None, None) + (None,) * Gp


class _CacheRead(torch.autograd.Function):
    """Zero-copy windows of G consecutive layers. Layer l0+g's window is rows
    [r0 - g*K*C, +Wv) of its cache (affine in g by the schedule). Returns
    xh/kf/vf [G, Wv*B, d] and qr [G, Wv, B, h, dh] aliases. backward: accumulate
    into the fp32 grad caches (same strided view)."""

    @staticmethod
    def forward(ctx, eng, groups, K, *toks):
        """groups: list of (l0, G, Wv, r0). Returns, per group, (xh_w, kf_w, vf_w, qr_w)."""
        B, d, h, dh, T = eng.B, eng.d, eng.h, eng.dh, eng.T
        sL = T * B * d
        bs = sL - K * eng.C * B * d                           # layer-to-layer stride
        specs, outs = [], []
        for l0, G, Wv, r0 in groups:
            off = l0 * sL + r0 * B * d
            flat = ((G, Wv * B, d), (bs, d, 1))
            five = ((G, Wv, B, h, dh), (bs, B * d, d, dh, 1))
            specs.append((off, flat, five))
            outs += [_alias(eng.xh, off, *flat), _alias(eng.kf, off, *flat),
                     _alias(eng.vf, off, *flat), _alias(eng.qr, off, *five)]
        ctx.meta = (eng, specs)
        return tuple(outs)

    @staticmethod
    def backward(ctx, *grads):
        eng, specs = ctx.meta
        for gi, (off, flat, five) in enumerate(specs):
            gx, gk, gv, gq = grads[4 * gi:4 * gi + 4]
            for gbuf, g, spec in ((eng.gxh, gx, flat), (eng.gkf, gk, flat),
                                  (eng.gvf, gv, flat), (eng.gqr, gq, five)):
                if g is not None:
                    gbuf.as_strided(spec[0], spec[1], off).add_(g)
        return (None,) * len(ctx.needs_input_grad)


def _accum_kv(dk, dv, dKbuf, dVbuf, row0, r1, commit):
    """Reverse-time prefix/commit bookkeeping shared by the flash and math paths
    (see _AttendCache docstring)."""
    if row0 > 0:
        dKbuf[:row0] += dk[:row0]
        dVbuf[:row0] += dv[:row0]
    dsK, dsV = dk[row0:r1], dv[row0:r1]
    if commit > 0:
        dsK[:commit] += dKbuf[row0:row0 + commit]
        dsV[:commit] += dVbuf[row0:row0 + commit]
    return dsK, dsV


def _lse_merge(oA, lseA, oB, lseB):
    """Online-softmax merge of two attention partials (dense prefix A, causal
    window B), fp32; == chunked_split._attend_split's merge."""
    lseA = lseA.reshape(oA.shape[:3])
    lseB = lseB.reshape(oB.shape[:3])
    m = torch.maximum(lseA, lseB)
    eA = (lseA - m).exp()
    eB = (lseB - m).exp()
    den = (eA + eB).unsqueeze(-1)
    return ((oA.float() * eA.unsqueeze(-1) + oB.float() * eB.unsqueeze(-1)) / den).to(oB.dtype)


_lse_merge_c = torch.compile(_lse_merge, dynamic=True)


class _AttendCacheK(torch.autograd.Function):
    """_AttendCache twin over a wave3d_attn kernel (flash | cudnn): per-layer node
    for the non-grouped path (stash_attn=False). Same in-place cache protocol
    (restore slab in backward, reverse-time prefix/commit accumulation)."""

    @staticmethod
    def forward(ctx, q, slabK, slabV, Kbuf, Vbuf, dKbuf, dVbuf, row0, r1, scale, commit, kern):
        Kbuf[row0:r1].copy_(slabK)
        Vbuf[row0:r1].copy_(slabV)
        out, lse, extra = kern.fwd(q, Kbuf[:r1].permute(1, 2, 0, 3), Vbuf[:r1].permute(1, 2, 0, 3), scale)
        ctx.save_for_backward(q, slabK, slabV, out, lse)
        ctx.bufs = (Kbuf, Vbuf, dKbuf, dVbuf)
        ctx.meta = (row0, r1, scale, commit, kern, extra)
        return out

    @staticmethod
    def backward(ctx, dout):
        q, slabK, slabV, out, lse = ctx.saved_tensors
        Kbuf, Vbuf, dKbuf, dVbuf = ctx.bufs
        row0, r1, scale, commit, kern, extra = ctx.meta
        Kbuf[row0:r1].copy_(slabK)
        Vbuf[row0:r1].copy_(slabV)
        dq, dkp, dvp = kern.bwd(dout, q, Kbuf[:r1].permute(1, 2, 0, 3), Vbuf[:r1].permute(1, 2, 0, 3),
                                out, lse, extra, scale)
        dsK = _accum_kv_pieces(dkp, dKbuf, row0, r1, commit)
        dsV = _accum_kv_pieces(dvp, dVbuf, row0, r1, commit)
        return (dq, dsK, dsV) + (None,) * 9


class _AttendCacheStash(_AttendCache):
    """_AttendCache whose forward memoizes the flash outputs per (step, layer) in
    eng._stash for the duration of one pipeline forward: the checkpoint RECOMPUTE
    (same inputs, deterministic) then skips the flash forward (~1.1 ms/step at
    K16) and only restores the slab. The backward is _AttendCache's, unchanged
    (it needs q, the slab, out, lse -- all still saved). Memory: o + lse per
    (step, active layer) ~ 33 GB at K16 / 73 GB at K32 (B4/T4096)."""

    @staticmethod
    def forward(ctx, q, slabK, slabV, Kbuf, Vbuf, dKbuf, dVbuf, row0, r1, scale, commit, stash, key,
                kr=None, vr=None):
        """slabK/V: previous refined window rows (chunks c0..tl-1); kr/vr: the raw
        new chunk (written right after them) -- passed separately so no cat copy."""
        n0 = slabK.shape[0]
        Kbuf[row0:row0 + n0].copy_(slabK)
        Vbuf[row0:row0 + n0].copy_(slabV)
        if kr is not None:
            Kbuf[row0 + n0:r1].copy_(kr)
            Vbuf[row0 + n0:r1].copy_(vr)
        hit = stash.get(key)
        if hit is None:
            k = Kbuf[:r1].permute(1, 2, 0, 3)
            v = Vbuf[:r1].permute(1, 2, 0, 3)
            out, lse, cq, ck, mq, mk, seed, off, _ = \
                torch.ops.aten._scaled_dot_product_flash_attention(
                    q, k, v, 0.0, True, False, scale=scale)
            hit = stash[key] = (out, lse, cq, ck, mq, mk, seed, off)
        out, lse, cq, ck, mq, mk, seed, off = hit
        if kr is None:
            ctx.save_for_backward(q, slabK, slabV, out, lse, cq, ck, seed, off)
        else:
            ctx.save_for_backward(q, slabK, slabV, out, lse, cq, ck, seed, off, kr, vr)
        ctx.bufs = (Kbuf, Vbuf, dKbuf, dVbuf)
        ctx.meta = (row0, r1, scale, commit, mq, mk)
        ctx.n0 = n0
        return out

    @staticmethod
    def backward(ctx, dout):
        saved = ctx.saved_tensors
        q, slabK, slabV, out, lse, cq, ck, seed, off = saved[:9]
        Kbuf, Vbuf, dKbuf, dVbuf = ctx.bufs
        row0, r1, scale, commit, mq, mk = ctx.meta
        n0 = ctx.n0
        Kbuf[row0:row0 + n0].copy_(slabK)
        Vbuf[row0:row0 + n0].copy_(slabV)
        if len(saved) > 9:
            Kbuf[row0 + n0:r1].copy_(saved[9])
            Vbuf[row0 + n0:r1].copy_(saved[10])
        k = Kbuf[:r1].permute(1, 2, 0, 3)
        v = Vbuf[:r1].permute(1, 2, 0, 3)
        dq, dk, dv = torch.ops.aten._scaled_dot_product_flash_attention_backward(
            dout.contiguous(), q, k, v, out, lse, cq, ck, mq, mk, 0.0, True, seed, off,
            scale=scale)
        dk = dk.permute(2, 0, 1, 3)
        dv = dv.permute(2, 0, 1, 3)
        dsK, dsV = _accum_kv(dk, dv, dKbuf, dVbuf, row0, r1, commit)
        dkr = dvr = None
        if len(saved) > 9:
            dsK, dkr = dsK[:n0], dsK[n0:]
            dsV, dvr = dsV[:n0], dsV[n0:]
        return (dq, dsK, dsV) + (None,) * 10 + (dkr, dvr)


class _AttendGroup(torch.autograd.Function):
    """One node per WINDOW GROUP: the G layers' flash attentions (each over its own
    in-place cache, slab = prev refined window ++ raw new chunk written here, no
    cat), the flash-output stash, and the layout work (q permute, o -> time-major
    merged [G, Wv*B, d], the tail rows [Gt, C*B, d]) all inside one autograd
    Function -- removes ~8 autograd nodes per layer per step from the recompute
    graph (the backward was ~30% CPU-bound on node count).

    inputs : qr_w [G, Wv, B, h, dh] (cache alias), kr_all [Gp, C, B, h, dh] / vf_all
             [Gp, C*B, d] (this step's prep outputs, indexed by prep slot), then the
             previous refined windows Kw_p/Vw_p of the layers that have one.
    meta   : per layer (l, row0, r1, commit, key, gp (prep slot or -1), prev (bool), tail (bool))"""

    @staticmethod
    def forward(ctx, qr_w, kr_all, vf_all, eng, meta, kf_w, vf_w, cs_w, sn_w, *kv_prev):
        """kf_w/vf_w [G, Wv*B, d] (window RAW k flat / v flat, cache aliases) + cs_w/sn_w
        [G, Wv, 1, 1, dh/2]: only with eng.raw_diag (the raw-self partial); else None."""
        G, Wv, B, h, dh = qr_w.shape
        C, d, scale, stash = eng.C, eng.d, eng.scale, eng._stash
        kern = _KERNELS[eng._impl]
        rd, sub, win = eng.raw_diag, (1 if eng.raw_diag else 0), eng._win_main
        gkey = (meta[0][4], "og")
        o_g = stash.get(gkey)                             # the group's stacked o (recompute hit)
        n0s = [kv_prev[2 * i].shape[0] if m[6] else 0
               for i, m in zip([sum(mm[6] for mm in meta[:g]) for g in range(G)], meta)]
        if o_g is None:
            # forward: write slab = prev refined window ++ raw new chunk, run the kernel.
            # (On a stash hit -- the checkpoint recompute -- nothing reads Kbuf/Vbuf: the
            # backward rewrites the slab itself, so the writes are skipped there.)
            direct = getattr(kern, "out_ok", False)      # kernel writes o straight into o_g
            o_g = torch.empty(G, Wv * B, d, device=qr_w.device, dtype=qr_w.dtype) if direct else None
            os, ip = [], 0
            for g, (l, row0, r1, commit, key, gp, prev, tail) in enumerate(meta):
                Kbuf, Vbuf = eng.Kbuf[l], eng.Vbuf[l]
                n0 = n0s[g]
                if prev:
                    Kbuf[row0:row0 + n0].copy_(kv_prev[ip])
                    Vbuf[row0:row0 + n0].copy_(kv_prev[ip + 1])
                    ip += 2
                if gp >= 0:
                    Kbuf[row0 + n0:r1].copy_(kr_all[gp])
                    Vbuf[row0 + n0:r1].copy_(vf_all[gp].view(C, B, h, dh))
                q = qr_w[g].permute(1, 2, 0, 3)
                out = o_g[g].view(Wv, B, h, dh).permute(1, 2, 0, 3) if direct else None
                kslab, vslab = Kbuf[:r1 - sub].permute(1, 2, 0, 3), Vbuf[:r1 - sub].permute(1, 2, 0, 3)
                wkw = {"window": win} if win else {}
                if rd and r1 - sub < Wv:
                    # row0 == 0: position 0 has NO past key => cuDNN's bottom-right mask needs
                    # s_q <= s_kv, so run the kernel on rows 1.. and give row 0 an empty partial
                    # (o = 0, lse = -inf); the raw-self merge then yields o[0] = v_self[0].
                    o_full = out if direct else torch.empty_like(q)
                    h1 = kern.fwd(q[:, :, 1:], kslab, vslab, scale, out=o_full[:, :, 1:], **wkw)
                    o_full[:, :, :1].zero_()
                    lse_full = torch.full((B, h, Wv, 1), float("-inf"), device=q.device, dtype=torch.float32)
                    lse_full[:, :, 1:] = h1[1]
                    hit = (o_full, lse_full, h1[2])
                else:
                    hit = kern.fwd(q, kslab, vslab, scale, out=out, **wkw)  # (out, lse, extra)
                if rd:                                    # merge the query's own RAW (k, v) as a one-key partial
                    kfv = kf_w[g].view(Wv, B, h, dh).permute(1, 2, 0, 3)     # raw UNROTATED key (rope in-kernel)
                    vfv = vf_w[g].view(Wv, B, h, dh).permute(1, 2, 0, 3)
                    s_self = _self_merge_fused(hit[0], hit[1], q, kfv, vfv, cs_w[g], sn_w[g], scale)  # in place
                    hit = (hit[0], hit[1], hit[2], s_self)
                if not eng._val_fwd:
                    stash[key] = hit                  # memo for the checkpoint RECOMPUTE only: the eager
                if not direct:                        #   no_grad (val) forward has none (~110 GB at K64)
                    os.append(hit[0].permute(2, 0, 1, 3).reshape(Wv * B, d))   # time-major merged (view)
            if not direct:
                o_g = torch.stack(os) if G > 1 else os[0].unsqueeze(0)
            if not eng._val_fwd:
                stash[gkey] = o_g
        tails = [o_g[g][:C * B] for g, m in enumerate(meta) if m[7]]
        o_tail = torch.stack(tails) if tails else qr_w.new_empty(0)
        ctx.save_for_backward(qr_w, kr_all, vf_all, *([kf_w, vf_w] if rd else []), *kv_prev)
        ctx.eng, ctx.meta, ctx.n0s, ctx.rd, ctx.trig = eng, meta, n0s, rd, (cs_w, sn_w)
        # view_as: NEVER hand the stash's own tensor object out of a Function -- autograd
        # would set its grad_fn on it and the stash (alive until the next forward) would
        # pin every segment's recompute graph + saved activations (measured +16 GB at K16)
        return o_g.view_as(o_g), o_tail

    @staticmethod
    def backward(ctx, do_g, do_tail):
        saved = ctx.saved_tensors
        rd = ctx.rd
        if rd:
            qr_w, kr_all, vf_all, kf_w, vf_w, kv_prev = saved[0], saved[1], saved[2], saved[3], saved[4], saved[5:]
        else:
            qr_w, kr_all, vf_all, kv_prev = saved[0], saved[1], saved[2], saved[3:]
            kf_w = vf_w = None
        cs_w, sn_w = ctx.trig
        eng, meta, n0s = ctx.eng, ctx.meta, ctx.n0s
        G, Wv, B, h, dh = qr_w.shape
        C, d, scale, stash = eng.C, eng.d, eng.scale, eng._stash
        kern = _KERNELS[eng._impl]
        direct = getattr(kern, "out_ok", False)
        sub, win = (1 if rd else 0), eng._win_main
        dkf_w = torch.empty_like(kf_w) if rd else None
        dvf_w = torch.empty_like(vf_w) if rd else None
        dq_all = torch.empty_like(qr_w)
        dkr = torch.empty_like(kr_all) if kr_all is not None else None
        dvf = torch.empty_like(vf_all) if vf_all is not None else None
        filled = set()                                    # prep slots this group's layers cover
        dkv = []
        ip = 0
        it = 0
        for g, (l, row0, r1, commit, key, gp, prev, tail) in enumerate(meta):
            Kbuf, Vbuf, dKbuf, dVbuf = eng.Kbuf[l], eng.Vbuf[l], eng.dKbuf[l], eng.dVbuf[l]
            n0 = n0s[g]
            if prev:
                Kbuf[row0:row0 + n0].copy_(kv_prev[ip])
                Vbuf[row0:row0 + n0].copy_(kv_prev[ip + 1])
            if gp >= 0:
                Kbuf[row0 + n0:r1].copy_(kr_all[gp])
                Vbuf[row0 + n0:r1].copy_(vf_all[gp].view(C, B, h, dh))
            out, lse, extra = stash[key][:3]
            do = do_g[g]
            if tail and do_tail is not None:
                do = do.clone()
                do[:C * B] += do_tail[it]
                it += 1
            dout = do.view(Wv, B, h, dh).permute(1, 2, 0, 3)
            q = qr_w[g].permute(1, 2, 0, 3)
            kslab, vslab = Kbuf[:r1 - sub].permute(1, 2, 0, 3), Vbuf[:r1 - sub].permute(1, 2, 0, 3)
            wkw = {"window": win} if win else {}
            dq_main = dq_all[g].view(Wv, B, h, dh).permute(1, 2, 0, 3) if direct else None
            # raw_diag: the slab's last row got no key/value grad from the window part (nobody in this
            # window attends it yet) -> the kernel pads ONE zero row so the pieces still cover [0, r1)
            if rd and r1 - sub < Wv:                      # row0 == 0 (see forward): row 0 has no window part
                dq1, dkp, dvp = kern.bwd(dout[:, :, 1:], q[:, :, 1:], kslab, vslab, out[:, :, 1:],
                                         lse[:, :, 1:].contiguous(), extra, scale,   # cuDNN<9.26: packed stats
                                         dq_out=dq_main[:, :, 1:] if direct else None, pad_rows=sub, **wkw)
                if direct:
                    dq_main[:, :, :1].zero_()
                else:
                    dq = torch.zeros_like(q)
                    dq[:, :, 1:] = dq1
            else:
                dq, dkp, dvp = kern.bwd(dout, q, kslab, vslab, out, lse, extra, scale, dq_out=dq_main,
                                        pad_rows=sub, **wkw)
            if rd:
                kfv = kf_w[g].view(Wv, B, h, dh).permute(1, 2, 0, 3)
                vfv = vf_w[g].view(Wv, B, h, dh).permute(1, 2, 0, 3)
                _self_bwd_fused(dout, q, kfv, vfv, cs_w[g], sn_w[g], out, lse, stash[key][3], scale,
                                dq_main if direct else dq,                        # dq += (in place)
                                dkf_w[g].view(Wv, B, h, dh).permute(1, 2, 0, 3),  # written (raw space)
                                dvf_w[g].view(Wv, B, h, dh).permute(1, 2, 0, 3))
            if not direct:
                dq_all[g].copy_(dq.permute(2, 0, 1, 3))
            dsK = _accum_kv_pieces(dkp, dKbuf, row0, r1, commit)
            dsV = _accum_kv_pieces(dvp, dVbuf, row0, r1, commit)
            if prev:
                dkv += [dsK[:n0], dsV[:n0]]
                ip += 2
            if gp >= 0:
                dkr[gp].copy_(dsK[n0:])
                dvf[gp].copy_(dsV[n0:].reshape(C * B, d))
                filled.add(gp)
        if dkr is not None:
            for gp in range(kr_all.shape[0]):
                if gp not in filled:                      # prepped layer attending in another group
                    dkr[gp].zero_()
                    dvf[gp].zero_()
        return (dq_all, dkr, dvf, None, None, dkf_w, dvf_w, None, None) + tuple(dkv)


class _AttendCacheMath(torch.autograd.Function):
    """fp32 twin of _AttendCache (dense math attention; validation only)."""

    @staticmethod
    def forward(ctx, q, slabK, slabV, Kbuf, Vbuf, dKbuf, dVbuf, row0, r1, scale, commit):
        Kbuf[row0:r1].copy_(slabK)
        Vbuf[row0:r1].copy_(slabV)
        out = _math_lr(q, Kbuf[:r1].permute(1, 2, 0, 3), Vbuf[:r1].permute(1, 2, 0, 3), scale)
        ctx.save_for_backward(q, slabK, slabV)
        ctx.bufs = (Kbuf, Vbuf, dKbuf, dVbuf)
        ctx.meta = (row0, r1, scale, commit)
        return out

    @staticmethod
    def backward(ctx, dout):
        q, slabK, slabV = ctx.saved_tensors
        Kbuf, Vbuf, dKbuf, dVbuf = ctx.bufs
        row0, r1, scale, commit = ctx.meta
        Kbuf[row0:r1].copy_(slabK)
        Vbuf[row0:r1].copy_(slabV)
        with torch.enable_grad():
            qq = q.detach().requires_grad_()
            kk = Kbuf[:r1].permute(1, 2, 0, 3).detach().requires_grad_()
            vv = Vbuf[:r1].permute(1, 2, 0, 3).detach().requires_grad_()
            out = _math_lr(qq, kk, vv, scale)
            dq, dk, dv = torch.autograd.grad(out, (qq, kk, vv), dout)
        dk = dk.permute(2, 0, 1, 3)
        dv = dv.permute(2, 0, 1, 3)
        dsK, dsV = _accum_kv(dk, dv, dKbuf, dVbuf, row0, r1, commit)
        return dq, dsK, dsV, None, None, None, None, None, None, None, None


# ----------------------------------------------------------------------------- params
class _PSlice(torch.autograd.Function):
    """Layer-range slice of a stacked weight whose grad goes STRAIGHT into the
    persistent fp32 grad stack (gstack[i0:i1] += g). Replaces autograd's
    SliceBackward (materializes a full [L,...] zero grad per use per segment) +
    the per-segment stack/cast/AccumulateGrad chain: measured ~0.4 ms/step of
    fills/adds at K16. The stacked P is a bf16 leaf whose own grad is never
    formed (backward returns None for it)."""

    @staticmethod
    def forward(ctx, p_full, gstack, i0, i1):
        ctx.meta = (gstack, i0, i1)
        return p_full[i0:i1]

    @staticmethod
    def backward(ctx, g):
        gstack, i0, i1 = ctx.meta
        if g is not None:
            gstack[i0:i1] += g
        return None, None, None, None


class _DLin(torch.autograd.Function):
    """bmm(x, w^T) with a DEFERRED weight grad: backward returns dX only and stashes
    dY into the engine's per-segment stash (x was stashed by the step); after the
    segment's backward, Wave3DX._flush_dw forms dW = sum_steps dY^T X as ONE fp32-
    output GEMM per (weight, layer) per segment straight into the grad stack.
    Replaces the per-step skinny dW GEMMs (K = C*B rows) + autograd's bf16
    InputBuffer accumulation of them across the segment (+ the _PSlice fp32 add):
    measured ~0.45 ms/step of adds/writes at K16 for the 7 big weights."""

    @staticmethod
    def forward(ctx, x, w, eng, name, l0, s):
        ctx.save_for_backward(w)
        ctx.meta = (eng, name, l0, s)
        return torch.bmm(x, w.transpose(1, 2))

    @staticmethod
    def backward(ctx, dy):
        (w,) = ctx.saved_tensors
        eng, name, l0, s = ctx.meta
        eng._stash_dy(name, l0, s, dy)
        return torch.bmm(dy, w), None, None, None, None, None


def _dlin(x, w, eng, name, l0, s):
    return _DLin.apply(x, w, eng, name, l0, s)


_DW_X = {"attn.wq.weight": "prep_x", "attn.wk.weight": "prep_x", "attn.wv.weight": "prep_x",
         "attn.wo.weight": "tail_o", "mlp.w1.weight": "tail_hb", "mlp.w3.weight": "tail_hb",
         "mlp.w2.weight": "tail_y"}


class _GradSink(torch.autograd.Function):
    """Identity on the embedding output whose backward runs LAST (all segments
    done): hands the fp32 grad stacks to the block params' .grad."""

    @staticmethod
    def forward(ctx, x0, eng):
        ctx.eng = eng
        return x0.view_as(x0)

    @staticmethod
    def backward(ctx, g):
        eng = ctx.eng
        if eng._defer and eng._dw_ranges:
            eng._flush_dw()                              # the earliest segment's deferred dW
        eng._finish_grads()
        return g, None


# ----------------------------------------------------------------------------- checkpoint
class _SegCheckpoint(torch.autograd.Function):
    """Reentrant time-segment checkpoint WITHOUT torch.utils.checkpoint: that
    function is wrapped in torch._disable_dynamo, so every torch.compile'd body
    called inside it (forward AND reentrant recompute) silently ran eager
    (profiled: no triton kernels, 2x bmm calls, +50% bwd). Forward runs the
    segment under no_grad; backward re-runs it under enable_grad on detached
    inputs and backprops -- param grads (the stacked bf16 P, closed over via
    `params`) accumulate straight into the block params' fp32 .grad."""

    @staticmethod
    def forward(ctx, seg_fn, t0, t1, K, keys, params, *vals):
        ctx.meta = (seg_fn, t0, t1, K, keys, params)
        ctx.req = [v.requires_grad for v in vals]
        ctx.ac = (torch.is_autocast_enabled("cuda"), torch.get_autocast_dtype("cuda"))
        ctx.save_for_backward(*vals)
        with torch.no_grad():
            out = seg_fn(t0, t1, K, keys, params, *vals)
        return out

    @staticmethod
    def backward(ctx, *grads):
        _raise_dynamo_limits()                           # thread-local config (engine thread)
        seg_fn, t0, t1, K, keys, params = ctx.meta
        vals = [v.detach().requires_grad_(r) for v, r in zip(ctx.saved_tensors, ctx.req)]
        ac_on, ac_dt = ctx.ac                            # re-enter the forward's autocast state
        with torch.enable_grad(), torch.autocast("cuda", dtype=ac_dt, enabled=ac_on):
            out = seg_fn(t0, t1, K, keys, params, *vals)
        pairs = [(o, g) for o, g in zip(out, grads) if o.requires_grad and g is not None]
        torch.autograd.backward([o for o, _ in pairs], [g for _, g in pairs])
        eng = getattr(seg_fn, "__self__", None)
        if eng is not None and getattr(eng, "_defer", False):
            eng._flush_dw()                              # segment-level dW GEMMs (see _DLin)
        return (None,) * 6 + tuple(v.grad for v in vals)


# ----------------------------------------------------------------------------- CUDA graphs
class _GraphedPipe(torch.autograd.Function):
    """The whole pipeline (x0 -> x_L) as CUDA-graph replays. forward: copy x0 into
    the static input, replay the prologue (zero grad buffers, restack the bf16
    weights) + the per-segment forward graphs; backward: copy dx_L into the static
    grad input, replay the per-segment recompute+backward graphs (reverse time),
    then hand the fp32 grad stacks to the params' .grad. Graphs are captured per K
    on first use (Wave3DX._capture) into ONE shared mempool."""

    @staticmethod
    def forward(ctx, x0, eng, K):
        gr = eng._graphs_for(K)
        with torch.no_grad():
            eng._x0s.copy_(x0.transpose(0, 1))
        eng._g_pro.replay()
        for g in gr["fwd"]:
            g.replay()
        eng.stats = dict(gr["stats"])
        ctx.eng, ctx.K = eng, K
        return eng._xf.transpose(0, 1)

    @staticmethod
    def backward(ctx, dxf):
        eng, K = ctx.eng, ctx.K
        gr = eng._graphs_for(K)
        with torch.no_grad():
            eng._dxf.copy_(dxf.transpose(0, 1))
        for g in gr["bwd"]:
            g.replay()
        eng._finish_grads(keep=True)
        return eng._dx0.transpose(0, 1), None, None


# ----------------------------------------------------------------------------- engine
class Wave3DX:
    def __init__(self, model, B, T, C=C_DEFAULT, seg_steps=None, use_ckpt=True,
                 attn_impl="cudnn_fe", n_snapshots=32, sac="none", store_frac=1.0,
                 compile_bodies=False, fused=False, reentrant=True, stash_attn=True,
                 graphs=False, defer_dw=True, fused_refine=True, raw_diag=False, local_window=None):
        """Backward memory/recompute policy for the STORED fraction f of the time
        segments (the rest are plain checkpoint = full recompute):
          sac='none'  : everything recomputed (f ignored)
          sac='flash' : selective checkpoint, policy keeps the flash outputs (o,
                        lse) -- eager or compiled bodies
          sac='gemm'  : policy keeps mm/bmm/flash outputs (EAGER bodies only)
          sac='lean'  : policy keeps flash out + refined K/V + LoRA downs (eager only)
          sac='store' : no checkpoint on stored segments: autograd keeps the
                        bodies' saved tensors (compiled: inductor's min-cut set)"""
        X.check_cfg(model.cfg)
        self.model = model
        cfg = model.cfg
        self.B, self.T, self.C = B, T, C
        self.n = T // C
        self.L = len(model.blocks)
        attn = model.blocks[0].attn
        self.h, self.dh, self.d = attn.n_heads, attn.head_dim, cfg.d_model
        self.scale = attn.scale
        self.seg_steps = seg_steps or 0
        self.n_snapshots = n_snapshots
        self.use_ckpt = use_ckpt
        self.attn_impl = attn_impl                      # cudnn_fe | cudnn | flash | math (wave3d_attn.KERNELS)
        self._merge = _lse_merge_c if compile_bodies else _lse_merge
        self.sac = sac
        self.store_frac = store_frac
        self._xargs = dict(dcut=getattr(cfg, "kmod_vraw_heads", 0) * cfg.head_dim,
                           tau=float(getattr(cfg, "v_clamp_tau", 0.0) or 0.0),
                           lam=float(getattr(cfg, "mult_res_lambda", 1.0)))
        self._pnames = [n for n, _ in model.blocks[0].named_parameters()]
        assert not list(model.blocks[0].named_buffers()), "Block buffers unsupported"
        self._prep, self._refine, self._tail = X.prep, X.refine, X.tail
        self._prep_a, self._prep_b, self._tail_a, self._tail_b = X.prep_a, X.prep_b, X.tail_a, X.tail_b
        self._body2 = X.refine_tail                     # fused refine(all groups)+tail region
        self.fused = fused
        # fused_refine (DEFAULT since 2026-09-02): hand-written Triton refine chain
        # (engines/wave3d/wave3d_fused.py) instead of the compiled wave3d_ops.refine: cuBLAS GEMMs +
        # one memory-bound pointwise kernel per stage in fwd and bwd (inductor's refine backward
        # re-read seven d-wide tensors 5x). Validated (validate_wave3d.py --engine x --compile all,
        # eager + --graphs 1, fp32 + bf16); K8/16/32 graph mode 1.191/1.839/3.227 -> 1.062/1.619/2.691
        # s/step, ladder 5.33 -> 4.56 d. fused_refine=False = the previous compiled path (kept for A/B).
        # Per-group path only (the fused refine+tail region keeps X.refine); set_compile leaves it uncompiled.
        self.fused_refine = bool(fused_refine) and not fused   # the fused refine+tail region keeps X.refine
        if self.fused_refine:
            from engines.wave3d.wave3d_fused import refine_fused
            self._refine = refine_fused
        self.reentrant = reentrant
        self.stash_attn = stash_attn                    # memoize flash outputs across the recompute
        # RAW-DIAGONAL operator + sliding window (2026-09-08). raw_diag: a token attends its OWN
        # raw (k, v) in every sweep (the module / WaveScan / chunked-engine convention => the
        # chunk system is nilpotent and K=C is exact); the legacy wave3d path attended the
        # refined self (a per-token fixed point that does not converge at some layers).
        # local_window: W tokens incl. self (GDN-2 SWA convention: flash window (W-1, 0)).
        self.raw_diag = bool(raw_diag)
        self.local_window = int(local_window) if local_window else None
        if self.raw_diag or self.local_window:
            assert attn_impl in ("cudnn_fe", "mathk") and stash_attn, "raw_diag/local_window: cudnn_fe (or fp32 mathk) grouped path only"
        # main (window-part) sliding length: with raw_diag the slab's last row is dropped so the
        # kernel's aligned diagonal is the previous token => W-1 keys + the raw self = W tokens.
        self._win_main = None if not self.local_window else (self.local_window - 1 if self.raw_diag else self.local_window)
        self._stash = {}
        self.defer_dw = defer_dw
        self._defer = defer_dw and not fused              # segment-level dW (see _DLin); fused region keeps X.tail
        self._dw_bufs = {}                              # (S, dt, dev) -> (x bufs, dy bufs): static per K (graphs)
        self._dw_x = self._dw_dy = None
        self._dw_ranges = {}
        self._seg_t0 = 0
        self._seg_S = 1
        self._val_fwd = False                           # eager no_grad forward: fused prep/tail bodies
        self.graphs = graphs                            # CUDA-graph the fwd + recompute/bwd (per K)
        self._graph = {}                                # K -> dict(fwd=[...], bwd=[...], stats)
        self._gpool = None
        self._g_pro = None
        self._P = None                                  # static stacked bf16 params (graph mode)
        if compile_bodies:
            self.set_compile()
        self._bufs_dt = None
        self._ar = {}
        self._pcache = {}
        self._tcache = {}
        self._last_keys = None
        self.stats = {}
        self.mem = {}

    # ------------------------------------------------------------ setup
    def set_compile(self, mode="all", dynamic=False):
        """mode: 'none' | 'refine' (per-group refine only) | 'all' (prep + fused
        refine/tail region). Static shapes by default: one graph per step pattern
        (~2K+A_max shapes per K; dynamic-shape kernels measured 2.7x slower)."""
        if mode == "none":
            self._prep, self._refine, self._tail = X.prep, X.refine, X.tail
            self._prep_a, self._prep_b, self._tail_a, self._tail_b = X.prep_a, X.prep_b, X.tail_a, X.tail_b
            self._body2 = X.refine_tail
            self._merge = _lse_merge
            if self.fused_refine:
                from engines.wave3d.wave3d_fused import refine_fused
                self._refine = refine_fused
            return
        self._merge = _lse_merge_c
        _raise_dynamo_limits()
        if not self.fused_refine:                       # the Triton refine is not a torch.compile body
            self._refine = torch.compile(X.refine, dynamic=dynamic)
        if mode == "all":
            self._prep = torch.compile(X.prep, dynamic=dynamic)
            self._tail = torch.compile(X.tail, dynamic=dynamic)
            self._prep_a, self._prep_b, self._tail_a, self._tail_b = (
                torch.compile(f, dynamic=dynamic) for f in (X.prep_a, X.prep_b, X.tail_a, X.tail_b))
            self._body2 = torch.compile(X.refine_tail, dynamic=dynamic)

    def n_steps(self, K):
        return self.n - 1 + self.L * K

    def _active(self, t, K):
        lim = self.n + K - 2
        la = 0 if t <= lim else (t - lim + K - 1) // K
        lb = min(self.L - 1, t // K)
        return la, lb

    def max_active(self, K):
        return max(self._active(t, K)[1] - self._active(t, K)[0] + 1
                   for t in range(self.n_steps(K)))

    def _arange(self, K, dev):
        key = (K, str(dev))
        if key not in self._ar:
            self._ar[key] = torch.arange(-K * self.C, self.T + K * self.C, device=dev)
        return self._ar[key]

    def _stack_params(self):
        """Stacked block params [L, ...] (2-D weights cast to the autocast dtype)
        as detached LEAVES requiring grad, plus zeroed fp32 grad stacks. Grads
        reach the params through _PSlice -> gstack -> _finish_grads."""
        per = [dict(b.named_parameters()) for b in self.model.blocks]
        dt = X.act_dtype()
        merged_src = {n for names in X.MERGED.values() for n in names}
        out, gst = {}, {}
        with torch.no_grad():
            raw = {}
            for n in self._pnames:
                s = torch.stack([p[n] for p in per])
                if dt is not None and s.dim() >= 2:
                    s = s.to(dt)
                raw[n] = s
            raw = X.combine_params(raw)                  # merged refine weights (see wave3d_ops)
            for n, s in raw.items():
                if n in merged_src:
                    continue
                out[n] = s.contiguous().requires_grad_(True)
                gst[n] = torch.zeros(s.shape, device=s.device, dtype=torch.float32)
        self._gstack = gst
        return out

    def _finish_grads(self, keep=False):
        """gstack -> block params' .grad (add if the trainer accumulates); the
        merged refine weights are split back to their source params. keep: the
        stacks are the static graph-mode buffers (zeroed by the prologue graph)."""
        gs = self._gstack_static if keep else self._gstack
        r = gs["attn.ud2"].shape[1] // 2
        view = {X.GA2[0]: gs["attn.ga2"][:, :r], X.GA2[1]: gs["attn.ga2"][:, r:],
                X.UD2[0]: gs["attn.ud2"][:, :r], X.UD2[1]: gs["attn.ud2"][:, r:]}
        for i, n in enumerate(X.UPS5):
            view[n] = gs["attn.ups5"][:, i]
        with torch.no_grad():
            for l, blk in enumerate(self.model.blocks):
                for n, p in blk.named_parameters():
                    g = (view[n] if n in view else gs[n])[l]
                    if p.grad is None:
                        p.grad = g.clone()
                    else:
                        p.grad.add_(g)
        if not keep:
            self._gstack = None

    def _alloc(self, dt, dev):
        if self._bufs_dt == (dt, str(dev)):
            return
        L, T, B, d, h, dh = self.L, self.T, self.B, self.d, self.h, self.dh
        z = lambda *shp, dtype=dt: torch.zeros(*shp, device=dev, dtype=dtype)
        self.xh, self.kf, self.vf = z(L, T, B, d), z(L, T, B, d), z(L, T, B, d)
        self.qr = z(L, T, B, h, dh)
        self.Kbuf, self.Vbuf = z(L, T, B, h, dh), z(L, T, B, h, dh)
        self.dKbuf, self.dVbuf = z(L, T, B, h, dh), z(L, T, B, h, dh)   # cache dtype (== autograd's)
        # grad caches in the CACHE dtype: the K window-read contributions of a chunk
        # accumulate in bf16 exactly as the reference's autograd does for its bf16
        # k/v/q slices (InputBuffer adds); same-dtype adds also take the vectorized
        # kernel (the fp32 += bf16 mixed path ran at ~3.8 TB/s) and the writer's
        # backward hands bf16 grads back without autograd's fp32->bf16 cast.
        self.gxh, self.gkf, self.gvf = (z(L, T, B, d) for _ in range(3))
        self.gqr = z(L, T, B, h, dh)
        self._tok0 = torch.zeros((), device=dev)
        self._bufs_dt = (dt, str(dev))
        if self.attn_impl == "cudnn_fe":
            # pre-size the cuDNN-FE workspace for the largest window any K can present (Lq <= T)
            # BEFORE any graph capture: captured graphs bake the workspace address (see wave3d_attn)
            _KERNELS["cudnn_fe"].reserve(dev, _KERNELS["cudnn_fe"].ws_bound(T * B, h))

    # ------------------------------------------------------------ one step
    def _attend(self, q, slabK, slabV, l, row0, r1, commit, t):
        assert not (self.raw_diag or self.local_window), "raw_diag/local_window need the grouped (stash_attn) path"
        args = (q, slabK, slabV, self.Kbuf[l], self.Vbuf[l], self.dKbuf[l], self.dVbuf[l],
                row0, r1, self.scale, commit)
        if self._impl in _KERNELS and self._impl != "flash":
            return _AttendCacheK.apply(*args, _KERNELS[self._impl])
        fn = _AttendCache if self._impl == "flash" else _AttendCacheMath
        return fn.apply(*args)

    def _slice(self, P, i0, i1):
        """Per-forward cache of the [i0:i1] layer-slices of the stacked params
        (the same (i0, i1) recurs every K steps; ~130 view ops/step otherwise)."""
        key = (i0, i1, torch.is_grad_enabled())        # grad-mode keyed: the reentrant
        c = self._pcache                                #   forward runs under no_grad
        if c.get("_P") is not P:
            c.clear()
            c["_P"] = P
        if key not in c:
            if torch.is_grad_enabled():
                c[key] = {k: _PSlice.apply(v, self._gstack[k], i0, i1) for k, v in P.items()}
            else:
                c[key] = {k: v[i0:i1] for k, v in P.items()}
        return c[key]

    def _trig(self, K, t, tag, rows):
        """Cached rope cos/sin gathers for (K, step, group): [G, Wv, 1, 1, dh/2].
        rows: list of (r0, Wv) absolute row ranges per layer of the group."""
        key = (K, t, tag)
        c = self._tcache
        if key not in c:
            cos_all, sin_all = self.model.rope_cos, self.model.rope_sin
            ar = self._arange(K, cos_all.device)
            Z = K * self.C
            pos = torch.stack([ar[Z + r0:Z + r0 + Wv] for r0, Wv in rows])
            c[key] = (cos_all[pos][:, :, None, None, :].contiguous(),
                      sin_all[pos][:, :, None, None, :].contiguous())
        return c[key]

    def _step(self, st, t, K, P):
        n, C, B, T, h, dh, d = self.n, self.C, self.B, self.T, self.h, self.dh, self.d
        KC = K * C
        la, lb = self._active(t, K)
        layers = list(range(la, lb + 1))
        tls = [t - l * K for l in layers]
        new = dict(st)

        # ---- prep: layers whose new chunk exists (tl <= n-1) = upper layer range
        prep = [(l, tl) for l, tl in zip(layers, tls) if tl <= n - 1]
        kr_new, vr_new = {}, {}
        if prep:
            lp, Gp = prep[0][0], len(prep)
            xs = [new[f"x{l}.{tl}"] for l, tl in prep]                        # Gp x [C, B, d] fp32
            cs, sn = self._trig(K, t, "prep", [(tl * C, C) for _, tl in prep])
            Pp = self._slice(P, lp, lb + 1)
            if self._defer and not self._val_fwd:
                s_ = t - self._seg_t0
                _, xb = self._prep_a(Pp, xs)                                  # stack fused in
                if torch.is_grad_enabled():
                    self._stash_x("prep_x", lp, s_, xb)
                q = _dlin(xb, Pp["attn.wq.weight"], self, "attn.wq.weight", lp, s_)
                kf = _dlin(xb, Pp["attn.wk.weight"], self, "attn.wk.weight", lp, s_)
                vf = _dlin(xb, Pp["attn.wv.weight"], self, "attn.wv.weight", lp, s_)
                q_r, k_r = self._prep_b(q, kf, cs, sn, h, dh, (C, B))
            else:
                xc = torch.stack(xs)
                _, xb, kf, vf, q_r, k_r = self._prep(Pp, xc.view(Gp, C * B, d), cs, sn, h, dh, (C, B))
            toks_in = [new[f"tok{l}"] if tl > 0 else self._tok0 for l, tl in prep]
            toks_out = _CacheWrite.apply(xb, kf, vf, q_r, self, tuple(prep), *toks_in)
            for g, (l, tl) in enumerate(prep):
                new[f"tok{l}"] = toks_out[g]
                kr_new[l] = k_r[g]
                vr_new[l] = vf[g].view(C, B, h, dh)

        # ---- windows: exact [lo, hi] chunk range per layer; group equal-length affine runs
        meta_w = {}
        groups = []                                   # (l0, G, Wv, r0)
        for l, tl in zip(layers, tls):
            lo, hi = max(0, tl - K + 1), min(n - 1, tl)
            Wv, r0 = (hi - lo + 1) * C, lo * C
            meta_w[l] = (lo, hi, Wv, r0)
            if groups and groups[-1][2] == Wv and groups[-1][3] - groups[-1][1] * KC == r0:
                l0, G, _, gr0 = groups[-1]
                groups[-1] = (l0, G + 1, Wv, gr0)
            else:
                groups.append((l, 1, Wv, r0))

        reads, o_lists, tail_o = {}, {}, {}
        all_toks = [new[f"tok{l}"] for l in layers]
        rd = _CacheRead.apply(self, tuple(groups), K, *all_toks)
        grouped = self._impl in _KERNELS and self.stash_attn
        if prep:
            kr_all = k_r                                                  # [Gp, C, B, h, dh]
            vf_all = vf                                                   # [Gp, C*B, d]
            pidx = {l: g for g, (l, _) in enumerate(prep)}
        else:
            kr_all = vf_all = None
            pidx = {}
        for gi, (l0, G, Wv, r0) in enumerate(groups):
            xh_w, kf_w, vf_w, qr_w = rd[4 * gi:4 * gi + 4]
            reads[gi] = (xh_w, kf_w, vf_w)
            if grouped:
                meta, kvp = [], []
                for g in range(G):
                    l = l0 + g
                    tl = t - l * K
                    lo, hi, _, _ = meta_w[l]
                    prev = f"Kw{l}" in new
                    if prev:
                        kvp += [new[f"Kw{l}"], new[f"Vw{l}"]]
                    meta.append((l, max(0, tl - K) * C, (hi + 1) * C, C if tl >= K else 0,
                                 (t, l), pidx.get(l, -1), prev, tl >= K - 1))
                if self.raw_diag:
                    cs_g, sn_g = self._trig(K, t, gi, [(r0 - g * KC, Wv) for g in range(G)])
                    kfw, vfw = reads[gi][1], reads[gi][2]
                else:
                    cs_g = sn_g = kfw = vfw = None
                o_g, o_tail = _AttendGroup.apply(qr_w, kr_all, vf_all, self, tuple(meta), kfw, vfw, cs_g, sn_g, *kvp)
                o_lists[gi] = [o_g]                                       # already stacked
                o_tails = o_tail.unbind(0) if o_tail.numel() else ()     # unbind: no select_backward zero-fills
                it = 0
                for g in range(G):
                    if meta[g][7]:
                        tail_o[l0 + g] = o_tails[it]
                        it += 1
                continue
            os = []
            for g in range(G):
                l = l0 + g
                tl = t - l * K
                lo, hi, _, _ = meta_w[l]
                q = qr_w[g].permute(1, 2, 0, 3)                              # [B, h, Wv, dh]
                c0 = max(0, tl - K)
                commit = C if tl >= K else 0
                Kw_p, Vw_p = new.get(f"Kw{l}"), new.get(f"Vw{l}")
                kr, vr = kr_new.get(l), vr_new.get(l)
                pk = [Kw_p] if Kw_p is not None else []
                pv = [Vw_p] if Vw_p is not None else []
                if kr is not None:
                    pk.append(kr); pv.append(vr)
                slabK = torch.cat(pk, 0) if len(pk) > 1 else pk[0]
                slabV = torch.cat(pv, 0) if len(pv) > 1 else pv[0]
                o = self._attend(q, slabK, slabV, l, c0 * C, (hi + 1) * C, commit, t)
                os.append(o.permute(2, 0, 1, 3).reshape(Wv * B, d))         # time-major merged
                if tl >= K - 1:
                    tail_o[l] = os[-1][:C * B]                               # chunk lo (final o)
            o_lists[gi] = os

        # ---- tail: layers with tl >= K-1 (oldest chunk final) = lower layer range
        tail = [(l, tl) for l, tl in zip(layers, tls) if tl >= K - 1]
        Pt = tail_x = None
        tail_os = []
        if tail:
            lt, Gt = tail[-1][0], len(tail)
            txs = [new[f"x{l}.{tl - K + 1}"] for l, tl in tail]
            tail_x = None if (self._defer and not self._val_fwd) else torch.stack(txs).view(Gt, C * B, d)
            tail_os = [tail_o[l] for l, _ in tail]
            Pt = self._slice(P, la, lt + 1)

        # ---- refine every group's window (+ tail) in one region
        keep = self._impl != "math" and self.sac == "lean"
        Pg, css, sns, rows_list = [], [], [], []
        for gi, (l0, G, Wv, r0) in enumerate(groups):
            Pg.append(self._slice(P, l0, l0 + G))
            cs, sn = self._trig(K, t, gi, [(r0 - g * KC, Wv) for g in range(G)])
            css.append(cs); sns.append(sn); rows_list.append((Wv, B))
        if self.fused:
            outs, xn = self._body2(Pg, [reads[gi][0] for gi in range(len(groups))],
                                   [o_lists[gi] for gi in range(len(groups))],
                                   [reads[gi][1] for gi in range(len(groups))],
                                   [reads[gi][2] for gi in range(len(groups))], css, sns, rows_list,
                                   Pt, tail_x, tail_os, h, dh, **self._xargs)
        else:
            outs = []
            for gi in range(len(groups)):
                os_ = o_lists[gi]
                if len(os_) == 1 and os_[0].dim() == 3:
                    o_g = os_[0]                                           # grouped attention output
                else:
                    o_g = torch.stack(os_) if len(os_) > 1 else os_[0].unsqueeze(0)
                outs.append(self._refine(Pg[gi], reads[gi][0], o_g, reads[gi][1], reads[gi][2],
                                         css[gi], sns[gi], h, dh, rows_dims=rows_list[gi], **self._xargs))
            xn = None
            if tail:
                o_c = torch.stack(tail_os)
                if self._defer and not self._val_fwd:
                    s_ = t - self._seg_t0
                    gr = torch.is_grad_enabled()
                    if gr:
                        self._stash_x("tail_o", la, s_, o_c)
                    h_wo = _dlin(o_c, Pt["attn.wo.weight"], self, "attn.wo.weight", la, s_)
                    x1, hb = self._tail_a(Pt, txs, h_wo)                      # stack fused in
                    if gr:
                        self._stash_x("tail_hb", la, s_, hb)
                    a1 = _dlin(hb, Pt["mlp.w1.weight"], self, "mlp.w1.weight", la, s_)
                    a3 = _dlin(hb, Pt["mlp.w3.weight"], self, "mlp.w3.weight", la, s_)
                    y = self._tail_b(a1, a3)
                    if gr:
                        self._stash_x("tail_y", la, s_, y)
                    xn = x1 + _dlin(y, Pt["mlp.w2.weight"], self, "mlp.w2.weight", la, s_)
                else:
                    xn = self._tail(Pt, tail_x, o_c)
        if tail:
            xns = xn.unbind(0)                                                # unbind, not select (see below)
            for g, (l, tl) in enumerate(tail):
                new[f"x{l + 1}.{tl - K + 1}"] = xns[g].view(C, B, d)
        for gi, (l0, G, Wv, r0) in enumerate(groups):
            Kw, Vw = outs[gi]
            if keep:
                Kw, Vw = torch.ops.wave3d.keep(Kw), torch.ops.wave3d.keep(Vw)
            # unbind: one gather in backward instead of G select_backward zero-fills of the
            # full [G, Wv, B, h, dh] + G InputBuffer adds (measured 0.1+ ms/step at K16)
            Kws, Vws = Kw.unbind(0), Vw.unbind(0)
            for g in range(G):
                l = l0 + g
                if t - l * K == n + K - 2:                                     # layer l finished
                    for s_ in ("Kw", "Vw", "tok"):
                        new.pop(f"{s_}{l}", None)
                    for c in range(n):
                        new.pop(f"x{l}.{c}", None)
                else:
                    new[f"Kw{l}"], new[f"Vw{l}"] = Kws[g], Vws[g]
            self._rows += G * Wv * B
        return new

    # ------------------------------------------------------------ deferred dW (see _DLin)
    def _alloc_dw(self, S):
        """Per-segment X / dY stash for the 7 big weights: [L, S, C*B, cols] in the
        cache dtype; kept per (S, dtype) forever (graph replays bake the addresses)."""
        dt, dev = self.xh.dtype, self.xh.device
        key = (S, dt, str(dev))
        bufs = self._dw_bufs.get(key)
        if bufs is None:
            L, R, d = self.L, self.C * self.B, self.d
            hid = self.model.blocks[0].mlp.w1.weight.shape[0]
            z = lambda cols: torch.empty(L, S, R, cols, device=dev, dtype=dt)
            xb = {"prep_x": z(d), "tail_o": z(d), "tail_hb": z(d), "tail_y": z(hid)}
            dyb = {n: z(hid if n in ("mlp.w1.weight", "mlp.w3.weight") else d) for n in _DW_X}
            bufs = self._dw_bufs[key] = (xb, dyb)
        self._dw_x, self._dw_dy = bufs

    @torch.no_grad()
    def _stash_x(self, xname, l0, s, x):
        # no_grad is essential: an in-place copy_ of a grad-requiring x under grad
        # mode would make the whole stash buffer autograd-tracked (CopySlices chain)
        self._dw_x[xname][l0:l0 + x.shape[0], s].copy_(x)

    @torch.no_grad()
    def _stash_dy(self, name, l0, s, dy):
        G = dy.shape[0]
        self._dw_dy[name][l0:l0 + G, s].copy_(dy)
        for l in range(l0, l0 + G):
            r = self._dw_ranges.get((name, l))
            self._dw_ranges[(name, l)] = (s, s + 1) if r is None else (min(r[0], s), max(r[1], s + 1))

    def _flush_dw(self):
        """After a segment's backward: dW[l] += dY_l^T X_l over the segment's steps
        (one GEMM per (weight, layer), fp32 output) into the fp32 grad stack."""
        gs = self._gstack
        for (name, l), (s0, s1) in self._dw_ranges.items():
            dy = self._dw_dy[name][l, s0:s1]
            x = self._dw_x[_DW_X[name]][l, s0:s1]
            dW = torch.ops.aten.mm.dtype(dy.reshape(-1, dy.shape[-1]).t(),
                                         x.reshape(-1, x.shape[-1]), torch.float32)
            gs[name][l].add_(dW)
        self._dw_ranges = {}

    def _segment(self, t0, t1, K, keys, params, *vals):
        self._pcache.clear()                            # fresh _PSlice nodes per segment
        self._seg_t0 = t0
        if self._defer:
            self._alloc_dw(self._seg_S)
            if torch.is_grad_enabled() and self._dw_ranges:
                # torch-checkpoint paths (no _SegCheckpoint/bseg hook): the previous
                # (later-in-time) segment's backward is complete once this recompute
                # starts (autograd runs nodes in decreasing sequence-nr order), so
                # its stashed dY/X are final -> flush before this segment reuses them
                self._flush_dw()
            self._dw_ranges = {}
        st = dict(zip(keys, vals))
        for t in range(t0, t1):
            st = self._step(st, t, K, params)
        keys_out = sorted(st.keys())
        self._last_keys = keys_out
        return tuple(st[k] for k in keys_out)

    # ------------------------------------------------------------ API
    def _pipeline(self, idx, K, seg_steps=None):
        model = self.model
        B, T = idx.shape
        assert B == self.B and T == self.T, (idx.shape, self.B, self.T)
        steps = self.n_steps(K)
        S = seg_steps or self.seg_steps or max(1, -(-steps // self.n_snapshots))
        self._seg_S = S
        # no_grad (validation) forward: the single fused prep / tail bodies (fewer launches;
        # the CPU-bound eager path). Training (grad) and graph capture use the split
        # deferred-dW path for BOTH forward and recompute (identical arithmetic).
        self._val_fwd = not torch.is_grad_enabled()
        params = self._stack_params()
        ac = torch.is_autocast_enabled("cuda")
        impl = self.attn_impl
        if impl not in ("math", "mathk") and not ac and model.tok_emb.weight.dtype == torch.float32:
            impl = "math"                                # no fp32 flash/cuDNN kernel
        self._impl = impl
        dt = torch.get_autocast_dtype("cuda") if ac else model.tok_emb.weight.dtype
        self._alloc(dt, idx.device)
        for g in (self.dKbuf, self.dVbuf, self.gxh, self.gkf, self.gvf, self.gqr):
            g.zero_()
        self._stash = {}                                 # new forward: drop the flash memo
        x0 = _GradSink.apply(model.tok_emb(idx), self).transpose(0, 1)         # [T, B, d] fp32 view
        st = {f"x0.{c}": x0[c * self.C:(c + 1) * self.C] for c in range(self.n)}
        self._rows = 0
        nseg = -(-steps // S)
        ctx_fn = None
        if self.sac in ("gemm", "lean"):
            assert self._body2 is X.refine_tail and self._refine is X.refine, \
                "policy-SAC over GEMMs needs eager bodies (see _GEMM_OPS note)"
            ctx_fn = functools.partial(
                create_selective_checkpoint_contexts,
                _mk_policy(_LEAN_OPS if self.sac == "lean" else _GEMM_OPS, self.sac == "lean"))
        elif self.sac == "flash":
            ctx_fn = functools.partial(create_selective_checkpoint_contexts,
                                       _mk_policy(_FLASH_OPS, False))
        f = self.store_frac
        self._stored = 0
        for i, t0 in enumerate(range(0, steps, S)):
            t1 = min(steps, t0 + S)
            keys = sorted(st.keys())
            vals = tuple(st[k] for k in keys)
            stored = self.sac != "none" and int((i + 1) * f) > int(i * f)
            if self.use_ckpt and not (stored and self.sac == "store"):
                if stored:
                    # selective checkpoint needs the non-reentrant (hooks) form
                    kw = dict(use_reentrant=False, preserve_rng_state=False, context_fn=ctx_fn)
                    self._stored += 1
                    out = checkpoint(self._segment, t0, t1, K, keys, params, *vals, **kw)
                elif self.reentrant:
                    # our own reentrant Function: no saved-tensor hooks (~2.7 ms/step
                    # at K16) and no dynamo-disable (compiled bodies stay compiled)
                    out = _SegCheckpoint.apply(self._segment, t0, t1, K, keys, params, *vals)
                else:
                    out = checkpoint(self._segment, t0, t1, K, keys, params, *vals,
                                     use_reentrant=False, preserve_rng_state=False)
            else:
                if stored:
                    self._stored += 1
                out = self._segment(t0, t1, K, keys, params, *vals)
            st = dict(zip(self._last_keys, out))
        assert len(st) == self.n, sorted(st.keys())
        self.stats = dict(steps=steps, rows_per_step=self._rows / steps,
                          max_active=self.max_active(K), seg_steps=S, n_seg=nseg,
                          stored_seg=self._stored)
        return torch.cat([st[f"x{self.L}.{c}"] for c in range(self.n)], 0).transpose(0, 1)

    def forward_logits(self, idx, K, seg_steps=None):
        if self.graphs and torch.is_grad_enabled() and torch.is_autocast_enabled("cuda"):
            x = self._pipeline_graphed(idx, K)
        else:
            x = self._pipeline(idx, K, seg_steps)
        return self.model.lm_head(self.model.norm_f(x))

    # ------------------------------------------------------------ CUDA graphs
    def set_graphs(self, flag=True):
        self.graphs = bool(flag)

    def _pipeline_graphed(self, idx, K):
        """Training forward (grad enabled) through _GraphedPipe: tok_emb and the
        head stay eager autograd; everything in between is graph replays."""
        B, T = idx.shape
        assert B == self.B and T == self.T, (idx.shape, self.B, self.T)
        x0 = self.model.tok_emb(idx)                                        # [B, T, d] fp32
        return _GraphedPipe.apply(x0, self, K)

    def _static_params(self, dev):
        """Static stacked params + fp32 grad stacks (allocated once; the prologue
        graph re-fills them from the live weights every training step)."""
        if self._P is not None:
            return
        P = self._stack_params()
        self._P = {n: v.detach().clone().requires_grad_(True) for n, v in P.items()}
        self._gstack_static = {n: g.clone() for n, g in self._gstack.items()}
        L, T, B, d = self.L, self.T, self.B, self.d
        f32 = dict(device=dev, dtype=torch.float32)
        self._x0s = torch.zeros(T, B, d, **f32)
        self._xf = torch.zeros(T, B, d, **f32)
        self._dxf = torch.zeros(T, B, d, **f32)
        self._dx0 = torch.zeros(T, B, d, **f32)

    @torch.no_grad()
    def _prologue(self):
        """Per training step: zero the grad buffers, restack the live weights."""
        for g in (self.dKbuf, self.dVbuf, self.gxh, self.gkf, self.gvf, self.gqr):
            g.zero_()
        for g in self._gstack_static.values():
            g.zero_()
        per = [dict(b.named_parameters()) for b in self.model.blocks]
        P = self._P
        r = P["attn.ud2"].shape[1] // 2
        for n in self._pnames:
            src = [p[n] for p in per]
            if n in X.GA2:
                dst = P["attn.ga2"][:, :r] if n == X.GA2[0] else P["attn.ga2"][:, r:]
            elif n in X.UD2:
                dst = P["attn.ud2"][:, :r] if n == X.UD2[0] else P["attn.ud2"][:, r:]
            elif n in X.UPS5:
                dst = P["attn.ups5"][:, X.UPS5.index(n)]
            else:
                dst = P[n]
            dst.copy_(torch.stack(src))

    def _capture_graph(self, fn):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self._gpool):
            out = fn()
        if self._gpool is None:
            self._gpool = g.pool()
        return g, out

    def _graphs_for(self, K):
        if K not in self._graph:
            self._capture(K)
        return self._graph[K]

    def _capture(self, K):
        """Capture prologue (once), per-segment forward and per-segment
        recompute+backward graphs for K into the shared pool. The SAME closures
        run twice: eagerly first (warm-up: every compiled body -- fwd and bwd --
        is compiled in exactly the capture's global state), then under capture
        with the compile stance 'fail_on_recompile' (any compile inside a capture
        would autotune on a foreign stream). Capture order == replay order (fwd
        0..N-1, bwd N-1..0) with the cross-graph tensors (segment boundary states,
        flash stash, boundary grads) alive exactly as long as in replay, so pool
        reuse is safe; all Python refs are dropped afterwards (another K's capture
        may then alias that memory: one K per training step)."""
        import gc
        import time as _time
        model = self.model
        dev = model.tok_emb.weight.device
        ac = torch.is_autocast_enabled("cuda")
        assert ac and self.attn_impl != "math", "graph mode: bf16 autocast + flash/cudnn only"
        dt = torch.get_autocast_dtype("cuda")
        self._impl = self.attn_impl
        self._alloc(dt, dev)
        self._static_params(dev)
        w0 = _time.perf_counter()
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        m0 = torch.cuda.memory_allocated()
        self._run_bodies(K, dt, capture=False)                  # warm-up / compile
        gc.collect()
        self.mem[f"K{K}_warmup_peak_GB"] = (torch.cuda.max_memory_allocated() - m0) / 1e9
        torch.cuda.reset_peak_memory_stats()
        with torch.compiler.set_stance("fail_on_recompile"):
            fwd, bwd, stats = self._run_bodies(K, dt, capture=True)
        self.mem[f"K{K}_capture_peak_GB"] = (torch.cuda.max_memory_allocated() - m0) / 1e9
        self._stash = {}
        self._pcache.clear()
        gc.collect()
        torch.cuda.synchronize()
        self._graph[K] = dict(fwd=fwd, bwd=bwd, stats=stats)
        self.mem[f"capture_K{K}_s"] = _time.perf_counter() - w0
        self.mem[f"capture_K{K}_reserved_GB"] = torch.cuda.memory_reserved() / 1e9

    def _run_bodies(self, K, dt, capture):
        params, self._gstack = self._P, self._gstack_static
        steps = self.n_steps(K)
        S = self.seg_steps or max(1, -(-steps // self.n_snapshots))
        self._seg_S = S
        self._val_fwd = False
        nseg = -(-steps // S)
        n, C, L = self.n, self.C, self.L
        cap = self._capture_graph if capture else (lambda fn: (None, fn()))
        if capture and self._g_pro is None:
            self._g_pro, _ = self._capture_graph(self._prologue)
        elif capture:
            self._g_pro.replay()
        else:
            self._prologue()
        self._stash = {}
        self._rows = 0
        # ---- forward segments
        fwd, bounds = [], []
        st = {f"x0.{c}": self._x0s[c * C:(c + 1) * C] for c in range(n)}
        for i, t0 in enumerate(range(0, steps, S)):
            t1 = min(steps, t0 + S)
            keys = sorted(st.keys())
            vals = tuple(st[k] for k in keys)
            bounds.append((t0, t1, keys, vals))

            def fseg(t0=t0, t1=t1, keys=keys, vals=vals, last=(t1 == steps)):
                with torch.no_grad(), torch.autocast("cuda", dtype=dt, cache_enabled=False):
                    out = self._segment(t0, t1, K, keys, params, *vals)
                    if last:
                        stf = dict(zip(self._last_keys, out))
                        torch.cat([stf[f"x{L}.{c}"] for c in range(n)], 0, out=self._xf)
                return out
            g, out = cap(fseg)
            fwd.append(g)
            st = dict(zip(self._last_keys, out))
        assert len(st) == n, sorted(st.keys())
        stats = dict(steps=steps, rows_per_step=self._rows / steps, max_active=self.max_active(K),
                     seg_steps=S, n_seg=nseg, stored_seg=0)
        # ---- backward segments (reverse)
        gr_out = {f"x{L}.{c}": self._dxf[c * C:(c + 1) * C] for c in range(n)}
        bwd = []
        for i in range(nseg - 1, -1, -1):
            t0, t1, keys, vals = bounds[i]

            def bseg(t0=t0, t1=t1, keys=keys, vals=vals, gr_out=gr_out, first=(i == 0)):
                vd = [v.detach().requires_grad_(True) for v in vals]
                with torch.enable_grad(), torch.autocast("cuda", dtype=dt, cache_enabled=False):
                    out = self._segment(t0, t1, K, keys, params, *vd)
                pairs = [(o, gr_out.get(k)) for k, o in zip(self._last_keys, out)]
                pairs = [(o, g) for o, g in pairs if g is not None and o.requires_grad]
                torch.autograd.backward([o for o, _ in pairs], [g for _, g in pairs])
                if self._defer:
                    self._flush_dw()
                gin = {k: v.grad for k, v in zip(keys, vd)}
                if first:
                    for c in range(n):
                        g = gin.get(f"x0.{c}")
                        dst = self._dx0[c * C:(c + 1) * C]
                        if g is None:
                            dst.zero_()
                        else:
                            dst.copy_(g)
                return gin
            g, gr_out = cap(bseg)
            bwd.append(g)
        return fwd, bwd, stats

    def forward(self, idx, targets, K, seg_steps=None):
        logits = self.forward_logits(idx, K, seg_steps)
        return F.cross_entropy(logits.float().view(-1, logits.shape[-1]),
                               targets.reshape(-1), ignore_index=-1)
