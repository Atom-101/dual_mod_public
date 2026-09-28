"""wave3d: LAYER-BATCHED 3D wavefront engine for the DM 1.3B.

Same operator as the production per-layer engine `engine1_layer_split`
(the earlier chunked_split engine, not shipped) -- c=64 chunked, K-sweep intra-layer diagonal
wavefront with committed prefix + sliding active window -- but ALL 18 layers'
diagonals run CONCURRENTLY, offset K pipeline steps apart, so every per-layer
op at a global step is batched over the ~A active layers.

    layer l at global step t is at intra-layer step t_l = t - l*K,
    active iff 0 <= t_l <= n_chunks+K-2.        total steps = n_chunks-1 + L*K

Dependency proof (why the offset is exactly K): layer l commits chunk k (its
K-th sweep, o -> _block_tail -> x_{l+1}[k]) at t = l*K + k + K - 1; layer l+1
preps chunk k at t = (l+1)*K + k = one step later.

PER-STEP BODY (mirrors engine1_layer_split at intra-step t_l, per active layer)
  prep    (t_l <= n-1)   : new chunk t_l enters RAW (wq/wk/project_v + rope on
                           x_l[chunk]) -> window slot K-1.            [vmap over layers]
  slab    (all active)   : cache[ext chunks t_l..t_l+K] <- (K_win refined at
                           t_l-1, raw new chunk). Committed prefix == cache
                           rows left of the window (a chunk exiting the window
                           simply stays put: its last refine IS its commit).
  attend  (all active)   : q = valid window slots, kv = cache[0 : (hi+1)*C],
                           lower-right causal flash == _attend_split's Part A
                           (dense prefix) + Part B (window causal) LSE-merged.
                           One aten flash call per layer (the kernel's
                           is_causal with Lq<Lk IS lower-right aligned).
  tail    (t_l >= K-1)   : chunk lo's o (slot 0) is FINAL -> x + wo(o) +
                           mlp(mlp_norm(.)) -> x_{l+1}[chunk lo]. [vmap over layers]
  refine  (all active)   : (K_win, V_win) = refine_kv + rope over the window
                           (padded slots computed then zeroed).   [vmap over layers]
Final layer's x -> norm_f -> lm_head -> CE (model head, fp32 logits).

Cache layout: per layer [B,h,T,dh] at absolute positions; the (K+1)-chunk slab
write (previous refined window + raw new chunk) is clipped to the sequence.

Params: a differentiable torch.stack of the 18 blocks' params each forward
(2-D weights pre-cast to bf16 once under autocast; 1-D norm gains / gate
biases stay fp32, exactly what autocast would have done per call) -> gathered
per step as a contiguous VIEW [A,...] of the active layers -> vmap(functional_
call(Block template)). nn.Linear -> bmm automatically; the module code
(refine_kv, gates, norms, MLP) is reused verbatim so semantics are identical.
Grads flow back through the stack onto the ORIGINAL model.blocks[i] params.

Backward: torch.utils.checkpoint over TIME SEGMENTS of `seg_steps` global
steps (use_reentrant=False); the pipeline state is a dict of tensors, keys
deterministic in t. No in-place writes on autograd-visible tensors.

ENGINES. This file is the ORIGINAL (vmap) engine, kept selectable for A/B:
  make_engine(model, B, T, engine="vmap", math="vmap"|"explicit")
    math="explicit" swaps the vmapped bodies for engines/wave3d/wave3d_ops.py
    (stacked-weight torch.bmm + written-out refine chain), same layout.
  make_engine(model, B, T, engine="x", ...)  -> engines/wave3d/wave3d_x.Wave3DX
    the fast engine: copy-free time-major caches, exact windows, explicit
    bmm, static-shape compiled bodies, own reentrant segment checkpoint with
    a flash-output stash. See its module docstring.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, vmap
from torch.utils.checkpoint import checkpoint

from models.dualmod.rope import apply_rope
from engines.wave3d import wave3d_ops as X

C_DEFAULT = 64


class _Body(nn.Module):
    """One Block wrapped so functional_call can reparametrize it; exposes the
    three per-layer bodies. Param names are 'blk.<block param name>'."""

    def __init__(self, blk):
        super().__init__()
        self.blk = blk

    def forward(self, mode, *args):
        blk = self.blk
        attn = blk.attn
        if mode == "prep":
            x_c, cos, sin = args
            xhat = blk.attn_norm(x_c)
            k_flat = attn.wk(xhat)
            v_flat = attn.project_v(xhat)
            q_r = apply_rope(attn._split(attn.wq(xhat)), cos, sin)
            k_r = apply_rope(attn._split(k_flat), cos, sin)
            v = attn._split(v_flat)
            return xhat, k_flat, v_flat, q_r, k_r, v
        if mode == "refine":
            xhat, o_merged, k_flat, v_flat, cos, sin = args
            kpp, vpp = attn.refine_kv(xhat, o_merged, k_flat, v_flat)
            return apply_rope(attn._split(kpp), cos, sin), attn._split(vpp)
        if mode == "tail":
            x_c, o_c = args
            x = x_c + attn.wo(o_c)
            return x + blk.mlp(blk.mlp_norm(x))
        raise ValueError(mode)


def _splice(base, r0, piece):
    """Out-of-place `base[r0:r0+len(piece)] = piece` along dim 0 via contiguous
    memcpy-speed copy_ (torch.cat's batched-copy kernel ran at ~1TB/s on these)."""
    out = torch.empty_like(base)
    n = piece.shape[0]
    if r0 > 0:
        out[:r0].copy_(base[:r0])
    out[r0:r0 + n].copy_(piece)
    if r0 + n < base.shape[0]:
        out[r0 + n:].copy_(base[r0 + n:])
    return out


def _shift(buf, new, dim):
    """Out-of-place sliding-window shift: drop the first new.size(dim) rows of
    buf along dim, append new."""
    c = new.shape[dim]
    W = buf.shape[dim]
    out = torch.empty_like(buf)
    out.narrow(dim, 0, W - c).copy_(buf.narrow(dim, c, W - c))
    out.narrow(dim, W - c, c).copy_(new)
    return out


class _AttendCache(torch.autograd.Function):
    """In-place cache attention (the zero-copy path).

    forward : Kbuf[row0:r1] <- slabK (the layer's PLAIN time-major cache buffer,
              not autograd-tracked), o = flash_lower_right(q, Kbuf[:r1]).
    backward: restore Kbuf[row0:r1] <- slabK (rows < row0 are the committed
              prefix, final since before this step; rows >= row0 belong to later
              steps whose backward has ALREADY run -- autograd processes a
              layer's steps strictly in reverse time because step t+1's slab is
              a function of step t's o), run flash backward on the view, then
                dKbuf[:row0] += dk[:row0]           (grad for the committed prefix,
                                                     owed to the earlier step that
                                                     wrote those rows)
                dslab[:commit] += dKbuf[row0:row0+commit]   (this step commits chunk
                                                     c0: collect what later steps owed it)
              The accumulation is reverse-time in the cache dtype, i.e. the same
              order/dtype autograd uses through the torch.cat chain of
              engine1_layer_split / the splice path -> identical numerics.
    dKbuf/dVbuf are zeroed once per forward call in Wave3D._pipeline."""

    @staticmethod
    def forward(ctx, q, slabK, slabV, Kbuf, Vbuf, dKbuf, dVbuf, row0, r1, scale, commit):
        Kbuf[row0:r1].copy_(slabK)
        Vbuf[row0:r1].copy_(slabV)
        k = Kbuf[:r1].permute(1, 2, 0, 3)
        v = Vbuf[:r1].permute(1, 2, 0, 3)
        out, lse, cq, ck, mq, mk, seed, off, _ = \
            torch.ops.aten._scaled_dot_product_flash_attention(
                q, k, v, 0.0, True, False, scale=scale)
        ctx.save_for_backward(q, slabK, slabV, out, lse, cq, ck, seed, off)
        ctx.bufs = (Kbuf, Vbuf, dKbuf, dVbuf)
        ctx.meta = (row0, r1, scale, commit, mq, mk)
        return out

    @staticmethod
    def backward(ctx, dout):
        q, slabK, slabV, out, lse, cq, ck, seed, off = ctx.saved_tensors
        Kbuf, Vbuf, dKbuf, dVbuf = ctx.bufs
        row0, r1, scale, commit, mq, mk = ctx.meta
        Kbuf[row0:r1].copy_(slabK)
        Vbuf[row0:r1].copy_(slabV)
        k = Kbuf[:r1].permute(1, 2, 0, 3)
        v = Vbuf[:r1].permute(1, 2, 0, 3)
        dq, dk, dv = torch.ops.aten._scaled_dot_product_flash_attention_backward(
            dout.contiguous(), q, k, v, out, lse, cq, ck, mq, mk, 0.0, True, seed, off,
            scale=scale)
        dk = dk.permute(2, 0, 1, 3)                   # [r1,B,h,dh] time-major
        dv = dv.permute(2, 0, 1, 3)
        if row0 > 0:
            dKbuf[:row0] += dk[:row0]
            dVbuf[:row0] += dv[:row0]
        dsK, dsV = dk[row0:r1], dv[row0:r1]
        if commit > 0:
            dsK[:commit] += dKbuf[row0:row0 + commit]
            dsV[:commit] += dVbuf[row0:row0 + commit]
        return dq, dsK, dsV, None, None, None, None, None, None, None, None


def _flash_lr(q, k, v, scale):
    """Lower-right causal attention: query i (of Lq) sees keys 0..i+(Lk-Lq).
    The aten flash kernel's is_causal IS lower-right aligned for Lq<Lk (FA2
    semantics; verified vs math to bf16 noise)."""
    return torch.ops.aten._scaled_dot_product_flash_attention(
        q, k, v, is_causal=True, scale=scale)[0]


def _math_lr(q, k, v, scale):
    Lq, Lk = q.shape[2], k.shape[2]
    s = (q.float() @ k.float().transpose(-1, -2)) * scale
    i = torch.arange(Lq, device=q.device)[:, None]
    j = torch.arange(Lk, device=q.device)[None, :]
    s = s.masked_fill(j > i + (Lk - Lq), float("-inf"))
    return (torch.softmax(s, dim=-1).to(v.dtype) @ v)


def make_engine(model, B, T, engine="x", **kw):
    """engine='vmap': this file's Wave3D (functional_call/vmap bodies, shifting
    windows); 'x': engines/wave3d/wave3d_x.Wave3DX (explicit bmm, copy-free caches,
    exact windows, SAC backward). Same forward/forward_logits/stats API."""
    if engine == "x":
        from engines.wave3d.wave3d_x import Wave3DX
        return Wave3DX(model, B, T, **kw)
    return Wave3D(model, B, T, **kw)


class Wave3D:
    def __init__(self, model, B, T, C=C_DEFAULT, seg_steps=None, use_ckpt=True,
                 attn_impl="flash", n_snapshots=32, math="vmap"):
        """seg_steps: global steps per checkpoint segment (None/0 => auto:
        ceil(n_steps/n_snapshots), i.e. ~n_snapshots state snapshots per K).
        math: 'vmap' (functional_call per layer, module code reused) or 'explicit'
        (engines/wave3d/wave3d_ops.py stacked-weight bmm + written-out refine chain;
        same tensors in/out, same layout -- the isolated GEMM-path A/B)."""
        self.model = model
        self.math = math
        if math == "explicit":
            X.check_cfg(model.cfg)
            cfg = model.cfg
            self._xargs = dict(dcut=getattr(cfg, "kmod_vraw_heads", 0) * cfg.head_dim,
                               tau=float(getattr(cfg, "v_clamp_tau", 0.0) or 0.0),
                               lam=float(getattr(cfg, "mult_res_lambda", 1.0)))
        self.B, self.T, self.C = B, T, C
        self.n = T // C
        self.L = len(model.blocks)
        self.body = _Body(model.blocks[0])
        attn = model.blocks[0].attn
        self.h, self.dh, self.d = attn.n_heads, attn.head_dim, model.cfg.d_model
        self.scale = attn.scale
        self.seg_steps = seg_steps or 0
        self.n_snapshots = n_snapshots
        self.use_ckpt = use_ckpt
        # "flash": zero-copy in-place cache (_AttendCache); "flash_splice": autograd
        # cache tensors + out-of-place splice (same flash kernel; plumbing cross-check);
        # "math": splice path with fp32 dense attention (fp32 validation). fp32 runs
        # with impl "flash" fall back to "math" (flash has no fp32 kernel).
        self.attn_impl = attn_impl
        self._pnames = [n for n, _ in model.blocks[0].named_parameters()]
        assert not list(model.blocks[0].named_buffers()), "Block buffers unsupported"
        self._last_keys = None
        self.stats = {}
        self._fns = {m: self._make_fn(m) for m in ("prep", "refine", "tail")}
        self._compiled = {}
        self._ar = {}

    # ------------------------------------------------------------ schedule
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

    # ------------------------------------------------------------ params
    def _stack_params(self):
        blocks = list(self.model.blocks)
        per = [dict(b.named_parameters()) for b in blocks]
        cast = torch.is_autocast_enabled("cuda")
        out = {}
        for n in self._pnames:
            s = torch.stack([p[n] for p in per])
            if cast and s.dim() >= 2:
                s = s.to(torch.get_autocast_dtype("cuda"))
            out["blk." + n] = s
        return out

    def _arange(self, K, dev):
        key = (K, str(dev))
        if key not in self._ar:
            self._ar[key] = torch.arange(-K * self.C, self.T + K * self.C, device=dev)
        return self._ar[key]

    def _make_fn(self, mode):
        body = self.body

        def fn(p, *args):
            f = lambda pp, *a: functional_call(body, pp, (mode, *a))
            return vmap(f, in_dims=(0,) + (0,) * len(args))(p, *args)
        return fn

    def set_compile(self, mode="refine"):
        """torch.compile the vmapped per-layer bodies. mode: 'none' | 'refine'
        | 'all' (prep+refine+tail). Static shapes: recompiles per A (<= A_max graphs)."""
        modes = {"none": (), "refine": ("refine",), "all": ("prep", "refine", "tail")}[mode]
        import torch._dynamo
        for attr in ("cache_size_limit", "recompile_limit"):
            if hasattr(torch._dynamo.config, attr):
                setattr(torch._dynamo.config, attr, 256)
        # dynamic=True dies inside vmap's matmul batching rule ("Cannot call numel()
        # on tensor with symbolic sizes"); static shapes => one graph per (A, dtype).
        if self.math == "explicit":
            self._compiled = {m: torch.compile(self._explicit, dynamic=False) for m in modes}
        else:
            self._compiled = {m: torch.compile(self._fns[m], dynamic=False) for m in modes}

    def _run(self, mode, params, i0, i1, *args):
        p = {k: v[i0:i1] for k, v in params.items()}
        if self.math == "explicit":
            fn = self._compiled.get(mode) or self._explicit
            return fn(mode, p, *args)
        fn = self._compiled.get(mode) or self._fns[mode]
        return fn(p, *args)

    def _explicit(self, mode, p, *args):
        """wave3d_ops bodies on the vmap engine's batch-major [A, B, rows, ...]
        tensors (same signatures/outputs as _Body via vmap)."""
        P = {k[4:]: v for k, v in p.items()}                 # strip 'blk.'
        h, dh = self.h, self.dh
        if mode == "prep":
            xc, cos, sin = args                              # [Ap,B,C,d] fp32, [Ap,C,dh/2]
            A, B, C, d = xc.shape
            cs, sn = cos[:, None, :, None, :], sin[:, None, :, None, :]
            xhat, xb, kf, vf, q_r, k_r = X.prep(P, xc.reshape(A, B * C, d), cs, sn, h, dh, (B, C))
            v = vf.view(A, B, C, h, dh).transpose(2, 3)
            return (xhat.view(A, B, C, d), kf.view(A, B, C, d), vf.view(A, B, C, d),
                    q_r.transpose(2, 3), k_r.transpose(2, 3), v)
        if mode == "refine":
            wx, o_m, wk, wv, cos, sin = args                 # [A,B,W,d] x4, [A,W,dh/2] x2
            A, B, W, d = o_m.shape
            cs, sn = cos[:, None, :, None, :], sin[:, None, :, None, :]
            Kr, Vr = X.refine(P, wx.reshape(A, B * W, d), o_m.reshape(A, B * W, d),
                              wk.reshape(A, B * W, d), wv.reshape(A, B * W, d), cs, sn,
                              h, dh, rows_dims=(B, W), **self._xargs)
            return Kr.transpose(2, 3), Vr.transpose(2, 3)   # [A,B,h,W,dh]
        if mode == "tail":
            xc, oc = args                                    # [Ac,B,C,d] fp32, [Ac,B,C,d]
            A, B, C, d = xc.shape
            return X.tail(P, xc.reshape(A, B * C, d), oc.reshape(A, B * C, d)).view(A, B, C, d)
        raise ValueError(mode)

    # ------------------------------------------------------------ attention
    def _attend(self, q, k, v):
        if self._impl == "flash_splice":
            return _flash_lr(q, k, v, self.scale)
        return _math_lr(q, k, v, self.scale)

    # ------------------------------------------------------------ one step
    def _step(self, st, t, K, params):
        n, C, B, L, T = self.n, self.C, self.B, self.L, self.T
        h, dh, d = self.h, self.dh, self.d
        W = K * C
        la, lb = self._active(t, K)
        layers = list(range(la, lb + 1))
        tls = [t - l * K for l in layers]
        A = len(layers)
        new = dict(st)
        cos_all, sin_all = self.model.rope_cos, self.model.rope_sin
        dev = cos_all.device
        ar = self._arange(K, dev)                       # arange(-K*C, T+K*C): sync-free positions
        Z = K * C                                       # ar[Z + p] == p
        buf_mode = self._impl == "flash"

        # ---- prep: layers whose new chunk exists (t_l <= n-1) = upper subrange
        prep = [(a, l, tl) for a, (l, tl) in enumerate(zip(layers, tls)) if tl <= n - 1]
        prep_out = None
        if prep:
            a0 = prep[0][0]
            xc = torch.stack([new[f"x{l}.{tl}"] for _, l, tl in prep])
            pos = torch.stack([ar[Z + tl * C:Z + (tl + 1) * C] for _, _, tl in prep])
            prep_out = list(self._run("prep", params, la + a0, lb + 1,
                                      xc, cos_all[pos], sin_all[pos]))
            # (xhat, k_flat, v_flat, q_r, k_r, v) each [Ap, B, ...]; k_r/v -> time-major
            prep_out[4] = prep_out[4].permute(0, 3, 1, 2, 4).contiguous()   # [Ap,C,B,h,dh]
            prep_out[5] = prep_out[5].permute(0, 3, 1, 2, 4).contiguous()
            self._spec = [(o.dtype) for o in prep_out]
        spec = self._spec

        wx_l, wk_l, wv_l, wq_l = [], [], [], []
        o_full = []
        for a, (l, tl) in enumerate(zip(layers, tls)):
            if tl <= n - 1:
                ap = a - prep[0][0]
                nx, nk, nv, nq, kr, vr = (o[ap] for o in prep_out)
            else:
                nx = torch.zeros(B, C, d, device=dev, dtype=spec[0])
                nk = torch.zeros(B, C, d, device=dev, dtype=spec[1])
                nv = torch.zeros(B, C, d, device=dev, dtype=spec[2])
                nq = torch.zeros(B, h, C, dh, device=dev, dtype=spec[3])
                kr = torch.zeros(C, B, h, dh, device=dev, dtype=spec[4])
                vr = torch.zeros(C, B, h, dh, device=dev, dtype=spec[5])
            if tl == 0:                                   # layer l starts
                wx = torch.zeros(B, W, d, device=dev, dtype=spec[0])
                wk = torch.zeros(B, W, d, device=dev, dtype=spec[1])
                wv = torch.zeros(B, W, d, device=dev, dtype=spec[2])
                wq = torch.zeros(B, h, W, dh, device=dev, dtype=spec[3])
                Kw = torch.zeros(W, B, h, dh, device=dev, dtype=spec[4])   # time-major
                Vw = torch.zeros(W, B, h, dh, device=dev, dtype=spec[5])
                if buf_mode:
                    Kc = Vc = None
                else:
                    Kc = torch.zeros(T, B, h, dh, device=dev, dtype=spec[4])   # time-major cache
                    Vc = torch.zeros(T, B, h, dh, device=dev, dtype=spec[5])
            else:
                wx, wk, wv, wq = (new[f"w{s}{l}"] for s in ("x", "k", "v", "q"))
                Kw, Vw = new[f"Kw{l}"], new[f"Vw{l}"]
                Kc, Vc = (None, None) if buf_mode else (new[f"Kc{l}"], new[f"Vc{l}"])
            # slide the window one slot
            wx, wk, wv = _shift(wx, nx, 1), _shift(wk, nk, 1), _shift(wv, nv, 1)
            wq = _shift(wq, nq, 2)
            # slab write: (window refined at t_l-1 = chunks t_l-K..t_l-1, raw new
            # chunk t_l) -> cache rows of chunks [c0,c1] (clipped to the sequence)
            c0, c1 = max(0, tl - K), min(n - 1, tl)
            r0, r1 = (c0 - (tl - K)) * C, (c1 - (tl - K) + 1) * C
            # time-major: cat along dim 0 == contiguous memcpy of 3 blocks (the
            # [B,h,T,dh] inner-dim cat was a 0.5TB/s strided kernel = 46% of step GPU time)
            slabK = torch.cat([Kw, kr], 0)[r0:r1]
            slabV = torch.cat([Vw, vr], 0)[r0:r1]
            # attention over [committed prefix ++ window], lower-right causal.
            lo, hi = max(0, tl - K + 1), min(n - 1, tl)
            s0, s1 = lo - (tl - K + 1), hi - (tl - K + 1)
            qv = wq[:, :, s0 * C:(s1 + 1) * C]
            if buf_mode:
                # chunk c0 is committed by this write iff no later step rewrites it
                commit = C if tl >= K else 0
                ov = _AttendCache.apply(qv, slabK, slabV, self._Kbuf[l], self._Vbuf[l],
                                        self._dKbuf[l], self._dVbuf[l], c0 * C, (hi + 1) * C,
                                        self.scale, commit)
            else:
                Kc = _splice(Kc, c0 * C, slabK)
                Vc = _splice(Vc, c0 * C, slabV)
                # flash takes the permuted [B,h,L,dh] view (last dim contiguous), no copy
                kv = Kc[:(hi + 1) * C].permute(1, 2, 0, 3)
                vv = Vc[:(hi + 1) * C].permute(1, 2, 0, 3)
                ov = self._attend(qv, kv, vv)
            pads = []
            if s0 > 0:
                pads.append(torch.zeros(B, h, s0 * C, dh, device=dev, dtype=ov.dtype))
            pads.append(ov)
            if s1 < K - 1:
                pads.append(torch.zeros(B, h, (K - 1 - s1) * C, dh, device=dev, dtype=ov.dtype))
            o_full.append(torch.cat(pads, 2) if len(pads) > 1 else ov)
            new[f"wx{l}"], new[f"wk{l}"], new[f"wv{l}"], new[f"wq{l}"] = wx, wk, wv, wq
            if not buf_mode:
                new[f"Kc{l}"], new[f"Vc{l}"] = Kc, Vc
            wx_l.append(wx); wk_l.append(wk); wv_l.append(wv)

        o_st = torch.stack(o_full)                                    # [A,B,h,W,dh]
        o_m = o_st.transpose(2, 3).reshape(A, B, W, h * dh)           # merge

        # ---- tail: layers with t_l >= K-1 (slot 0 final) = lower subrange
        tail = [(a, l, tl) for a, (l, tl) in enumerate(zip(layers, tls)) if tl >= K - 1]
        if tail:
            a1 = tail[-1][0]
            oc = o_m[:a1 + 1, :, :C]
            xc = torch.stack([new[f"x{l}.{tl - K + 1}"] for _, l, tl in tail])
            xn = self._run("tail", params, la, la + a1 + 1, xc, oc)     # [Ac,B,C,d]
            for a, l, tl in tail:
                new[f"x{l + 1}.{tl - K + 1}"] = xn[a]                    # handoff to layer l+1

        # ---- refine the whole window of every active layer
        base = torch.stack([ar[Z + (tl - K + 1) * C:Z + (tl - K + 1) * C + W] for tl in tls])  # [A,W] abs pos
        valid = (base >= 0) & (base < T)
        posw = base.clamp(0, T - 1)
        Kw, Vw = self._run("refine", params, la, lb + 1,
                           torch.stack(wx_l), o_m, torch.stack(wk_l), torch.stack(wv_l),
                           cos_all[posw], sin_all[posw])              # [A,B,h,W,dh]
        vm = valid[:, :, None, None, None]                          # [A,W,1,1,1]
        Kw = (Kw.permute(0, 3, 1, 2, 4) * vm.to(Kw.dtype)).contiguous()   # [A,W,B,h,dh] time-major
        Vw = (Vw.permute(0, 3, 1, 2, 4) * vm.to(Vw.dtype)).contiguous()
        for a, (l, tl) in enumerate(zip(layers, tls)):
            if tl == n + K - 2:                                       # layer l finished
                for s in ("Kc", "Vc", "wx", "wk", "wv", "wq", "Kw", "Vw"):
                    new.pop(f"{s}{l}", None)
                for c in range(n):
                    new.pop(f"x{l}.{c}", None)
            else:
                new[f"Kw{l}"], new[f"Vw{l}"] = Kw[a], Vw[a]
        self._rows += A * B * W
        return new

    def _segment(self, t0, t1, K, keys, params, *vals):
        st = dict(zip(keys, vals))
        for t in range(t0, t1):
            st = self._step(st, t, K, params)
        keys_out = sorted(st.keys())
        self._last_keys = keys_out
        return tuple(st[k] for k in keys_out)

    # ------------------------------------------------------------ API
    def _pipeline(self, idx, K, seg_steps=None):
        """Run the 3D wavefront; returns the final residual stream x [B,T,d]."""
        model = self.model
        B, T = idx.shape
        assert B == self.B and T == self.T, (idx.shape, self.B, self.T)
        steps = self.n_steps(K)
        # auto: ~n_snapshots state snapshots; K>32 (W=4096-row windows) doubles them
        # so per-segment activations halve (K64 B4/T4096 eager peaked 273GB at 32)
        ns = self.n_snapshots * (2 if K > 32 else 1)
        S = seg_steps or self.seg_steps or max(1, -(-steps // ns))
        self._seg_used = S
        params = self._stack_params()
        impl = self.attn_impl
        ac = torch.is_autocast_enabled("cuda")
        if impl.startswith("flash") and not ac and model.tok_emb.weight.dtype == torch.float32:
            impl = "math"                               # no fp32 flash kernel
        self._impl = impl
        if impl == "flash":
            dt = torch.get_autocast_dtype("cuda") if ac else model.tok_emb.weight.dtype
            shp = (self.T, self.B, self.h, self.dh)
            dev = idx.device
            self._Kbuf = [torch.zeros(shp, device=dev, dtype=dt) for _ in range(self.L)]
            self._Vbuf = [torch.zeros(shp, device=dev, dtype=dt) for _ in range(self.L)]
            self._dKbuf = [torch.zeros(shp, device=dev, dtype=dt) for _ in range(self.L)]
            self._dVbuf = [torch.zeros(shp, device=dev, dtype=dt) for _ in range(self.L)]
        x0 = model.tok_emb(idx)
        st = {f"x0.{c}": x0[:, c * self.C:(c + 1) * self.C] for c in range(self.n)}
        self._rows = 0
        for t0 in range(0, steps, S):
            t1 = min(steps, t0 + S)
            keys = sorted(st.keys())
            vals = tuple(st[k] for k in keys)
            if self.use_ckpt:
                out = checkpoint(self._segment, t0, t1, K, keys, params, *vals,
                                 use_reentrant=False, preserve_rng_state=False)
            else:
                out = self._segment(t0, t1, K, keys, params, *vals)
            st = dict(zip(self._last_keys, out))
        assert len(st) == self.n, sorted(st.keys())
        self.stats = dict(steps=steps, rows_per_step=self._rows / steps,
                          max_active=self.max_active(K), seg_steps=S)
        return torch.cat([st[f"x{self.L}.{c}"] for c in range(self.n)], 1)

    def forward_logits(self, idx, K, seg_steps=None):
        x = self._pipeline(idx, K, seg_steps)
        return self.model.lm_head(self.model.norm_f(x))

    def forward(self, idx, targets, K, seg_steps=None):
        """CE loss (fp32 logits, ignore_index=-1), identical head to DualModLM.forward."""
        logits = self.forward_logits(idx, K, seg_steps)
        return F.cross_entropy(logits.float().view(-1, logits.shape[-1]),
                               targets.reshape(-1), ignore_index=-1)
