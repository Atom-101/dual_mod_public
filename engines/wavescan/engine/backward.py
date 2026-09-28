"""Reverse-scan backward (megakernel.md §5), layer-major (Phase A §6).

Strategy: chunks are re-run in reverse order; each backward cell RECOMPUTES its
forward (same cell code, same sweep count — the schedule is saved metadata)
under local autograd, then one `torch.autograd.grad` call performs the whole
within-cell chain rule (attention backward, gate σ′, ctx GEMM grads, RMSNorm′,
W_O/MLP backward). Gradients w.r.t. cache rows flow through two channels:

  - the cell's own written rows K″/V″[pos0:pos0+c]: their grads are read from
    the fp32 grad-cache (accumulated there by all LATER chunks — the only
    readers, append-only order guarantees this is complete by the time the
    chunk runs in reverse), and fed as grad_outputs of the recomputed rows;
  - the committed prefix the cell READ: autograd.grad returns dense
    d(prefix) [B,h,pos0,d_h], accumulated into grad-cache[:, :, :pos0].

Everything is torch ops on static shapes → CUDA-graph capturable (Phase A
captures the whole reverse scan as one graph). fp32 grad-cache because rows
accumulate hundreds of contributions.
"""

import torch

from .forward import MaskCache
from ..cells.scan_cell_fwd import chunk_cell_forward


class GradBuffers:
    """fp32 gradient accumulators keyed by parameter identity (tied params —
    tok_emb/lm_head — accumulate into one buffer, matching eager autograd).
    All buffers are VIEWS into one flat tensor so DDP is a single flat-bucket
    all-reduce (megakernel.md §6) — one NCCL call, rank-order-independent."""

    def __init__(self, params, bf16_ids=None):
        """bf16_ids: optional set of id(param) whose grad accumulator is stored
        in bf16 (the stable trunk); everything else (the storm-prone DM
        machinery) stays fp32. Two flat buckets (one per dtype) -> two flat NCCL
        buckets. Default (bf16_ids=None) = the original single fp32 buffer."""
        uniq = []
        seen = set()
        for p in params:
            if id(p) not in seen:
                seen.add(id(p))
                uniq.append(p)
        bf16_ids = bf16_ids or set()
        self.buf = {}
        self.order = uniq
        self.flats = []          # (flat_tensor, dtype) buckets, in reduce order
        groups = [("fp32", torch.float32, [p for p in uniq if id(p) not in bf16_ids])]
        if bf16_ids:
            groups.append(("bf16", torch.bfloat16, [p for p in uniq if id(p) in bf16_ids]))
        for _name, dt, plist in groups:
            if not plist:
                continue
            tot = sum(p.numel() for p in plist)
            flat = torch.zeros(tot, dtype=dt, device=uniq[0].device)
            self.flats.append(flat)
            off = 0
            for p in plist:
                self.buf[id(p)] = flat[off:off + p.numel()].view_as(p)
                off += p.numel()
        self.flat = self.flats[0]   # back-compat: fp32 bucket

    def zero_(self):
        for f in self.flats:
            f.zero_()

    def add(self, param, g):
        if g is not None:
            self.buf[id(param)].add_(g.float())

    def get(self, param):
        return self.buf[id(param)]

    def to_param_grads(self):
        for p in self.order:
            buf = self.buf[id(p)]
            if buf.dtype == p.dtype:
                # ALIAS the flat-buffer view (no clone) — avoids a full second
                # copy of the grads (the bf16 trunk bucket is ~17GB at 8B). The
                # optimizer reads p.grad; the engine re-zeros the buffer each step.
                p.grad = buf
            else:
                g = buf.to(p.dtype)            # dtype change already copies
                p.grad = g if p.grad is None else p.grad.copy_(g)


def _grad(outputs, inputs, grad_outputs):
    return torch.autograd.grad(outputs, inputs, grad_outputs,
                               allow_unused=True, retain_graph=False)


def layer_backward(blk, x_l, cos, sin, schedule, K_cache, V_cache, dx_next,
                   masks, grads, dK, dV):
    """Backward for one layer. x_l: saved layer input [B,T,d]; dx_next: grad
    w.r.t. this layer's output x^{l+1} [B,T,d]; dK/dV: zeroed fp32 grad-cache
    [B,h,T,d_h]. Returns dx_l [B,T,d]. Accumulates param grads into `grads`."""
    attn = blk.attn
    params = [p for p in blk.parameters()]
    dx_l = torch.zeros_like(x_l)
    for pos0, c, n_sweeps in reversed(schedule):
        x_c = x_l[:, pos0:pos0 + c].detach().requires_grad_(True)
        K_comm = K_cache[:, :, :pos0].detach().requires_grad_(True)
        V_comm = V_cache[:, :, :pos0].detach().requires_grad_(True)
        with torch.enable_grad():
            xhat_c = blk.attn_norm(x_c)
            o_cat, K_rows, V_rows = chunk_cell_forward(
                attn, xhat_c, cos[pos0:pos0 + c], sin[pos0:pos0 + c],
                K_comm, V_comm, n_sweeps, intra_mask=masks.get(c))
            x_mid = x_c + attn.wo(o_cat)
            x_next = x_mid + blk.mlp(blk.mlp_norm(x_mid))
        outs = [x_next, K_rows, V_rows]
        gouts = [dx_next[:, pos0:pos0 + c],
                 dK[:, :, pos0:pos0 + c].to(K_rows.dtype),
                 dV[:, :, pos0:pos0 + c].to(V_rows.dtype)]
        inputs = [x_c, K_comm, V_comm] + params
        gs = _grad(outs, inputs, gouts)
        dx_l[:, pos0:pos0 + c] = gs[0]
        if pos0 > 0:
            dK[:, :, :pos0] += gs[1].float()
            dV[:, :, :pos0] += gs[2].float()
        for p, g in zip(params, gs[3:]):
            grads.add(p, g)
    return dx_l


def _layers_and_embedding_backward(model, idx, schedule, K_caches, V_caches,
                                   xl_saved, dx, masks, grads, dK, dV):
    L = len(model.blocks)
    T = idx.shape[1]
    cos, sin = model.rope_cos[:T], model.rope_sin[:T]
    for li in range(L - 1, -1, -1):
        dK.zero_()
        dV.zero_()
        dx = layer_backward(model.blocks[li], xl_saved[li], cos, sin,
                            schedule, K_caches[li], V_caches[li], dx,
                            masks, grads, dK, dV)
    # embedding backward via autograd so torch.use_deterministic_algorithms
    # (K8 / spec §5 deterministic mode) picks the deterministic kernel.
    emb_w = model.tok_emb.weight
    with torch.enable_grad():
        emb_leaf = emb_w.detach().requires_grad_(True)
        out = torch.nn.functional.embedding(idx, emb_leaf)
    (g,) = _grad([out], [emb_leaf], [dx.to(out.dtype)])
    grads.add(emb_w, g)


def model_backward(model, idx, schedule, K_caches, V_caches, xl_saved,
                   d_logits, masks, grads, dK, dV, deterministic=False):
    """Full-model reverse from an explicit d(logits) [B,T,V] (test/WaveScanFn
    path — needs the logits-sized grad to exist, so dev-scale only).
    xl_saved: [L+1] saved inputs (x^0..x^L where x^L feeds norm_f). dK/dV: one
    layer's grad-cache, reused across layers (layer-major keeps a single layer
    live — §3). Fills `grads`."""
    x_fin = xl_saved[len(model.blocks)].detach().requires_grad_(True)
    with torch.enable_grad():
        logits = model.lm_head(model.norm_f(x_fin))
    head_params = [model.norm_f.weight, model.lm_head.weight]
    gs = _grad([logits], [x_fin] + head_params, [d_logits])
    dx = gs[0]
    for p, g in zip(head_params, gs[1:]):
        grads.add(p, g)
    _layers_and_embedding_backward(model, idx, schedule, K_caches, V_caches,
                                   xl_saved, dx, masks, grads, dK, dV)


def model_backward_from_targets(model, idx, targets, schedule, K_caches,
                                V_caches, xl_saved, masks, grads, dK, dV,
                                loss_chunk=64):
    """Fused-loss backward: mean-CE over all B*T tokens, head recomputed in
    T-chunks so no [B,T,V] tensor ever materializes (§3 memory plan). The
    matching forward loss is `fused_loss_forward`."""
    B, T = idx.shape
    L = len(model.blocks)
    x_fin_full = xl_saved[L]
    dx = torch.empty_like(x_fin_full)
    inv_n = 1.0 / (B * T)
    head_params = [model.norm_f.weight, model.lm_head.weight]
    for t0 in range(0, T, loss_chunk):
        t1 = min(t0 + loss_chunk, T)
        x_fin = x_fin_full[:, t0:t1].detach().requires_grad_(True)
        with torch.enable_grad():
            logits = model.lm_head(model.norm_f(x_fin))
            loss = torch.nn.functional.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]),
                targets[:, t0:t1].reshape(-1), reduction="sum") * inv_n
        gs = _grad([loss], [x_fin] + head_params, [torch.ones_like(loss)])
        dx[:, t0:t1] = gs[0]
        for p, g in zip(head_params, gs[1:]):
            grads.add(p, g)
    _layers_and_embedding_backward(model, idx, schedule, K_caches, V_caches,
                                   xl_saved, dx, masks, grads, dK, dV)


def fused_loss_forward(model, x_fin, targets, loss_chunk=64):
    """Mean-CE computed from the final pre-norm activations in T-chunks;
    writes nothing logits-sized. Returns scalar loss."""
    B, T, _ = x_fin.shape
    inv_n = 1.0 / (B * T)
    loss = torch.zeros((), device=x_fin.device, dtype=torch.float32)
    for t0 in range(0, T, loss_chunk):
        t1 = min(t0 + loss_chunk, T)
        logits = model.lm_head(model.norm_f(x_fin[:, t0:t1]))
        loss += torch.nn.functional.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            targets[:, t0:t1].reshape(-1), reduction="sum") * inv_n
    return loss
