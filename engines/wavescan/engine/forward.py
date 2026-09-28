"""Layer-major chunked forward (Phase A math; megakernel.md §6).

Everything here is expressible inside a CUDA-graph capture: static shapes per
schedule, writes into preallocated cache buffers, no data-dependent control
flow. The same functions serve eager (uncaptured) execution for tests.
"""

import torch

from ..cells.scan_cell_fwd import chunk_cell_forward, make_intra_mask


class MaskCache:
    """Per-chunk-size additive intra masks, built once per device."""

    def __init__(self, device):
        self.device = device
        self._m = {}

    def get(self, c: int):
        if c not in self._m:
            self._m[c] = make_intra_mask(c, self.device)
        return self._m[c]


def layer_forward(attn, xhat, cos, sin, schedule, K_cache, V_cache, masks,
                  o_out=None):
    """Run one layer's scan over all chunks of `schedule`.

    K_cache/V_cache: preallocated [B, h, T_max, d_h]; rows [0, T) are written
    append-only in schedule order. Returns o_cat [B, T, d] (pre-W_O).
    """
    B, T, _ = xhat.shape
    if o_out is None:
        o_out = xhat.new_empty(B, T, attn.n_heads * attn.head_dim)
    for pos0, c, n_sweeps in schedule:
        o_cat, K_rows, V_rows = chunk_cell_forward(
            attn, xhat[:, pos0:pos0 + c], cos[pos0:pos0 + c],
            sin[pos0:pos0 + c], K_cache[:, :, :pos0], V_cache[:, :, :pos0],
            n_sweeps, intra_mask=masks.get(c))
        K_cache[:, :, pos0:pos0 + c] = K_rows.to(K_cache.dtype)
        V_cache[:, :, pos0:pos0 + c] = V_rows.to(V_cache.dtype)
        o_out[:, pos0:pos0 + c] = o_cat
    return o_out


def model_forward(model, idx, schedule, K_caches, V_caches, masks,
                  xl_saved=None, ocat_saved=None):
    """Full-model chunked forward. K_caches/V_caches: [L] list of cache
    buffers. xl_saved: optional [L+1] buffers recording per-layer inputs
    x^0..x^L (x^L feeds norm_f) for the backward (megakernel.md §3).
    Returns logits [B, T, V]."""
    x = model.tok_emb(idx)
    T = x.shape[1]
    cos, sin = model.rope_cos[:T], model.rope_sin[:T]
    for li, blk in enumerate(model.blocks):
        if xl_saved is not None:
            xl_saved[li].copy_(x)
        xhat = blk.attn_norm(x)
        o_cat = layer_forward(blk.attn, xhat, cos, sin, schedule,
                              K_caches[li], V_caches[li], masks)
        if ocat_saved is not None:
            ocat_saved[li].copy_(o_cat)
        x = x + blk.attn.wo(o_cat)
        x = x + blk.mlp(blk.mlp_norm(x))
    if xl_saved is not None:
        xl_saved[len(model.blocks)].copy_(x)
    return model.lm_head(model.norm_f(x))
