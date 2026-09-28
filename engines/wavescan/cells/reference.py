"""Eager oracle wrappers (megakernel.md §14 cells/reference.py).

The repo's per-position step (`attend_one` + `refine_kv`, driven by
`_scan_chunk`) IS the bit-level oracle for everything in wavescan. These
helpers expose it in the shapes the kernel tests want. Nothing here is fast.
"""

import torch

from models.dualmod.rope import apply_rope
from models.dualmod.scan import _scan_chunk


@torch.no_grad()
def eager_layer(attn, xhat, cos, sin):
    """Exact scan for one layer. Returns (o [B,h,T,d_h], K_rows, V_rows) —
    same contract as the wavescan layer forward, straight from the repo path."""
    q_r = apply_rope(attn._split(attn.wq(xhat)), cos, sin)
    k = attn._split(attn.wk(xhat))
    k_self_r = apply_rope(k, cos, sin)
    v = attn._split(attn.project_v(xhat))
    B = xhat.shape[0]
    Kp = q_r.new_zeros(B, attn.n_heads, 0, attn.head_dim)
    Vp = v.new_zeros(B, attn.n_heads, 0, attn.head_dim)
    return _scan_chunk(attn, xhat, q_r, k, k_self_r, v, cos, sin, Kp, Vp)


@torch.no_grad()
def eager_cell(attn, xhat_c, cos_c, sin_c, K_comm, V_comm):
    """Exact per-position steps for one chunk given a committed prefix —
    the oracle for a single cell (T-K1)."""
    q_r = apply_rope(attn._split(attn.wq(xhat_c)), cos_c, sin_c)
    k = attn._split(attn.wk(xhat_c))
    k_self_r = apply_rope(k, cos_c, sin_c)
    v = attn._split(attn.project_v(xhat_c))
    return _scan_chunk(attn, xhat_c, q_r, k, k_self_r, v, cos_c, sin_c,
                       K_comm, V_comm)
