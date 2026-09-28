"""Exact sequential scan (plan.md §5) with chunked gradient checkpointing.

Layer-major: all raw projections for the T positions are computed in one batched
matmul; only the j-axis is sequential. Each step is batch/head-parallel and uses
the same `attend_one` / `refine_kv` code path as incremental decode.

Checkpointing exploits the append-only cache: the recurrent state at chunk start
`a` is exactly the first `a` rows of the final cache tensors, so each chunk can
be recomputed in backward from (chunk inputs, K_past, V_past) alone. This bounds
live activation memory to O(chunk) while keeping the autograd graph exact.
"""

import torch
from torch.utils.checkpoint import checkpoint

from .rope import apply_rope


def _scan_chunk(attn, xhat_c, q_r_c, k_c, k_self_r_c, v_c, cos_c, sin_c,
                K_past, V_past, collect=None):
    """Run §3 steps 3-10 for the positions of one chunk.

    xhat_c: [B, Tc, d]; q_r_c/k_c/k_self_r_c/v_c: [B, h, Tc, d_h] (k_c unrotated);
    cos_c/sin_c: [Tc, d_h//2]; K_past/V_past: [B, h, P, d_h] refined cache so far.
    Returns (o_chunk [B,h,Tc,d_h], K_new [B,h,Tc,d_h], V_new [B,h,Tc,d_h]).
    """
    Tc = xhat_c.shape[1]
    new_K, new_V, o_list = [], [], []
    for j in range(Tc):
        K_cur = torch.cat([K_past] + new_K, dim=2) if new_K else K_past
        V_cur = torch.cat([V_past] + new_V, dim=2) if new_V else V_past
        o_j, probs = attn.attend_one(
            q_r_c[:, :, j:j + 1], k_self_r_c[:, :, j:j + 1], v_c[:, :, j:j + 1],
            K_cur, V_cur, need_probs=collect is not None, collect=collect)
        o_cat = attn._merge(o_j)                                   # [B, 1, d]

        kpp, vpp = attn.refine_kv(
            xhat_c[:, j:j + 1], o_cat,
            attn._merge(k_c[:, :, j:j + 1]), attn._merge(v_c[:, :, j:j + 1]),
            collect=collect)
        kpp_r = apply_rope(attn._split(kpp), cos_c[j:j + 1], sin_c[j:j + 1])
        new_K.append(kpp_r)
        new_V.append(attn._split(vpp))
        o_list.append(o_j)

        if collect is not None and probs is not None:
            # incoming attention mass per key position (§10.4), head-averaged
            n = probs.shape[-1]
            collect["inc_mass"][:, :n] += probs.mean(dim=1)[:, 0, :]
            # hop mass (dilution probe): mass this position puts on its PREDECESSOR key (n-2) and on itself (n-1),
            # head-averaged; appended per position -> [B, T] after the scan.
            pm = probs.mean(dim=1)[:, 0, :]                                    # [B, n]
            collect.setdefault("hop_mass", []).append(pm[:, -2] if n >= 2 else torch.zeros_like(pm[:, -1]))
            collect.setdefault("self_mass", []).append(pm[:, -1])
            collect.setdefault("max_mass", []).append(pm.amax(dim=-1))

    return torch.cat(o_list, dim=2), torch.cat(new_K, dim=2), torch.cat(new_V, dim=2)


def sequential_scan(attn, xhat, cos, sin, collect=None):
    """Full-layer exact scan. Returns o_cat [B, T, d] (pre-W_O)."""
    cfg = attn.cfg
    B, T, _ = xhat.shape

    # (i) parallel precompute of all raw projections (§5)
    q_r = apply_rope(attn._split(attn.wq(xhat)), cos, sin)
    k = attn._split(attn.wk(xhat))                       # unrotated (blend space)
    k_self_r = apply_rope(k, cos, sin)
    v = attn._split(attn.project_v(xhat))

    if collect is not None:
        collect["inc_mass"] = torch.zeros(B, T, device=xhat.device)

    C = cfg.checkpoint_chunk if cfg.checkpoint_chunk and cfg.checkpoint_chunk > 0 else T
    use_ckpt = (cfg.checkpoint_chunk and cfg.checkpoint_chunk > 0 and T > C
                and torch.is_grad_enabled() and attn.training and collect is None)

    K_past = q_r.new_zeros(B, attn.n_heads, 0, attn.head_dim)
    V_past = v.new_zeros(B, attn.n_heads, 0, attn.head_dim)
    o_chunks = []
    for a in range(0, T, C):
        b = min(a + C, T)
        args = (attn, xhat[:, a:b], q_r[:, :, a:b], k[:, :, a:b],
                k_self_r[:, :, a:b], v[:, :, a:b], cos[a:b], sin[a:b],
                K_past, V_past)
        if use_ckpt:
            o_c, K_new, V_new = checkpoint(_scan_chunk, *args, use_reentrant=False)
        else:
            o_c, K_new, V_new = _scan_chunk(*args, collect=collect)
        K_past = torch.cat([K_past, K_new], dim=2)
        V_past = torch.cat([V_past, V_new], dim=2)
        o_chunks.append(o_c)

    o = torch.cat(o_chunks, dim=2)                       # [B, h, T, d_h]
    return attn._merge(o)
