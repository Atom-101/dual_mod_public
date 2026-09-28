"""RoPE helpers.

Convention (plan.md §3): queries are rotated at read time with R_j; the *blended*
key k''_j is rotated once, at cache-write time, with R_j. The cache therefore
holds pre-rotated keys and attention scores are plain dot products.

Pairing: LLaMA-style half-split (rotate_half).
"""

import torch


def build_rope_cache(max_seq_len: int, head_dim: int, theta: float,
                     device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (cos, sin), each [max_seq_len, head_dim//2], fp32."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # [T, head_dim//2]
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate x by position.

    x:   [..., T, head_dim] (per-head layout; T may be 1 for a single step)
    cos/sin: [T, head_dim//2] for those positions (fp32).
    Math in fp32, result cast back to x.dtype.
    """
    xf = x.float()
    x1, x2 = xf.chunk(2, dim=-1)
    # broadcast cos/sin over leading dims
    out = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    return out.to(x.dtype)
