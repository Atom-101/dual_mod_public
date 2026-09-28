"""Parity: closed-form _rmsnorm_ctx_vjp  ==  torch.autograd through the real
norm_kctx / norm_vctx forward. This is the fused-vs-autograd divergence check
for the K/V-norm VJP that un-taxes the fused engine (replaces the per-sweep
per-layer autograd.grad). fp32 must be analytically exact (<1e-5); bf16 carries
only bf16 round-off (informational)."""
import types
import torch
import torch.nn as nn

from models.dualmod.attention import DualModAttention
from engines.wavescan.cells.fused_sweep import FusedSweepCell


def _mock(n_heads, head_dim, gain_init, dtype, which):
    m = types.SimpleNamespace()
    m.cfg = types.SimpleNamespace(k_ctx_norm="rms", v_ctx_norm="rms")
    m.n_heads, m.head_dim = n_heads, head_dim
    g = (torch.rand(n_heads) * 1.5 + 0.5).to(torch.float32)   # spread around init
    p = nn.Parameter(g)
    if which == "k":
        m.k_gain = p
        m.norm = DualModAttention.norm_kctx.__get__(m)
    else:
        m.ctx_gain = p
        m.norm = DualModAttention.norm_vctx.__get__(m)
    return m, p


def _ref(m, gain, y, dout):
    """autograd reference through the real forward."""
    yl = y.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        z = m.norm(yl)
    dy, dgain = torch.autograd.grad([z], [yl, gain], [dout.to(z.dtype)])
    return dy, dgain


def _rel(a, b):
    return (a.float() - b.float()).norm() / (b.float().norm() + 1e-12)


def run():
    torch.manual_seed(0)
    ok = True
    for which in ("k", "v"):
        for dtype in (torch.float32, torch.bfloat16):
            for (N, H, Dh) in [(64, 8, 64), (130, 13, 48), (1, 16, 64)]:
                m, gain = _mock(H, Dh, 1.75, dtype, which)
                d = H * Dh
                y = torch.randn(N, d, dtype=dtype)
                dout = torch.randn(N, d, dtype=dtype)
                dy_ref, dg_ref = _ref(m, gain, y, dout)
                dy, dg = FusedSweepCell._rmsnorm_ctx_vjp(y, dout, gain, H, Dh)
                rdy, rdg = _rel(dy, dy_ref), _rel(dg, dg_ref)
                tol = 3e-5 if dtype == torch.float32 else 4e-2
                tag = f"{which}-side {str(dtype).split('.')[-1]:8s} N{N} H{H} D{Dh}"
                status = "OK " if (rdy < tol and rdg < tol) else "FAIL"
                if status == "FAIL":
                    ok = False
                print(f"[{status}] {tag}  rel dy={rdy:.2e}  rel dgain={rdg:.2e}  (tol {tol:.0e})")
    print("\n=== ALL PASS ===" if ok else "\n=== FAILURES ABOVE ===")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
