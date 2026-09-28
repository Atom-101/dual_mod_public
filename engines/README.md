# Training engines

Both engines train the same `DualModLM` and reproduce the exact sequential scan (`models/dualmod/scan.py`)
once the sweep count reaches the chunk length; they differ in how the recurrence is scheduled on the GPU.

| | `wavescan/` (340M runs, H100 and B300) | `wave3d/` (1.3B runs, B300) |
|---|---|---|
| operator | chunk of c tokens, K Jacobi sweeps within the chunk, chunks chained Gauss-Seidel (committed chunks are never re-refined; gradients still flow through every read) | same c=64 / K operator with the *raw-diagonal* convention (each position reads its own raw key/value: exact by nilpotency at K = c) and an optional sliding window |
| parallelism | one layer at a time over the whole sequence; per-cell CUDA graphs; fused Triton sweep (3 launches per sweep) | layer-batched 3D wavefront: layer l runs intra-step t - lK, so all layers sweep at once; per-K CUDA graphs |
| memory | reverse-scan backward with per-chunk forward recompute; fp32 flat gradient buffers (one all-reduce) | time-major write-once caches, zero-copy strided window views, fused refine chain (`wave3d_fused.py`), activation checkpointing by segment |
| attention kernels | SDPA / Triton | Flash, cuDNN-frontend (sm100 native, ~2.5x), or math fallback (`wave3d_attn.py KERNELS`) |
| entry points | `engines.wavescan.engine.graphs.WaveScanEngine` (`train_step`, `finish_grads`), `engine.schedule.uniform_chunks` | `engines.wave3d.wave3d.make_engine(model, B, T, engine="x", graphs=True, raw_diag=..., local_window=...)` |
| trainers | `lm/train_340m/train_flagship.py`, `formal_language/harness/train_statetrack_ws.py` | `lm/train_1p3b/train_hero_wave3d.py` |
| exactness tests | `wavescan/tests/test_k1_k9.py` (26 tests: cell vs eager, nilpotency frontier, RoPE at write, grads vs eager, captured graphs, determinism, DDP all-reduce), `wavescan/tests/test_ctxnorm_vjp.py` | `lm/train_1p3b/validate_rawdiag.py` (loss / logits / per-group grads vs the scan at K = c, with and without the window; K < c CE ladder) |

```bash
pytest -q engines/wavescan/tests                          # ~5 min on one GPU (Triton autotune dominates)
torchrun --nproc_per_node=2 engines/wavescan/tests/ddp_equiv.py
PYTHONPATH=. python lm/train_1p3b/validate_rawdiag.py --T 8192 --B 1 --L 2
```

Cost model (paper Sec. on the engine): a sweep recomputes the fraction f of a layer, so training costs
(1 + K f) x a parallel Transformer. Measured at the 1.3B schedule (c = 64, K = 24): 3.3x, i.e. f ~ 0.1; the
exact chunk (K = c = 64) is 7.4x. Decoding is unchanged: one refine step per token, same cache, same attention.
