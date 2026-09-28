# Training-FLOP estimate: FBT vs DM at the 1.3B hero schedule (FineWeb-Edu 100B tokens, 4K ctx)
Parallel baseline (Transformer++ 1.3B): 6ND = 6 x 1.3e9 x 1.0e11 = 7.8e20 FLOPs.
Chunk-parallel cost model (paper Sec. engine): (1 + K f) x baseline; K sweeps per chunk, f = fraction of a layer recomputed per sweep.
DM hero: c = 64, K = 24, 3.3x measured => K f = 2.3 => f ~= 0.10 (attention read + LoRA contextual writes; trunk MLP computed once). ~2.6e21 FLOPs.
FBT: the recurrence is h_{t-1} (top of the stack) -> layer-0 input at t, so every sweep recomputes the WHOLE stack: f = 1.
  - Same sweep count as DM (K = 24, i.e. assuming the same 16-positions-per-sweep correctness front, which is unmeasured for FBT): (1 + 24) = 25x => ~1.95e22 FLOPs, 7.6x DM.
  - Exact within a 64-token chunk (K = c = 64; DM exact at K = 64 is 1 + 64 x 0.10 = 7.4x): 65x => ~5.1e22 FLOPs, 8.8x DM-exact.
  - Wang et al.'s scheduled multi-pass: a few full-stack passes (their "3% deeper passes"), cheap, but it trains a contraction and inherits the fixed-point behaviour measured for the loops (paper Sec. related work).
Ratio that matters: DM's recurrence lives inside the attention layer (f ~ 0.1); FBT's spans the stack (f = 1). At matched sweep count the FBT training premium is ~10x DM's premium (24 vs 2.3 extra passes).

Apples-to-apples framing (user 2026-09-25): both under the WaveScan scheme (Jacobi sweeps within a chunk, Gauss-Seidel across chunks), hero schedule c=64, K=24. FLOPs: DM 3.3x vs FBT 25x. Critical path per chunk: DM = one stack + K partial sweeps per layer that pipeline across layers/chunks (the 3-D wave); FBT = K full stacks in series (layer 0 at sweep k+1 needs the top layer at sweep k), so no layer pipelining within a chunk. Exact chunk (K=c=64): DM 7.4x, FBT 65x.
