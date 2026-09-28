# Full-bandwidth transformer (FBT, Wang et al. 2026, arXiv 2608.08888) vs DM — consolidated analysis (2026-09-26)

Implementation: models/fbt/fbt_lm.py (v2, paper-faithful: u_1 = e_1, u_t = RMSNorm(W^U h_{t-1} ⊙ σ(W^G RMSNorm(e_t))), jitter 0.02 on the carried state,
W^U/W^G square no bias N(0,0.02), tied embeddings; h_{t-1} = top-of-stack post-norm vector). Two training modes:
  (a) EXACT sequential: one teacher-forced pass with a KV cache, full backprop through every position (what every FBT number in the paper's tables uses).
  (b) PAPER multi-pass: pass 1 plain stack on e; pass k re-runs the full stack in parallel on the previous pass's hidden states shifted one position and fused
      through the gate; gradients through all passes; --fbt_sched "75:1,22:2,3:3" (their Sec. 3.3 schedule) or a fixed pass count "100:P". Eval sweeps pass
      count P (P=0 = exact sequential unroll, which is how the paper decodes). Logs: analysis/fbtp/.

## 1. What exact-trained FBT does (the arm in the paper)
- CVP: 1 layer 8M tracks depth like DM: exact at 16/32/64/128/256 gates (256: min per-depth 0.965). 512 from 256: 0.77/0.77 at 5k of 30k steps (bsz 16, ~3.5 h per 1k steps) -> not certified. 1024: floor.
- A5 T=64: 1 layer forms, 1.0 in-dist, 1.0 at 2x and 4x, 0.987 at 32x. 9 layers: forms in 1/2 seeds.
- Keyed A5 (9 layers, d=1280, 180M): depth 8 at K=2 (1/2 seeds, 7k steps), K=8, K=16 (warm from 8); K=32 chain OOM'd at bsz 64 (0.97 at 4k), relaunched at bsz 16 from the 16 solver; K=64 from 16 running. At lr 4e-4 (the first sweep) every K was depth 1: the tuned cells are at 1e-4.
  1-LAYER FBT keyed (8.4M / 23M): launched 2026-09-26 03:xx, pending (analysis/looplr/keyedK*_fbt_L1_*.log).
- Length generalization (C-RASP DFA protocol, 63m 1L, lr 4e-4): parity flat 1.0 to 10x in 2/3 seeds (third: 0.66 at 10x, 0.55 at 32x); Z5 1.0; A5 1.0; Z10 0.68 at 10x; both C-RASP languages 1.0; flip-flop (sep) 1.0.
- Loop-style fixed point: none (no iteration axis).

## 2. Training cost at the 1.3B hero schedule (FineWeb-Edu 100B tokens, 4K ctx), identical WaveScan scheme for both
Chunk-parallel cost = (1 + K f) x one parallel pass; hero: c = 64 tokens, K = 24 Jacobi sweeps per chunk, Gauss-Seidel across chunks.
  parallel baseline 6ND = 6 x 1.3e9 x 1e11 = 7.8e20 FLOPs.
  DM: a sweep recomputes ~1/10 of a layer (attention read + LoRA contextual writes; trunk MLP once): f ~= 0.10 -> 3.3x measured = 2.6e21. Sweeps pipeline across layers and chunks (3-D wave).
  FBT: the recurrence spans the stack (layer 0 at sweep k+1 needs the top layer at sweep k) -> f = 1 and no layer pipelining inside a chunk.
    same schedule (K=24): 25x = 1.95e22 FLOPs, 7.6x DM; critical path K full stacks per chunk vs one.
    exact chunk (K = c = 64): 65x vs DM 7.4x.
  Ratio of premiums over the baseline: 24 extra passes vs 2.3 = ~10x.
  (Assumes FBT's correctness front advances as fast as DM's ~16 positions/sweep, which is unmeasured and favorable to FBT.)

## 3. What FBT forms as a function of training passes (analysis/fbtp/, exact-unroll eval, 12k steps A5/keyed, 30k CVP, same sizes/lrs as the exact arm)
| passes / step        | FLOPs vs parallel | A5 T=64 (exact to pos.) | CVP 16 gates          | keyed K=8 (9L)      |
| paper 75/22/3        | 1.3x              | 0.33, 0.30 (pos 16)     | 0.58 floor            | depth 1             |
| 3  (~DM 3.3x)        | 3x                | 0.10 (pos 8)            | 0.54 floor            | depth 0             |
| 4  (~DM exact 4.1x)  | 4x                | 0.45 (pos 16)           | 0.55 floor            | depth 1             |
| 8                    | 8x                | 0.62 (pos ~28)          | 0.57 floor            | depth 6 (0.84)      |
| 16                   | 16x               | 0.60 (pos 32)           | SOLVED 0.998 (19k)    | depth 2 (0.43)      |
| exact sequential     | 1x, T serial steps| solved                  | solved                | solved              |
Reach grows ~2 positions per pass; the first task to form outright (16 gates) needs 16x. Nothing forms at or below DM's cost.
P-pass parallel readout (train-time computation) can score high where the sequential unroll fails: keyed K8 P3 -> 1.00 at P=3 eval (unroll 0.06); CVP16 P8 -> 0.91 at P=8 (unroll 0.57).
The passes act as extra depth on shifted states (a looped transformer with feedback), not as the recurrence; the paper's sequential decoding does not reproduce it until P ~ T.
Extra eval passes never help a paper-schedule model (P64 = P0).

## 4. Where FBT and DM differ (paper wording)
- Both are token recurrences that track serial depth (CVP) and length-generalize better than attention; FBT is LSTM-like on length gen where DM is a lottery.
- FBT composes in one d-vector carried through the next token's input; DM composes in addressable tape entries. Keyed: DM 1L/12M to K=1024; FBT 9L/180M to 16 (32 pending); 1L pending.
- Exactness: FBT's recurrence is trainable exactly only sequentially in T; its parallel recipe is a contraction (fixed-point regime) and forms none of our tasks at <=4x FLOPs. DM's exact recurrence is chunk-parallel at ~1/10 layer per sweep.
- Decoding cost: both add ~nothing per token.

## 5. Open at freeze
FBT keyed K=32 (from 16, bsz 16) and K=64 (from 16) running; 1-layer keyed K=2/K=8 running; FBT CVP 512 will not certify in time (0.77 at 5k/30k).
Files: paper App. "Full-bandwidth transformer" paragraph; analysis/fbt_flops_note.md; analysis/fbtp/*.log; analysis/looplr/keyedK*_fbt*.log; analysis/lr1e4/fbt_n*.log; analysis/fbt/*.log (A5/Z2/Z10/flip-flop).
