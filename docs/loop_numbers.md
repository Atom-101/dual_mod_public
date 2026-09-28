# Loop numbers for the paper (tuned runs = the ones in fig loop_fixedpoints_3x3), 2026-09-25 09:40

Traces: analysis/loop_fixed_point/ood/<task>_<recipe>_{in,ood}.txt (tool formal_language/loop_fixed_point/loop_fixed_point.py; x_k = recurrent state fed to the next iteration; bf16 relative resolution 2^-8 = 3.9e-3).

## 1. Relative state change, in-distribution

| task | recipe | k=2 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|
| A5 T=64 (OOD T=128) | iteration-agnostic | 4.655e-01 | 7.989e-02 | 3.442e-02 | 1.629e-02 | 7.946e-03 |
| A5 T=64 (OOD T=128) | K∝length | 2.917e-01 | 2.831e-01 | 5.399e-02 | 1.971e-02 | 8.688e-03 |
| A5 T=64 (OOD T=128) | recurrent-depth | 8.633e-01 | 4.974e-02 | 5.357e-03 | 5.351e-03 | 5.349e-03 |
| keyed A5 K=8 D=8 (OOD D=16) | iteration-agnostic | 4.308e-01 | 4.849e-02 | 2.375e-02 | 1.215e-02 | 6.109e-03 |
| keyed A5 K=8 D=8 (OOD D=16) | K∝length | 6.427e-01 | 7.183e-02 | 3.357e-02 | 1.646e-02 | 8.094e-03 |
| keyed A5 K=8 D=8 (OOD D=16) | recurrent-depth | 7.012e-01 | 3.035e-02 | 1.703e-02 | 1.698e-02 | 1.698e-02 |
| CVP 64 gates (OOD 128; K∝len: 16 -> 32) | iteration-agnostic | 2.947e-01 | 1.778e-01 | 7.122e-02 | 2.128e-02 | 8.990e-03 |
| CVP 64 gates (OOD 128; K∝len: 16 -> 32) | K∝length | 3.157e-01 | 5.461e-02 | 2.849e-02 | 1.488e-02 | 7.691e-03 |
| CVP 64 gates (OOD 128; K∝len: 16 -> 32) | recurrent-depth | 4.827e-01 | 4.263e-02 | 4.125e-03 | 4.122e-03 | 4.116e-03 |

Notes: formed = in-dist deepest acc >= 0.95 (see table 2). Unformed panels in the figure: keyed iteration-agnostic (depth-1 shortcut), keyed K∝length K=8 (depth 1), CVP K∝length (0.58; the formed 2e-4 solver was not saved and a re-run did not form).

## 2. Deepest-bin accuracy (deepest eighth of positions / deepest gate / depth-D register)

| task | recipe | in k=16 | 32 | 64 | 128 | OOD k=16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|---|---|
| A5 T=64 (OOD T=128) | iteration-agnostic | 1.000 | 1.000 | 1.000 | 0.998 | 0.021 | 0.064 | 0.051 | 0.045 |
| A5 T=64 (OOD T=128) | K∝length | 0.878 | 0.999 | 0.998 | 0.998 | 0.016 | 0.018 | 0.018 | 0.018 |
| A5 T=64 (OOD T=128) | recurrent-depth | 0.999 | 1.000 | 1.000 | 1.000 | 0.017 | 0.026 | 0.026 | 0.024 |
| keyed A5 K=8 D=8 (OOD D=16) | iteration-agnostic | 0.021 | 0.017 | 0.018 | 0.020 | 0.019 | 0.019 | 0.021 | 0.013 |
| keyed A5 K=8 D=8 (OOD D=16) | K∝length | 0.018 | 0.017 | 0.016 | 0.016 | 0.010 | 0.013 | 0.014 | 0.016 |
| keyed A5 K=8 D=8 (OOD D=16) | recurrent-depth | 0.990 | 0.990 | 0.991 | 0.991 | 0.158 | 0.708 | 0.716 | 0.714 |
| CVP 64 gates (OOD 128; K∝len: 16 -> 32) | iteration-agnostic | 0.500 | 1.000 | 1.000 | 1.000 | 0.500 | 0.500 | 0.504 | 0.500 |
| CVP 64 gates (OOD 128; K∝len: 16 -> 32) | K∝length | 0.520 | 0.527 | 0.484 | 0.492 | 0.559 | 0.500 | 0.500 | 0.500 |
| CVP 64 gates (OOD 128; K∝len: 16 -> 32) | recurrent-depth | 0.840 | 1.000 | 1.000 | 1.000 | 0.523 | 0.523 | 0.512 | 0.516 |

Chance: A5 1/60 = 0.017; CVP 0.5; keyed 1/60.

## 3. A5 position figure: per-bin accuracy of the formed loops (bin end position) and first bin at chance (acc <= 0.05)

| recipe | 8 | 16 | 24 | 32 | 40 | 48 | 56 | 64 | 80 | 96 | 112 | 128 | 160 | 192 | 224 | 256 | first bin at chance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| iteration-agnostic (lr 1e-4, 24k steps, eval K=32) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.97 | 0.83 | 0.48 | 0.06 | 0.02 | 0.02 | 0.02 | 0.02 | 160 |
| recurrent-depth (lr 1e-4, seed 1338, eval K=64) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.96 | 0.39 | 0.08 | 0.02 | 0.02 | 0.02 | 0.02 | 0.01 | 128 |
| K∝length (102k steps, cont3) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.85 | 0.92 | 0.12 | 0.02 | 0.01 | 0.02 | 0.02 | 0.02 | 0.01 | 112 |

## 4. CVP deepest exact depth (deepest d with per-depth acc >= 0.95 for all depths <= d), best eval K

| recipe | n=64 | n=128 |
|---|---|---|
| iteration-agnostic | 64 (acc 1.000) | 29 (acc 0.721) |
| recurrent-depth | 64 (acc 1.000) | 128 (acc 1.000) |

## 5. K∝length loop on A5 T=64

Runs: cold lr 4e-4 12k steps: 0.24 in-dist (unformed). Warm continuations of that run: +30k at 2e-4 (cont), +30k at 5e-5 (cont2), +30k at 5e-5 (cont3) = 102k total -> in-dist 0.981, deepest 0.847, 2x 0.634, 4x 0.326 (the tab:a5 row). NO cold run at lr 1e-4 or 2e-4 existed until now: launched analysis/looplr/a5_kt_lr1e-4_st24k.log (24k steps, cold, 1e-4). The 8.5x-budget clause survives only if that run fails to form in 24k.

- statetrack_group_a5_loop2x64p4_tag_w1280_lr0.0004_looped.json: 1x 0.2369 2x 0.1277 4x 0.071
- statetrack_group_a5_loop2x64p4_tag_w1280_st30k_warm-a5_loop2x64p_lr0.0002_cont_looped.json: 1x 0.1909 2x 0.1021 4x 0.0558
- statetrack_group_a5_loop2x64p4_tag_w1280_st30k_warm-a5_loop2x64p_lr5e-05_KeqN4_L2_cont3_looped.: 1x 0.9807 2x 0.634 4x 0.326
- statetrack_group_a5_loop2x64p4_tag_w1280_st30k_warm-a5_loop2x64p_lr5e-05_cont2_looped.json: 1x 0.837 2x 0.4845 4x 0.2463
- statetrack_group_a5_loop2x64p4_tag_w1280_st30k_warm-a5_loop2x64p_lr5e-05_cont4_looped.json: 1x 0.9999 2x 0.7816 4x 0.4008
- statetrack_group_a5_loop2x64p4_tag_w1280_st30k_warm-a5_loop2x64p_lr5e-05_cont5e-5_looped.json: 1x 0.5148 2x 0.2871 4x 0.1491

## 6. Recurrent-depth (Huginn) keyed reach, lr 1e-4 (live)

- keyedK128_hug_from32_lr1e-4_x4: step 5000, acc 0.9779, exact depth 8
- keyedK128_hug_from64_lr1e-4: FINAL, acc 0.9965, exact depth 8
- keyedK16_hug_L1_lr1e-4: FINAL, acc 0.1816, exact depth 1
- keyedK16_hug_from8_lr1e-4: FINAL, acc 0.9996, exact depth 8
- keyedK16_hug_lr1e-4: FINAL, acc 0.9999, exact depth 8
- keyedK16_hug_lr1e-4_s1338: step 8000, acc 0.1387, exact depth 1
- keyedK256_hug_from128_lr1e-4: step 1000, acc 0.2651, exact depth 1
- keyedK256_hug_from64_lr1e-4_x4: step 3000, acc 0.126, exact depth 1
- keyedK2_hug_lr1e-4: FINAL, acc 0.24, exact depth 2
- keyedK32_hug_cold_lr1e-4: step 5000, acc 0.1258, exact depth 1
- keyedK32_hug_cold_lr1e-4_s1338: step 3000, acc 0.1297, exact depth 1
- keyedK32_hug_from16_lr1e-4: FINAL, acc 0.8904, exact depth 1
- keyedK32_hug_from8_lr1e-4: FINAL, acc 0.9975, exact depth 8
- keyedK32_hug_from8_lr1e-4_x4: FINAL, acc 0.9999, exact depth 8
- keyedK512_hug_from128_lr1e-4_x4: step 1000, acc 0.1814, exact depth 0
- keyedK64_hug_cold_lr1e-4: step 0, acc 0.0194, exact depth 0
- keyedK64_hug_from16_lr1e-4_x4: FINAL, acc 0.9983, exact depth 8
- keyedK64_hug_from32_lr1e-4: FINAL, acc 0.9983, exact depth 8
- keyedK64_hug_from8_lr1e-4: FINAL, acc 0.9995, exact depth 8
- keyedK64_hug_from8_lr1e-4_s1338: FINAL, acc 0.9998, exact depth 8
- keyedK8_hug_L1_63m_lr1e-4: FINAL, acc 0.1393, exact depth 0
- keyedK8_hug_L1_63m_lr1e-4_s1338: FINAL, acc 0.2061, exact depth 1
- keyedK8_hug_L1_lr1e-4: FINAL, acc 0.1293, exact depth 1
- keyedK8_hug_lr1e-4: FINAL, acc 0.999, exact depth 8
## 7. CORRECTIONS found while compiling this file (09:50)
- **Huginn (recurrent-depth) on CVP is NOT walled at 128.** Both 128-gate chains (from 64 and from 32, lr 1e-4) are exact at ALL 128 depths (min per-depth 1.000) when evaluated with >= 64 iterations (K=64/128/256 evals all 1.0). The table's "0.79 (54)" was the K=16 eval only. Section 4 above ("n=128: 128 (acc 1.000)") is the right reading. The earlier 256 attempt was killed on the K=16 misreading; relaunched: analysis/looplr/cvp256_hug_from128_lr1e-4{,_s1338}.log (eval K 64..512).
  Iteration-agnostic on CVP 128 gates: deepest exact 29 (0.72) at its best eval K -> its wall at 64 stands.
- **1-block Huginn does NOT hold keyed state at 12k steps**: K=8 d=1280 depth 1 (0.129); K=8 d=768 depth 0-1 (0.139, 0.206; 2 seeds); K=16 d=1280 depth 1 (0.182). 2-block Huginn holds K=8..128. Matched-depth claim (1 layer): DM holds K=512 at 12M; Huginn holds nothing at 23M/62M.
- **Huginn keyed K=128 SOLVED** (2 blocks, from64: 0.9965 exact depth 8; from32: 0.978 at 5k, forming). K=256 from64/from128 and K=512 from128 running.
- **K∝length A5 best run is cont4, not cont3**: cont4 (from cont3, +30k at 5e-5 = 132k total) 1x 0.9999, 2x 0.782, 4x 0.401 (cont3: 0.981/0.634/0.326). Table should use cont4 with '132k steps' (11x the 12k budget). Cold 1e-4 24k run launched to test whether the long-budget clause survives.
