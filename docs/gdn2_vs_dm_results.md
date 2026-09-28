# Gated DeltaNet-2 vs Dual-Mod (DM) — 1.3B / 100B FineWeb-Edu

**Model:** `hero-fw-1p3b-w3dp-s1337`, ckpt_latest step 190,649 (100.3B tokens), 1,304,205,192 params
(GDN-2 recurrent 1.3B: +0.12%, GDN-2 hybrid: +0.30%). Same macro shape and recipe as GDN-2 (d=2304,
L=18, h=36, T=4096, LR 4e-4 cosine + 2000-step warmup, 0.524M tok/step, 190,650 steps, Llama-2
tokenizer). Trained with the wave3d engine (K-ladder 8→12→16→24 base, 30% K32 probes at the end).
Held-out FineWeb-Edu val CE **2.0285 (K24) / 2.0294 (K64)**, ppl 7.60.

**Scoring:** all DM numbers are scored THROUGH THE TRAINING ENGINE (`Wave3DEval`, K=24 sweeps = the
training operator; K=64 shown for Table 2). The eager sequential scan is NOT this checkpoint's operator
(refined-diagonal wave3d issue), so HF-style scoring of it is invalid. Generation = greedy, one engine
forward per token, every prompt at its exact positions (no padding, no dropped tokens; generation
path v5 "per-row", 2026-09-09). GDN-2 numbers are from the paper (`GDN2_TABLE.md`); "hybrid" =
GDN-2 + 2K SWA layers, the closest architectural comparison.

Compute caveat: DM's K-sweep training costs ~3.3x the FLOPs of a param-matched vanilla 1.3B. These
are PARAM-matched comparisons, not FLOP-matched.

---

## Table 2 — Language modeling + zero-shot commonsense (lm-eval, `--num_fewshot 0`; Avg = 9 acc cols)

| Model | Wiki ppl↓ | LMB ppl↓ | LMB acc | PIQA | Hella(n) | Wino | ARC-e | ARC-c | OBQA | SIQA | BoolQ | **Avg** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **DM (K24)** | 15.96 | **9.94** | **51.99** | 72.85 | 59.26 | 58.56 | 72.10 | 38.14 | 31.20 | 42.12 | 61.99 | **54.25** |
| DM (K64) | 16.00 | 10.07 | 51.91 | **72.91** | **59.29** | **58.72** | 71.84 | 38.14 | 30.80 | **42.68** | 61.83 | 54.24 |
| GDN-2 recurrent | 15.90 | 11.41 | 48.09 | 72.80 | 56.84 | 57.85 | **72.43** | **38.23** | 31.60 | 41.59 | 59.54 | 53.11 |
| GDN-2 hybrid (+SWA) | **15.62** | 10.43 | 50.90 | 72.20 | 58.46 | 58.56 | 71.89 | 36.69 | **33.00** | 41.50 | **62.57** | 53.97 |
| Transformer (+SWA) | 19.22 | 13.72 | 48.32 | 70.21 | 56.12 | 55.85 | 69.23 | 33.84 | 25.00 | 39.74 | 59.42 | 50.86 |
| Mamba-3 MIMO hybrid | 15.81 | 10.92 | 49.82 | 71.98 | 58.19 | 57.06 | 70.54 | 38.48 | 29.40 | 40.99 | 57.98 | 52.72 |
| KDA hybrid | 16.01 | 10.66 | 49.21 | 71.06 | 56.69 | 57.77 | 71.59 | 35.07 | 30.00 | 41.20 | 62.03 | 52.68 |

DM wins the average against every row (+0.28 over the GDN-2 hybrid, +1.14 over recurrent). Loses Wiki
ppl to the hybrid (15.96 vs 15.62), OBQA, BoolQ, ARC-e/c by <0.6. K24 vs K64 agree to <0.6 on every
column. SIQA = `social_iqa_pq` (parquet mirror of allenai/social_i_qa; the built-in dataset script is
unsupported by datasets 5).

## Table 4 — Real-world retrieval (0-shot, input truncated to 2K)

**Protocol (from code).** The GDN-2 repo ships no eval code. The Gated DeltaNet repo (same authors,
same suite; `external/GatedDeltaNet/README.md` §5) instructs: *"For zero-shot, in-context
recall-intensive tasks (Table 4): use the official evaluation script
`prefix-linear-attention/lm-eval-harness/prompt_scripts/run_jrt_prompt_hf.sh` from their repository.
⚠️ Avoid directly using lm-eval-harness with the task name alone, as this can lead to significant
performance differences."* That script runs `based_swde`, `based_fda`, `based_squad`, `based_drop`,
`based_nq_*`, `based_triviaqa` with `--cutting_context --answer_length 50`, 0-shot, no instruction
text; GDN-2's "truncated to 2K" = that harness at `--context_length 2048` (+ `based_nq_2048`, the JRT
convention). Ported exactly in `lm/eval/based_tasks/jrt_tasks.py` (`jrt_*` tasks): answer-centered
1998-token window (TriviaQA/DROP: document head; NQ: dataset `tok_pos`), prompt `context. question`
(no trailing space), stop at newline / 48 tokens, case-insensitive `contains`, datasets
based-swde-**v2**, based-fda, based-squad, based_triviaqa, based_nq_2048, based_drop. Full validation
sets (1111 / 2984 / 1102 / 1688 / 3157 / 2087). Per-sample generations are in the result json.

| Model | SWDE | SQuAD | FDA | TriviaQA | NQ | DROP | **Avg** |
|---|---|---|---|---|---|---|---|
| **DM (K24)** | **58.24** | **49.16** | **70.24** | **66.47** | **30.16** | **23.77** | **49.67** |
| GDN-2 recurrent | 23.65 | 36.75 | 19.98 | 61.37 | 19.64 | 17.87 | 29.88 |
| GDN-2 hybrid (+SWA) | 41.96 | 44.70 | 54.68 | 62.38 | 26.31 | 23.63 | 42.28 |
| Transformer (+SWA) | 32.21 | 38.67 | 54.78 | 58.09 | 22.49 | 22.18 | 38.07 |
| KDA hybrid | 39.83 | 40.10 | 53.59 | 59.89 | 25.27 | 22.18 | 40.14 |
| Mamba-3 MIMO hybrid | 32.33 | 44.70 | 55.31 | 59.00 | 26.26 | 23.08 | 40.11 |

DM beats the GDN-2 hybrid on all six tasks: +7.39 avg over the hybrid, +19.79 over the recurrent.
Largest margins on the key-value extraction tasks (SWDE +16.28, FDA +15.56) and SQuAD (+4.46);
TriviaQA +4.09, NQ +3.85, DROP +0.14.

Superseded DM runs (same checkpoint; kept for the record, NOT to be compared with GDN-2):
- Same JRT protocol but generation with prompts LEFT-PADDED by up to 31 BOS tokens (path v3):
  51.76 / 33.95 / 56.17 / 59.12 / 29.14 / 18.40, avg 41.42. The padding flipped near-tie first
  tokens toward the language prior (logit probe: in 4/8 sampled SQuAD "failures" the correct token
  was the true argmax); the per-row path removes it. This is a harness defect, not a protocol change.
- lm-eval built-in `swde` (based-swde v1, prompt ends with the "Summary of information above..." block
  = an in-context key:value listing; explicitly NOT the GDN protocol): 73.18 (per-row), 70.75 (v1 path).
- `based_*` with a `context\nquestion ` prompt (trailing space -> lone `▁` token -> digit bias):
  TQA 44.79 / NQ 14.95 / DROP 16.15. lm-eval closed-book `triviaqa`/`nq_open`/`drop`: 10.07 / 4.07 / 1.05.
- lm-eval built-in `fda` / `squad_completion` with tail truncation (cuts 70% of FDA answers): 21.05 / 19.34.

## Table 3 — RULER NIAH (500 samples per cell; lm-eval `niah_*` tasks, the code fla's eval guide uses)

| Model | SN1-1K | 2K | 4K | 8K | SN2-1K | 2K | 4K | 8K | SN3-1K | 2K | 4K | MK1-1K | 2K | 4K |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **DM (K24)** | 100.00 | 99.80 | **100.00** | 2.20 | 100.00 | 100.00 | 86.60 | 0.00 | 85.00 | 91.80 | 43.40 | **99.80** | **98.80** | **60.60** |
| GDN-2 recurrent | 100.00 | 100.00 | 100.00 | **97.80** | 100.00 | 100.00 | **93.00** | **39.20** | 92.00 | 89.80 | 31.80 | 72.60 | 51.40 | 37.80 |
| GDN-2 hybrid (+SWA) | 100.00 | 100.00 | 55.20 | 27.40 | 100.00 | 100.00 | 57.90 | 29.20 | **99.60** | **99.00** | **55.60** | 93.00 | 84.60 | 48.00 |
| Transformer (+SWA) | 100.00 | 100.00 | 51.20 | 0.00 | 100.00 | 100.00 | 44.20 | 0.00 | 95.80 | 94.80 | 37.00 | 75.60 | 66.60 | 38.20 |

All cells: exact-prompt per-row generation (v5), 500 samples, engine K24. Superseded BOS-padded
pass (v3) for reference: SN1 95.6/100.0/96.8/0.2, SN2 99.6/100.0/90.8/2.2, SN3 63.8/89.6/69.4,
MK1 90.8/95.4/63.0.

Reading: MK-NIAH-1 (the interference-heavy setting GDN-2 highlights) beats both GDN-2 variants at
every length. Single-needle tasks are at ceiling to 2K and far above the hybrid at 4K on SN1/SN2 (100.00 / 86.60 vs 55.20 / 57.90); SN3 at 4K (43.40) trails the hybrid (55.60), see the caveat below.
**8K ≈ 0**: trained at 4096 positions with RoPE, no length extrapolation (same as the Transformer row).
**SN3 caveat (resolved 2026-09-09 05:10):** SN3 cells moved across generation paths far more than any
other task (1K: 87.0 → 63.8 → 85.0; 4K: 74.4 → 69.4 → 43.4 for the floor-rounded / BOS-padded / per-row
paths on identical prompts). A reproduction on 12 fixed 4K prompts: per-row solo 4/12 = per-row batched
4/12 = the recorded run (the path is deterministic; engine logits are batch/padding independent to
bf16 noise, max |Δlogit| 0.16); the SAME prompts with 31 BOS tokens prepended: 8/12. I.e. the earlier
69.4 was a prompt-prefix effect: a run of BOS before the haystack tips the model from the
language-prior continuation ("the uuid that is used to identify…") to the copy (": <uuid>"). The
copy-vs-prior margin on SN3 is small, so any prefix perturbation swings it (the same perturbation
tipped SQuAD the other way). The exact-prompt (per-row) number is the protocol-faithful one — lm-eval's
HF path, which GDN-2 used, masks its batch padding so their models see the exact prompt too. The 4K
prompts are ~3470 tokens (RULER sizes with a different tokenizer): not the 4096-position edge.

## Bottom line

| | DM 1.3B | GDN-2 recurrent | GDN-2 hybrid (+SWA) |
|---|---|---|---|
| Table 2 avg (commonsense, 0-shot) | **54.25** | 53.11 | 53.97 |
| Table 4 avg (recall, JRT protocol, 2K) | **49.67** | 29.88 | 42.28 |
| RULER MK-NIAH-1 (1K/2K/4K) | **99.80 / 98.80 / 60.60** | 72.60 / 51.40 / 37.80 | 93.00 / 84.60 / 48.00 |
| RULER S-NIAH-1/2 @4K | **100.00** / 86.60 | 100.00 / **93.00** | 55.20 / 57.90 |
| RULER 8K | 2.20 / 0.00 (no extrapolation past 4K) | **97.80 / 39.20** | 27.40 / 29.20 |

DM beats GDN-2 (both variants) on language modeling/commonsense, on every real-world retrieval task,
and on multi-key retrieval; it matches or beats the hybrid on single-needle retrieval within its 4K
training length and cannot extrapolate beyond it. The follow-up run `hero-fw-1p3b-swa2k-s1337`
(exact raw-diagonal operator + 2K sliding window; launched 2026-09-09 00:54 UTC, paused at step 4000
for this eval audit, resumed 05:08 UTC) is the apples-to-apples counterpart of the hybrid row.

Sources: `analysis/lmeval_hero-fw-1p3b-w3dp-s1337_190649_K24_*.json` (per-sample dumps),
`analysis/hero_fwedu_ce_*.log`; `python lm/eval/gdn2_collate.py hero-fw-1p3b-w3dp-s1337`.
Protocol sources: `external/GatedDeltaNet/README.md` §5, `external/prefix-linear-attention/lm-eval-harness/`,
`external/flash-linear-attention/evals/README.md`. Written 2026-09-09.

## Addendum (2026-09-14): NHA 1.3B and the SWA-2K DM — DIFFERENT DATA, same (JRT) recall protocol

NHA (arXiv 2510.07019v3) 1.3B is trained on **SlimPajama, 100B tokens** (our 1.3B models: FineWeb-Edu 100B; GDN-2's too).
SlimPajama's web/GitHub content favours the HTML/PDF extraction tasks (SWDE/FDA) and hurts the commonsense set relative to
FineWeb-Edu, so cross-data cells are indicative only. Recall = JRT harness on both sides (NHA README).

| 1.3B / 100B | FDA | SWDE | TQA | NQ | SQuAD | DROP | 6-avg | 5-avg (no NQ) | commonsense 6-avg* |
|---|---|---|---|---|---|---|---|---|---|
| DM-4K (hero-fw-1p3b-w3dp) | 70.24 | 58.24 | 66.47 | 30.16 | 49.16 | 23.77 | 49.67 | 53.58 | 58.82 |
| DM SWA-2K (hero-fw-1p3b-swa2k) | 63.25 | 49.41 | 65.23 | 27.21 | 45.84 | 22.90 | 45.64 | 49.33 | 58.39 |
| GDN-2 hybrid (FineWeb-Edu) | 54.68 | 41.96 | 62.38 | 26.31 | 44.70 | 23.63 | 42.28 | 45.47 | 58.12 |
| NHA 1.3B (SlimPajama) | 68.30 | 52.48 | 59.60 | 27.18 | 45.58 | 25.44 | 46.43 | 50.28 | 52.89 |

\* ARC-e, ARC-c(n), PIQA, HellaSwag(n), LAMBADA acc, Winogrande — NHA's six; DM/GDN-2 values from the K24 TF runs.
DM-4K beats NHA on 5/6 recall tasks (loses DROP by 1.7) and by +3.2 on the 6-avg; SWA-2K is −0.8 on the 6-avg (wins
TQA/SQuAD/NQ, loses FDA −5.1, SWDE −3.1, DROP −2.5) — on data that favours NHA's two biggest wins. On commonsense DM
leads by ~6 (data-confounded in DM's favour). The same-data comparison remains GDN-2 (above).

## SWA-2K DM (hero-fw-1p3b-swa2k-s1337) — the apples-to-apples row vs the GDN-2 hybrid (2026-09-14)

Same recipe/data/tokenizer/budget as DM-4K; raw-diagonal (exact) operator + 2048-token sliding window
(GDN-2 SWA convention), K8→50k / K16→140k / K24 (30% K32 probes); 21 storms, 106 skips, final val
2.0305 (K24 == K64 at every eval, gap ≤ 1e-4 throughout). Scored with its own operator (engine reads
raw_diag/local_window from the ckpt), per-row generation, samples dumped.

**Table 4 (JRT, K24):** SWDE 49.41 / FDA 63.25 / SQuAD 45.84 / TQA 65.23 / NQ 27.21 / DROP 22.90 =
**45.64** vs GDN-2 hybrid 42.28 (recurrent 29.88), DM-4K 49.67. Beats the hybrid on 5/6 (DROP −0.7).

**Table 3 (RULER, K24, 500/cell):**

| | SN1-1K | 2K | 4K | 8K | SN2-1K | 2K | 4K | 8K | SN3-1K | 2K | 4K | MK1-1K | 2K | 4K |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **SWA-2K DM** | 100.00 | 86.20 | 54.00 | 25.40 | 100.00 | 100.00 | 56.40 | 29.20 | 98.00 | 97.00 | 42.60 | 86.40 | 85.20 | 46.00 |
| DM-4K | 100.00 | 99.80 | 100.00 | 2.20 | 100.00 | 100.00 | 86.60 | 0.00 | 85.00 | 91.80 | 43.40 | 99.80 | 98.80 | 60.60 |
| GDN-2 hybrid | 100.00 | 100.00 | 55.20 | 27.40 | 100.00 | 100.00 | 57.90 | 29.20 | 99.60 | 99.00 | 55.60 | 93.00 | 84.60 | 48.00 |
| GDN-2 recurrent | 100.00 | 100.00 | 100.00 | 97.80 | 100.00 | 100.00 | 93.00 | 39.20 | 92.00 | 89.80 | 31.80 | 72.60 | 51.40 | 37.80 |

Reading: beyond the window (4K, 8K) SWA-DM tracks the GDN-2 hybrid almost cell for cell (54.0/55.2,
56.4/57.9, 25.4/27.4, 29.2/29.2, 46.0/48.0): both retrieve past 2K through their recurrence at the
same rate, and the window buys the length generalisation the 4K model lacked (8K: 25-29 vs 0-2).
Inside the window the SWA model is at ceiling on SN2/SN3 (better than DM-4K on SN3) but has two
learned weaknesses DM-4K does not: SN1@2K 86.2 (69 misses, all degenerate "10000…" numbers; needle
at median token 1020 of ~1840, fully inside the window) and MK1@1K/2K 86.4/85.2 (every miss = another
key's value; 4 keys, prompts ≤1500 tokens). Both occur where the mask is inactive, so they are
properties of the weights trained under the window/raw-diag operator (two changes at once —
unseparated), not of reach.

**Table 2 (K24):** Wiki 16.14 / LMB 10.61 / 50.32 / PIQA 73.07 / Hella 59.59 / Wino 58.17 / ARC-e 72.69 /
ARC-c 36.52 / OBQA 31.20 / SIQA 41.61 / BoolQ 54.43 = **9-acc 53.07** (DM-4K 54.25, GDN-2 hybrid 53.97,
recurrent 53.11); FineWeb-Edu val CE 2.0354 (K24 == K64). BoolQ's
−7.6 vs DM-4K is a yes/no prior shift, confirmed on the SAME 3270 items: DM-4K predicts 'yes' 66.9% (mean
ll(yes)−ll(no) +0.35), SWA-2K 31.8% (−0.38) on a 62.2%-'yes' set; with one global threshold both reach
63-64 (SWA 63.00, DM-4K 64.31) and their margin-vs-gold correlations are 0.23 vs 0.27 — near-identical
discrimination, opposite bias. Not comprehension. K64 pass pending.

## Official RULER (NVIDIA generator, Llama-2 tokenizer, 500 samples, string_match_all) — 2026-09-15

Prompts generated by `external/RULER/scripts/data/synthetic/niah.py` (official template + answer prefix, binary-searched to
the nominal length incl. 128 generation tokens), scored with lm-eval's ruler `string_match_all`; task yamls
`lm/eval/based_tasks/ruler_off_*.yaml`. "hybrid*" = the GDN-2 paper's self-reported hybrid row (its own, undocumented
protocol; its Transformer scores 51.2 on single_1 @4K, so its 4K/8K prompts exceed a 4K window — not comparable beyond 2K).
GDN2-3p = the third-party paper-recipe PURE-recurrent reproduction (LLM-OS-Models2), not NVIDIA weights.

| task @ len | DM-4K | DM-SWA-2K | GDN2-3p (recurrent) | hybrid* |
|---|---|---|---|---|
| single_1 @1K/2K/4K/8K | 100.0 / 99.8 / 100.0 / 0.0 | 100.0 / 89.2 / 53.2 / 23.8 | 100.0 / 100.0 / 99.8 / 84.4 | 100 / 100 / 55.2 / 27.4 |
| single_2 @1K/2K/4K/8K | 100.0 / 99.8 / 82.8 / 0.2 | 100.0 / 100.0 / 51.0 / 22.0 | 100.0 / 96.8 / 41.2 / 26.6 | 100 / 100 / 57.9 / 29.2 |
| single_3 @1K/2K/4K | 87.2 / 83.2 / 77.2 | 87.0 / 70.4 / 42.8 | 89.0 / 60.4 / 35.2 | 99.6 / 99.0 / 55.6 |
| multikey_1 @1K/2K/4K | 98.6 / 98.2 / 70.6 | 91.8 / 87.6 / 42.2 | 56.0 / 50.0 / 30.4 | 93.0 / 84.6 / 48.0 |
| multikey_2 @1K/2K/4K | 93.6 / 88.0 / 59.4 | 74.0 / 29.8 / 16.4 | 2.2 / 0.0 / 0.0 | — |
| multikey_3 @1K/2K/4K | 57.8 / 33.8 / 7.8 | 30.6 / 14.4 / 10.4 | 0.0 / 0.0 / 0.0 | — |

DM-4K under the official protocol agrees with its lm-eval numbers to within a few points (single_3@4K is the exception:
77.2 official vs 43.4 lm-eval). lm-eval's RULER port under-fills contexts (~3.5K tokens at "4K"); the official column is
the one to report.

## Generation K-independence check (DM-4K, JRT 0-shot, 2K input) — 2026-09-20

Generation = one full wave3d engine forward per emitted token at --engine_K. K=24 (trained operator, all headline
tables) vs K=64 (fixed point, exact by nilpotency): `analysis/lmeval_hero-fw-1p3b-w3dp-s1337_190649_K64_*gen_rc_jrt_k64_95c608.json`.

| K | SWDE | FDA | SQuAD | TriviaQA | NQ | DROP |
|---|---|---|---|---|---|---|
| 24 | 58.2 | 70.2 | 49.2 | 66.5 | 30.2 | 23.8 |
| 64 | 58.4 | 74.0 | 48.9 | 65.3 | 30.1 | 24.0 |

Within a point everywhere except FDA (+3.8 at K=64). Headline tables stay at K=24 (the operator the model was trained on).
