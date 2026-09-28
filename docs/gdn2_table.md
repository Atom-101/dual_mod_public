# Gated DeltaNet-2 (GDN-2) — 1.3B / 100B-token FineWeb-Edu targets to beat

Source: `external/gateddeltanet-2/paper/GDN2_paper.pdf` (Hatamizadeh, Choi,
Kautz, "Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention",
arXiv:2605.22791) + `external/gateddeltanet-2/README.md`.

All GDN-2 baselines below are **1.3B params, trained on 100B tokens of
FineWeb-Edu, T=4K, 0.5M-token global batch, matched recurrent state**
(paper §4 + §E.1). Our single DM-1.3B run drops directly into these tables —
there is no control run, so every metric/num_fewshot/protocol we use MUST equal
GDN-2's or the comparison is invalid.

---

## DATA-RECIPE 1:1 VERDICT  →  **MATCH** (one low-risk residual, one training-config TODO)

| Item | GDN-2 (cited) | Our plan (`lm/data/tokenize_fwedu.py`) | Verdict |
|---|---|---|---|
| Corpus | **FineWeb-Edu** — paper §E.1 "100B tokens sampled from FineWeb-Edu [26]"; README "100B tokens of FineWeb-Edu"; `scripts/tsz1024x4k_100B_swa_gdn2.sh` `TRAIN_DATA=/data/fineweb-edu/data` | `HuggingFaceFW/fineweb-edu` | ✅ match |
| (NOT SlimPajama) | `corpus_name` default `'slimpajama'` in `pretrain.py:461` / `data.py` is an **unused argparse default**; reported runs read pre-tokenized FineWeb-Edu bins | — | ✅ SlimPajama ruled out |
| Subset / size | "100B tokens **sampled** from FineWeb-Edu"; `data.py` has a dedicated `'fineweb-edu-sample'` flat-parquet loader | `sample/100BT` (the canonical 100B-token FineWeb-Edu sample) | ✅ match |
| Token budget | **100B** — `pretrain.py:511` `max_tokens=int(1e11)`; paper §E.1/§4 | 100B (`--budget_gb 100`) | ✅ match |
| Tokenizer | **TinyLlama_v1.1** = Llama-2 32k, vocab 32000 — `pretrain.py:473` | `data/llama2_tok` (TinyLlama 32k) | ✅ match |
| Context length | **4096** — `lit_gpt/config.py:135` `block_size=4096`; paper §E.1 "4K tokens" | 4096 | ✅ match |
| BOS / EOS | `data.py` `tokenizer(text)` w/ default `add_special_tokens=True` on Llama = **BOS prepended, NO EOS**; docs concatenated then chunked | `[BOS=1] + tok(text, add_special_tokens=False)`, concatenated | ✅ match (BOS yes, EOS no) |
| Doc separator | Stream path (`data.py`) inserts **no** separator — plain BOS-prefixed concat | no separator | ✅ match (see residual) |
| Packing dtype | `int16` (vocab 32000 < 32767) — `data.py` | `uint16` | ✅ equivalent (2-byte, both hold 32000; signedness only) |
| Global batch | **0.5M tokens** (=128 seq × 4096) — paper §E.1; README | not set in `config_1p3b.py` | ⚠️ **TODO in trainer** (data recipe unaffected) |

**Residual (low risk):** GDN-2's *reported* runs consume pre-tokenized `.bin`
files via `lit_gpt/packed_dataset.py` (`PackedDatasetBuilder` takes a
`sep_token`), whose **pretokenizer script is not in the repo**. The only
tokenization logic present (the stream path `data.py`) is BOS-prefix + concat +
**no** inter-doc separator — which we replicate exactly. If their offline
pretokenizer inserted an EOS/sep between documents, that is a minor packing
nuance very unlikely to move table-level numbers. Our recipe matches the
in-repo convention 1:1.

**Bottom line:** tokenize FineWeb-Edu `sample/100BT`, TinyLlama-32k, BOS/no-EOS,
100B tokens, T=4096 — **this is correct, proceed**. Only open item is a
training-config one: set the trainer's global batch to 0.5M tokens.

---

## EVAL MAPPING — each GDN-2 metric → exact lm-eval task, metric key, shots

Cited to paper §E.2 (Evaluation) + Tables 2/3/4. All **zero-shot**
(paper §E.2: "For zero-shot transfer, we report..."). Runner:
`lm/eval/run_lm_eval.py` (DM checkpoint, `--tok_dir data/llama2_tok`).

### A. Language modeling + commonsense (Table 2) — `num_fewshot=0`

| GDN-2 col | lm-eval task | metric key | shots | notes / cite |
|---|---|---|---|---|
| Wiki ppl ↓ | `wikitext` | `word_perplexity` | 0 | §E.2 WikiText [27] |
| LMB ppl ↓ | `lambada_openai` | `perplexity` | 0 | §E.2 LAMBADA [28] |
| LMB acc ↑ | `lambada_openai` | `acc` | 0 | same task reports both |
| PIQA ↑ | `piqa` | `acc` | 0 | Table 2 header = **acc** (not acc_norm) |
| Hella ↑ | `hellaswag` | **`acc_norm`** | 0 | Table 2 header = **acc_n** |
| Wino ↑ | `winogrande` | `acc` | 0 | Table 2 header = acc |
| ARC-e ↑ | `arc_easy` | `acc` | 0 | Table 2 header = acc |
| ARC-c ↑ | `arc_challenge` | `acc` | 0 | **CONFIRMED acc** (Table 2 header "ARC-c acc", not acc_n) |
| OBQA ↑ | `openbookqa` | `acc` | 0 | **CONFIRMED acc** (Table 2 header "OBQA acc", not acc_n) |
| SIQA ↑ | `social_iqa` | `acc` | 0 | §E.2 Social IQa [34] |
| BoolQ ↑ | `boolq` | `acc` | 0 | §E.2 BoolQ [35] |
| Avg. acc | mean of the 9 acc cols (LMB acc, PIQA, Hella acc_n, Wino, ARC-e, ARC-c, OBQA, SIQA, BoolQ) | — | — | paper "Avg. acc" |

> **Metric caveat:** lm-eval auto-reports BOTH `acc` and `acc_norm` for the MC
> tasks, so `gdn2_evals.sh` captures both — but the table column to read is
> **acc for every task except HellaSwag (acc_norm)**. This differs from the
> common fla/lm-eval habit of using acc_norm for ARC-c/OBQA/PIQA; GDN-2's
> Table 2 headers explicitly say "acc" for those, so match on `acc`.

All commonsense/LM tasks are in the standard lm-eval build — **run_lm_eval.py
supports them today, no new task needed.**

### B. Long-context RULER NIAH (Table 3) — `num_fewshot=0`

Map: S-NIAH-1/2/3 → `niah_single_1/2/3`; MK-NIAH-1 → `niah_multikey_1`.
Lengths taken from the exact columns present in Table 3:

| GDN-2 row | lm-eval task | lengths reported | `--ruler_lengths` |
|---|---|---|---|
| S-NIAH-1 | `niah_single_1` | 1K,2K,4K,8K | `1024,2048,4096,8192` |
| S-NIAH-2 | `niah_single_2` | 1K,2K,4K,8K | `1024,2048,4096,8192` |
| S-NIAH-3 | `niah_single_3` | 1K,2K,4K **(no 8K)** | `1024,2048,4096` |
| MK-NIAH-1 | `niah_multikey_1` | 1K,2K,4K **(no 8K)** | `1024,2048,4096` |

- Metric: RULER string-match accuracy (task built-in; the `--ruler_samples`
  count is not stated by GDN-2 → use lm-eval's canonical **500/length** for the
  table run; our fast posttrain uses 100).
- `--ctx_len` must cover the longest window incl. RULER template overhead:
  8448 for the 8K tasks, 4352 for the ≤4K tasks (raw-RoPE extrapolation, no
  scaling — same as `POSTTRAIN_SUITE.md` §E).
- **run_lm_eval.py supports these today** (`--ruler_lengths/--ctx_len/
  --ruler_samples`; task names confirmed in the installed lm-eval).

### C. Real-world retrieval (Table 4) — `num_fewshot=0`, **input truncated to 2K tokens**

Paper §E.2 + Table 4 caption ("input length truncated to 2K tokens"; SQD=SQuAD,
TQA=TriviaQA). Inherited from the Based/Zoology recall suite [37] as packaged in
the fla eval suite [41] — all **0-shot in-context** (document in the prompt).

| GDN-2 col | lm-eval task | metric | shots | supported? |
|---|---|---|---|---|
| SWDE | `swde` | contains/`exact_match` (task built-in) | 0 | ✅ built-in |
| SQD (SQuAD) | `squad_completion` | contains (Based variant) | 0 | ✅ built-in |
| FDA | `fda` | contains/`exact_match` | 0 | ✅ built-in |
| TQA (TriviaQA) | `triviaqa` | `exact_match` | 0 | ✅ built-in |
| NQ | `nq_open` | `exact_match` | 0 | ✅ built-in |
| DROP | `drop` | `f1`/`em` | 0 | ✅ built-in |

> **CORRECTION (2026-09-08):** TriviaQA / NQ / DROP in the Based/fla suite are **document-grounded**
> (the passage is in the prompt; `contains` metric), NOT lm-eval's closed-book `triviaqa`/`nq_open`/`drop`
> (which gave 10.1 / 4.1 / 1.1 on the w3dp hero — not comparable to GDN-2's 61 / 26 / 23). Use
> `based_triviaqa`, `based_nq` (hazyresearch/based_nq_2048 = the 2K-truncated split), `based_drop`
> from `lm/eval/based_tasks/` (phase `gen_rc_based` in `gdn2_evals_gen.sh`; `gdn2_collate.py` reads them).
> `swde` / `fda` / `squad_completion` builtins are already the Based doc-grounded tasks.

> **Protocol flag:** GDN-2's Table 4 is **0-shot with 2K truncation**. This is
> NOT the same as our internal `POSTTRAIN_SUITE.md` (which runs nq_open/triviaqa
> 5-shot and drop 3-shot). For the GDN-2 table match, use **0-shot** and the 2K
> input cap. `gdn2_evals.sh` sets `--num_fewshot 0` for all six.
> `run_lm_eval.py` supports all six via the built-in lm-eval tasks (the repo's
> `lm/eval/based_tasks/based_triviaqa|based_drop` are an alternate hazyresearch
> implementation — NOT used here, to keep names/metrics identical to fla/GDN-2).

---

## TARGET NUMBERS — beat these (all percentages except ppl)

### Table 2 — Language modeling + commonsense (zero-shot)

**Recurrent models**

| Model | Wiki ppl↓ | LMB ppl↓ | LMB acc | PIQA | Hella (n) | Wino | ARC-e | ARC-c | OBQA | SIQA | BoolQ | Avg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Mamba-2 | 16.79 | 12.38 | 45.24 | 72.58 | 55.51 | 55.33 | 70.68 | 35.26 | 31.00 | 40.63 | 60.19 | 51.82 |
| Gated DeltaNet | 16.40 | 11.89 | 49.62 | 72.31 | 56.50 | 56.75 | 68.81 | 35.15 | 30.20 | 40.53 | 58.78 | 52.07 |
| KDA | 16.81 | 11.68 | 48.13 | 72.09 | 55.75 | 55.72 | 70.83 | 35.92 | 30.40 | 40.99 | 60.67 | 52.28 |
| Mamba-3 (SISO) | 16.30 | 12.99 | 45.06 | 72.31 | 55.58 | 56.20 | 70.45 | 34.56 | 31.00 | 41.76 | 55.90 | 51.42 |
| Mamba-3 (MIMO) | 16.45 | 11.66 | 47.82 | 72.36 | 56.49 | 55.78 | 72.38 | 38.07 | 30.00 | 40.89 | 57.74 | 52.39 |
| **Gated DeltaNet-2** | **15.90** | **11.41** | 48.09 | 72.80 | 56.84 | 57.85 | 72.43 | 38.23 | 31.60 | 41.59 | 59.54 | **53.11** |

**Attention / hybrid models (+ 2K SWA)**

| Model | Wiki ppl↓ | LMB ppl↓ | LMB acc | PIQA | Hella (n) | Wino | ARC-e | ARC-c | OBQA | SIQA | BoolQ | Avg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Transformer | 19.22 | 13.72 | 48.32 | 70.21 | 56.12 | 55.85 | 69.23 | 33.84 | 25.00 | 39.74 | 59.42 | 50.86 |
| Mamba-2 | 17.46 | 11.29 | 48.05 | 71.47 | 57.52 | 56.17 | 70.50 | 34.73 | 29.80 | 40.35 | 59.31 | 51.99 |
| Gated DeltaNet | 16.00 | 10.82 | 48.71 | 70.06 | 57.50 | 56.83 | 70.41 | 35.15 | 30.60 | 40.97 | 60.00 | 52.25 |
| KDA | 16.01 | 10.66 | 49.21 | 71.06 | 56.69 | 57.77 | 71.59 | 35.07 | 30.00 | 41.20 | 62.03 | 52.68 |
| Mamba-3 (SISO) | 15.54 | 10.65 | 49.19 | 71.01 | 58.75 | 57.30 | 70.54 | 36.35 | 32.00 | 41.20 | 57.86 | 52.69 |
| Mamba-3 (MIMO) | 15.81 | 10.92 | 49.82 | 71.98 | 58.19 | 57.06 | 70.54 | 38.48 | 29.40 | 40.99 | 57.98 | 52.72 |
| **Gated DeltaNet-2** | **15.62** | **10.43** | **50.90** | 72.20 | 58.46 | 58.56 | 71.89 | 36.69 | 33.00 | 41.50 | 62.57 | **53.97** |

### Table 3 — RULER NIAH accuracy

**Recurrent models**

| Model | SN1-1K | 2K | 4K | 8K | SN2-1K | 2K | 4K | 8K | SN3-1K | 2K | 4K | MK1-1K | 2K | 4K |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Mamba-2 | 100.0 | 100.0 | 97.0 | 55.8 | 99.6 | 99.6 | 62.6 | 21.0 | 59.2 | 38.6 | 14.4 | 29.0 | 21.2 | 21.4 |
| Gated DeltaNet | 99.8 | 100.0 | 100.0 | 97.6 | 100.0 | 100.0 | 87.2 | 32.0 | 89.8 | 54.2 | 60.6 | 54.0 | 27.8 | 27.8 |
| KDA | 100.0 | 100.0 | 99.2 | 70.6 | 100.0 | 100.0 | 89.0 | 30.6 | 77.4 | 63.2 | 26.2 | 54.0 | 44.2 | 28.0 |
| Mamba-3 (SISO) | 100.0 | 99.0 | 63.4 | 27.8 | 100.0 | 99.4 | 25.2 | — | 60.2 | 35.6 | 12.2 | 44.8 | 27.4 | 26.0 |
| Mamba-3 (MIMO) | 100.0 | 99.8 | 93.0 | 35.6 | 99.8 | 98.8 | 64.2 | 27.2 | 89.2 | 72.4 | 29.2 | 49.4 | 19.2 | 18.0 |
| **Gated DeltaNet-2** | 100.0 | 100.0 | 100.0 | 97.8 | 100.0 | 100.0 | 93.0 | 39.2 | 92.0 | 89.8 | 31.8 | 72.6 | 51.4 | 37.8 |

**Attention / hybrid models**

| Model | SN1-1K | 2K | 4K | 8K | SN2-1K | 2K | 4K | 8K | SN3-1K | 2K | 4K | MK1-1K | 2K | 4K |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Transformer | 100.0 | 100.0 | 51.2 | 0.0 | 100.0 | 100.0 | 44.2 | 0.0 | 95.8 | 94.8 | 37.0 | 75.6 | 66.6 | 38.2 |
| Mamba-2 | 100.0 | 100.0 | 51.8 | 25.4 | 100.0 | 99.6 | 52.4 | 25.8 | 97.8 | 86.8 | 48.0 | 82.6 | 58.6 | 39.0 |
| Gated DeltaNet | 100.0 | 100.0 | 47.2 | 22.4 | 100.0 | 99.8 | 57.3 | 25.6 | 94.8 | 91.2 | 47.2 | 91.0 | 78.4 | 44.8 |
| KDA | 100.0 | 100.0 | 51.8 | 26.2 | 100.0 | 100.0 | 56.0 | 23.0 | 97.2 | 93.4 | 51.6 | 82.4 | 79.0 | 40.4 |
| Mamba-3 (MIMO) | 100.0 | 100.0 | 49.0 | 22.8 | 100.0 | 100.0 | 53.0 | 27.8 | 95.0 | 90.4 | 44.0 | 78.8 | 65.6 | 33.6 |
| **Gated DeltaNet-2** | 100.0 | 100.0 | 55.2 | 27.4 | 100.0 | 100.0 | 57.9 | 29.2 | 99.6 | 99.0 | 55.6 | 93.0 | 84.6 | 48.0 |

(SN1=S-NIAH-1, SN2=S-NIAH-2, SN3=S-NIAH-3, MK1=MK-NIAH-1. "—" = value not
legible/reported. Best per family/length is bolded in the paper.)

### Table 4 — Real-world retrieval (0-shot, input truncated to 2K)

**Recurrent models**

| Model | SWDE | SQuAD | FDA | TriviaQA | NQ | DROP | Avg |
|---|---|---|---|---|---|---|---|
| Mamba-2 | 17.24 | 32.38 | 14.53 | 58.35 | 18.91 | 19.60 | 26.84 |
| Gated DeltaNet | 17.90 | 32.67 | 18.52 | 59.60 | 20.16 | 19.69 | 28.09 |
| Mamba-3 (SISO) | 17.62 | 35.07 | 11.08 | 58.89 | 18.18 | 21.32 | 27.03 |
| KDA | 22.49 | 35.10 | 14.90 | 58.12 | 19.58 | 21.80 | 28.67 |
| Mamba-3 (MIMO) | 16.68 | 36.65 | 17.44 | 59.06 | 19.16 | 21.08 | 28.35 |
| **Gated DeltaNet-2** | 23.65 | 36.75 | 19.98 | 61.37 | 19.64 | 17.87 | **29.88** |

**Attention / hybrid models**

| Model | SWDE | SQuAD | FDA | TriviaQA | NQ | DROP | Avg |
|---|---|---|---|---|---|---|---|
| Transformer | 32.21 | 38.67 | 54.78 | 58.09 | 22.49 | 22.18 | 38.07 |
| Mamba-2 | 34.67 | 40.74 | 52.31 | 60.13 | 25.91 | 24.68 | 39.74 |
| Gated DeltaNet | 33.18 | 42.28 | 50.86 | 60.60 | 25.78 | 21.95 | 39.11 |
| Mamba-3 (SISO) | 35.30 | 46.42 | 54.95 | 59.54 | 25.91 | 23.96 | 41.01 |
| KDA | 39.83 | 40.10 | 53.59 | 59.89 | 25.27 | 22.18 | 40.14 |
| Mamba-3 (MIMO) | 32.33 | 44.70 | 55.31 | 59.00 | 26.26 | 23.08 | 40.11 |
| **Gated DeltaNet-2** | 41.96 | 44.70 | 54.68 | 62.38 | 26.31 | 23.63 | **42.28** |

**Primary bar to clear (recurrent, our setting):** GDN-2 recurrent —
Table-2 Avg **53.11**, Table-4 retrieval Avg **29.88**, plus the per-task
RULER frontier (esp. S-NIAH-3 89.8 / MK-NIAH-1@4K 37.8).
