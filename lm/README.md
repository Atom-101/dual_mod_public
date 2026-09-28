# Language modeling

## Data (`lm/data/`)
* `pretokenize_slimpj627.py`: SlimPajama-627B (hub `gmongaras/SlimPajama-627B_Reupload`), Mistral-32k tokenizer
  (`data/mistral32k_tok`, the `fla-hub/transformer-1.3B-100B` tokenizer), BOS + document, uint16. Writes
  `data/slimpj627_{train,val,test}.bin`, `data/slimpj627_val.pt` (21 x 48 x 2049 held-out batches) and the meta file.
  The 340M runs consume the first 15.36B tokens. `data/slimpj627_test.pt` (reserved test CE) is cut the same way from
  the test split.
* `tokenize_fwedu.py`: FineWeb-Edu `sample/100BT` (140 parquet files), Llama-2 tokenizer (`data/llama2_tok`), BOS per
  document, no EOS (the GDN-2 convention). `tokenize` per file slice, then `merge --budget_gb 100` ->
  `data/fwedu_100b/{fwedu_train.bin (99.98B tokens), fwedu_val.bin (20M tokens), fwedu_meta.json}`.
  `validate_fwedu.py <bin>` checks a stream the way the loader reads it.

## 340M (`lm/train_340m/`)
`train_flagship.py` trains DM through the WaveScan engine; `train_baseline.py` trains Transformer++
(`--arch vanilla340`, DualModLM with the modulation off) and Gated DeltaNet (`--arch gdn`, `gdn340m_config.json`).
`--recipe pdn` fixes the shared protocol: global batch 256 x T2048 (0.5M tokens/step), 30k steps, 1024 warmup
steps, AdamW(0.9, 0.95) eps 1e-8, wd 0.01, cosine to 0.1x. The DM flagship adds the K-ladder (`--ladder k`) on
c = 64 chunks, calibrated group clipping, the storm governor (`--auto_rewind`) and a DM-path learning rate of 3e-4
under the trunk's 4e-4. Launchers with the paper's exact flags are in `scripts/`:

| script | run | paper row |
|---|---|---|
| `scripts/launch_dm_d960.sh` | `PDN-DM-d960-mrk-lora8-sync-s42` | DM: d=960 x 26, mult-res write operator, rank-120 factors, k-mod head, k-ctx RMS norm, entry-norm clamp |
| `NOKMOD=1 scripts/launch_dm_d960.sh` | `PDN-DM-d960-mrk-lora8-sync-nokmod-s42` | k-mod ablation |
| `scripts/launch_tpp.sh` | `PDN-TPP-s42` | Transformer++ d=1024 x 24 |
| `scripts/launch_gdn.sh` | `PDN-GDN-s42` | Gated DeltaNet 4 heads x 256 |

Every arm uses seed 42 and the identical token stream. `SMOKE=1` runs 30 steps on the GPUs given.

## 1.3B (`lm/train_1p3b/`)
`train_hero_wave3d.py` is the production trainer on the wave3d engine: K-ladder scheduler (`--ladder_k 8,16,24,32
--phase_bounds ...`, 10% of steps probe one rung up), calibrated clipping with separate base/probe buckets, the
storm governor (skip / rewind to an anchor with re-seeded data order), two-group weight decay, val loss at the
phase's K and `val/loss_exact` at K = 64, checkpoint format = the plain per-block state dict. `launch_hero_wave3d.sh`
is the per-node torchrun command with the paper's hyper-parameters (lr 4e-4 -> 4e-5, 2000 warmup steps, wd 0.1,
clip 1.0, 0.5M-token batches, 190,650 steps = one pass over the stream); `launch_swa2k.sh` fans it out over the
nodes with the reported run's operator (`--raw_diag 1 --local_window 2048`, ladder 8 -> 16 at step 50k -> 24 at
140k, probes 10% / 10% / 30%). `smoke_local.sh` runs the same recipe for a few steps on one node.
`validate_rawdiag.py` is the exactness check of the engine at K = c against the sequential scan.

The same trainer without `--local_window` is the full-4K-attention run reported alongside (`hero-fw-1p3b-w3dp-s1337`).

## Evaluation (`lm/eval/`)
`run_lm_eval.py` + `lm_eval_adapter.py` wrap a DM checkpoint for lm-eval-harness 0.4.12: likelihoods through the
exact sequential scan (`DualModEval`) or through the wave3d engine at a chosen K (`--engine_K`, `Wave3DEval`);
generation through incremental decode (`decode_init` / `decode_step`), the CUDA-graph decoder (`--graph_decode`,
validate first with `validate_graph_decode.py`) or the engine. Custom tasks in `based_tasks/` are registered
automatically: the JRT recall protocol (`jrt_swde, jrt_fda, jrt_squad, jrt_triviaqa, jrt_nq, jrt_drop`: answer-
centred 2K context, greedy generation, `contains`), `social_iqa_pq`, and the official-RULER readers
(`ruler_off_niah_*`, scored with lm-eval's `string_match_all`).

| what | command |
|---|---|
| held-out / test cross-entropy | `python lm/eval/heldout_loss.py --ckpt <ckpt> --val data/slimpj627_val.pt --n_batches 21` |
| commonsense (fla-seven protocol), Wiki and LAMBADA ppl | `python lm/eval/run_lm_eval.py --ckpt <ckpt> --tok_dir data/mistral32k_tok --tasks lambada_openai,piqa,hellaswag,winogrande,arc_easy,arc_challenge,boolq,sciq,wikitext --batch_rows 64` |
| Based recall (FDA / SWDE / SQuAD) | `... --tasks swde,fda,squad_completion` |
| JRT recall (six tasks) | `EVAL_LEN_QUANT=1 ... --tasks jrt_swde,jrt_fda,jrt_squad,jrt_triviaqa,jrt_nq,jrt_drop` (shard with `JRT_SHARD=i/n`, pool with `merge_shards.py`) |
| MQAR zero-shot | `python lm/eval/mqar.py --ckpt <ckpt> --pairs 8,16,32,64 --n_seq 64` (`posttrain_mqar.py` = the 1500-step recall fine-tune) |
| RULER NIAH, 340M | `CKPT=<ckpt> GPUS=0,1,2,3 bash lm/eval/niah_340m.sh` (lm-eval's RULER port, 500 samples per cell, `--ctx_len 8448` raw RoPE extrapolation) |
| the whole 340M battery | `bash lm/eval/run_posttrain_suite.sh <ckpt> <gpu_csv> <tag>` |
| 1.3B commonsense / LM through the engine | `NODES=... CKPT=<ckpt> PHASES="tf_k24 siqa_k24" bash lm/eval/gdn2_evals_dist.sh` |
| 1.3B JRT recall through the engine | `NODES=... CKPT=<ckpt> PHASES="gen_rc_jrt" bash lm/eval/gdn2_evals_gen.sh` |
| 1.3B official RULER NIAH | `bash lm/eval/ruler/ruler_official_gen.sh 500` (NVIDIA generator, Llama-2 tokenizer, seed 42) then `PHASES="gen_ruler_off_s gen_ruler_off_mk" bash lm/eval/gdn2_evals_gen.sh`; `ruler/ruler_official_chain.sh` chains both |
| collate the 1.3B tables | `python lm/eval/gdn2_collate.py <run>` (targets in `docs/gdn2_table.md`) |

The official RULER generator is NVIDIA's `RULER` repository at commit `c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a`,
expected under `external/RULER` (only `scripts/data/synthetic/niah.py` and the essay download are used).
Environment knobs of the adapter (all default to the paper's protocol): `EVAL_LEN_QUANT` (1 = exact-length generation
buckets), `GEN_TRUNC` (tail | head), `EVAL_GEN_CACHE` (resumable generation cache), `JRT_CONTEXT_LENGTH` (2048),
`JRT_ANSWER_LENGTH` (50), `EVAL_ARCH` (`dualmod` | `gdn` | `gdn2lit` for the third-party GDN-2 checkpoint).
`POSTTRAIN_SUITE.md` lists the protocol notes and the pitfalls that bit us.
