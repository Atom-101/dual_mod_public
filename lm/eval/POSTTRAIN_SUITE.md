# Standard post-training eval battery — run ALL of these for every arm

One command: `bash tools/run_posttrain_suite.sh <ckpt> <gpu_list>` (see below).
GDN-arch checkpoints run under `.venv-fla` (the launcher dispatches by the
ckpt's `arch` field). All outputs land in `analysis/` with run+step+split in
the filename. Always use absolute paths when launching over ssh.

## A. LM quality (ppl family)
1. **Held-out val CE** — `heldout_loss.py --val data/slimpj_val.pt --n_batches 21`
2. **Reserved-test CE** — same with `data/slimpj_test.pt` (final-table number;
   never used for any decision during training)
3. **WikiText word ppl** — `run_lm_eval.py --tasks wikitext --batch_rows 32`
4. **LAMBADA ppl+acc** — part of A/B likelihood job below

## B. Accuracy suite (fla-seven likelihood protocol)
5. `--tasks lambada_openai,piqa,hellaswag --batch_rows 64`
6. `--tasks winogrande,arc_easy,arc_challenge --batch_rows 64`
   (Absolute logprobs require the joint-encode continuation fix — in the
   adapter since ef59f07; never score split-tokenized continuations.)

## C. Recall — few-shot natural documents (Based-lineage; the discriminating
   recall rows for pretrained models)
7. `--tasks swde,fda,squad_completion` (task-default shots)
8. `--tasks nq_open,triviaqa --num_fewshot 5`
9. `--tasks drop --num_fewshot 3 --limit 2000`

## D. Recall — MQAR (all three variants)
10. **Zero-shot curve** — `mqar.py --pairs 8,16,32,64 --n_seq 64`
    (associative recall EMERGES from NTP; the acc-vs-pairs decay separates
    fixed-state from attention-family — post ea25cbf scorer only)
11. **Post-trained** — `posttrain_mqar.py` (masked value-CE @ 1e-4, 1500
    steps; certifies recall trainability; does NOT discriminate at flagship
    state sizes — all arms reach ~1.0 at ≤100 pairs)
12. **From-scratch capacity** (per-architecture, optional) —
    `train_mqar.py --arch <a> --pairs 16,64,110`; only claim within-arch
    walls unless per-arch hyperparameters are tuned (DualModLM-family learns
    slowly at zoology scale under the default recipe)

## E. Length extrapolation — RULER/NIAH
13. `--tasks niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2 \
     --ruler_lengths 2048,4096,8192 --ruler_samples 100 --ctx_len 8448`
    (raw RoPE extrapolation — ctx_len rebuilds tables, no scaling; report
    format-rate alongside accuracy; needs wonderwords+nltk in the venv)

## F. Mechanism studies (DM checkpoints only, over training snapshots)
14. **Gate/register/bias + DEQ depth grid** — `wavescan/bench/fs_analysis.py
    --snaps <all>` sharded across GPUs (gate opening trajectory, register
    fracs per layer, bias inertness, ballistic depth demand m∈{1..64},
    Jacobi oscillation)
15. **Chunked (c,K) residual grid** — `wavescan/bench/fs_ck_grid.py --steps
    <key snaps>` (the frontier instrument that matters at scale; parallel-m
    grids are uninformative once depth demand saturates)

## Gotchas that already burned us once
- GDN evals under `.venv-fla` (transformers 4.57 + fla + tilelang) with
  absolute `--tok_dir`; main venv for everything else.
- Don't run evals on a GPU hosting a training rank-0 (eval spikes to 75GiB).
- heldout output filenames include the val-file name (val/test used to
  collide).
- `--limit` on RULER-style multi-length tasks truncates the FIRST length
  only — use `--ruler_samples`, never `--limit`.
- lm-eval `wikitext` rolling windows are batched across docs in the adapter
  (latency-bound scan) — do not revert to per-window scoring.
