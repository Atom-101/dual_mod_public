#!/usr/bin/env bash
# 340M RULER NIAH exactly as reported (lm-eval's RULER port, 500 samples per (task, length) cell, exact-length
# generation buckets, resumable generation cache, raw RoPE extrapolation with --ctx_len 8448). One (task, length)
# unit per job; jobs are spread round-robin over the GPU list.
#   CKPT=out/PDN-DM-d960-mrk-lora8-sync-s42/ckpt_latest.pt GPUS=0,1,2,3 bash lm/eval/niah_340m.sh
# Needs: pip install wonderwords nltk (punkt_tab is downloaded on first use); the essay haystack is the HF
# dataset baber/paul_graham_essays. Lengths per task: single_1/2 -> 2048,4096,8192; single_3, multikey_1/2 ->
# 1024,2048,4096(,8192 for the table's 8K cells of single_3/multikey_1).
set -u
R=${R:-$(cd "$(dirname "$0")/../.." && pwd)}; cd "$R"; export PYTHONPATH=$R
PY=${PY:-python}; CKPT=${CKPT:?}; TOK=${TOK_DIR:-data/mistral32k_tok}
IFS=',' read -ra GPUS <<< "${GPUS:-0}"; i=0; ngpu=${#GPUS[@]}
TASKS=${TASKS:-niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multikey_2 niah_multikey_3}
LENS=${LENS:-1024 2048 4096 8192}; N=${RULER_SAMPLES:-500}
mkdir -p analysis/gencache analysis/niah
for T in $TASKS; do for L in $LENS; do
  g=${GPUS[$((i % ngpu))]}; i=$((i+1))
  EVAL_LEN_QUANT=1 EVAL_TAG=niah_${T}_${L} EVAL_GEN_CACHE=$R/analysis/gencache/niah_${T}_${L}.jsonl \
  CUDA_VISIBLE_DEVICES=$g nohup $PY lm/eval/run_lm_eval.py --ckpt $CKPT --tok_dir $TOK --tasks $T \
    --ruler_lengths $L --ruler_samples $N --ctx_len 8448 --batch_rows 16 --log_samples \
    > analysis/niah/niah_${T}_${L}.log 2>&1 &
done; done
wait; echo "done: analysis/lmeval_*niah*.json"
