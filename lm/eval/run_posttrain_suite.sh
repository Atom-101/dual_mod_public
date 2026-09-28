#!/bin/bash
# Launch the FULL standard post-training eval battery for one checkpoint.
#   bash lm/eval/run_posttrain_suite.sh <ckpt> <gpu_csv> [tag]
# e.g. bash lm/eval/run_posttrain_suite.sh out/PDN-DM-d960-mrk-lora8-sync-s42/ckpt_latest.pt 0,1,2,3,4,5 dm
# Dispatches the interpreter by the ckpt's arch field (gdn -> $PY_GDN). Jobs are
# assigned round-robin over the GPU list. Mechanism studies (DM-only,
# snapshot sweeps) are NOT launched here — see lm/eval/POSTTRAIN_SUITE.md §F.
set -e
R=${R:-$(cd "$(dirname "$0")/../.." && pwd)}
CKPT=$(realpath "$1"); IFS=',' read -ra GPUS <<< "$2"; TAG=${3:-$(basename $(dirname "$CKPT"))}
cd $R
ARCH=$(${PY:-python} -c "import torch; print(torch.load('$CKPT', map_location='cpu', weights_only=False).get('arch','dualmod'))")
PY=${PY:-python}; [ "$ARCH" = "gdn" ] && PY=${PY_GDN:-$PY}   # GDN checkpoints need the fla stack (see README)
export HF_TOKEN=${HF_TOKEN:-}   # needed only for gated HF datasets
TOK=${TOK_DIR:-$R/data/mistral32k_tok}   # 340M PDN-recipe arms (FS1-era arms used data/llama2_tok)
VAL=${VAL_PT:-$R/data/slimpj627_val.pt}
TEST=${TEST_PT:-$R/data/slimpj627_test.pt}
i=0; ngpu=${#GPUS[@]}
launch() { # launch <name> <cmd...>
  local name=$1; shift
  local gpu=${GPUS[$((i % ngpu))]}; i=$((i+1))
  CUDA_VISIBLE_DEVICES=$gpu HF_TOKEN=$HF_TOKEN nohup "$@" \
      > $R/analysis/suite_${TAG}_${name}.log 2>&1 &
  echo "[$name] gpu $gpu"
}
launch heldout_val  $PY $R/lm/eval/heldout_loss.py --ckpt $CKPT --n_batches 21 --val $VAL
launch heldout_test $PY $R/lm/eval/heldout_loss.py --ckpt $CKPT --n_batches 21 --val $TEST
launch wikitext     $PY $R/lm/eval/run_lm_eval.py --ckpt $CKPT --tasks wikitext --batch_rows 32 --tok_dir $TOK
launch likeli1      $PY $R/lm/eval/run_lm_eval.py --ckpt $CKPT --tasks lambada_openai,piqa,hellaswag --batch_rows 64 --tok_dir $TOK
launch likeli2      $PY $R/lm/eval/run_lm_eval.py --ckpt $CKPT --tasks winogrande,arc_easy,arc_challenge --batch_rows 64 --tok_dir $TOK
launch boolq_sciq   $PY $R/lm/eval/run_lm_eval.py --ckpt $CKPT --tasks boolq,sciq --batch_rows 64 --tok_dir $TOK  # 8-acc composite (6-acc + BoolQ + SciQ)
launch based        $PY $R/lm/eval/run_lm_eval.py --ckpt $CKPT --tasks swde,fda,squad_completion --tok_dir $TOK
# JRT protocol (Arora et al. 2024b): document-grounded greedy generation, exact-length batching.
# This is the recall protocol of the paper's 340M retrieval table (b); shard with JRT_SHARD=i/n over
# GPUs for the long-context tasks and pool the dumps with lm/eval/merge_shards.py.
launch jrt          env EVAL_LEN_QUANT=1 $PY $R/lm/eval/run_lm_eval.py --ckpt $CKPT --tasks jrt_swde,jrt_fda,jrt_squad,jrt_triviaqa,jrt_nq,jrt_drop --batch_rows 16 --tok_dir $TOK
launch qa5          $PY $R/lm/eval/run_lm_eval.py --ckpt $CKPT --tasks nq_open,triviaqa --num_fewshot 5 --tok_dir $TOK
launch drop         $PY $R/lm/eval/run_lm_eval.py --ckpt $CKPT --tasks drop --num_fewshot 3 --limit 2000 --tok_dir $TOK
launch mqar_zs      $PY $R/lm/eval/mqar.py --ckpt $CKPT --pairs 8,16,32,64 --n_seq 64
# mqar_post DROPPED: post-training equalizes MQAR (every arch -> 1.0 at all pair
# loads) so it is non-discriminating, and it *trains* 1500 steps (the slow tail).
# Zero-shot mqar_zs above is the discriminating recall metric. (analysis/mqar_post_*.json)
# NIAH: the paper's 340M table used 500 samples per (task,length) cell, one cell per job with a
# resumable generation cache -- see lm/eval/niah_340m.sh. The 100-sample sweep below is the quick check.
launch ruler        $PY $R/lm/eval/run_lm_eval.py --ckpt $CKPT \
    --tasks niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2 \
    --ruler_lengths 2048,4096,8192 --ruler_samples 100 --ctx_len 8448 --batch_rows 16 --tok_dir $TOK
echo "launched 12 jobs for $TAG (arch $ARCH) over GPUs $2; logs analysis/suite_${TAG}_*.log"
