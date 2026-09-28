#!/usr/bin/env bash
# GDN-2 table-match eval suite, DISTRIBUTED: every phase is ONE torchrun job over all
# 4 nodes x 8 GPUs (lm-eval shards requests by rank; rank 0 writes the json).
#   Phase 1  tf_k24  : all loglikelihood tasks scored through the wave3d TRAINING engine at K=24
#   Phase 2  tf_k64  : same tasks at K=64 (exact by nilpotency at C=64)
#   Phase 3+ gen_*   : generation tasks on the exact decode path (K-independent), sharded 32-way
#                      (Based recall + QA, then RULER NIAH) -- phases run SEQUENTIALLY so the
#                      teacher-forced numbers land first and the NIAH tail gets all 32 GPUs.
# Usage:  CKPT=out/<run>/ckpt_latest.pt bash lm/eval/gdn2_evals_dist.sh
# Env:    NODES="host0 host1 host2 host3" (one entry per node; passwordless ssh + shared repo mount)  TOK=data/llama2_tok  RULER_SAMPLES=500
#         PHASES="tf_k24 tf_k64 gen_rc gen_ruler_s12 gen_ruler_s3mk"  LIMIT=<n> (smoke)  PORT=29600
set -u
R=${R:-$(cd "$(dirname "$0")/../.." && pwd)}
CKPT=${CKPT:?set CKPT=out/<run>/ckpt_latest.pt}; [[ "$CKPT" = /* ]] || CKPT="$R/$CKPT"
TOK=${TOK:-data/llama2_tok}; [[ "$TOK" = /* ]] || TOK="$R/$TOK"
RS=${RULER_SAMPLES:-500}
PORT=${PORT:-29600}
IFS=' ' read -ra NODES <<< "${NODES:?set NODES=\"host0 host1 ...\" (passwordless ssh, shared filesystem)}"
NN=${#NODES[@]}; MASTER=${NODES[0]}
HFTOK=${HF_TOKEN:-}   # optional; only gated HF datasets need it
run=$(basename "$(dirname "$CKPT")")
LIM=${LIMIT:+--limit $LIMIT}
GD=${GRAPH_DECODE:+--graph_decode}      # GRAPH_DECODE=1 after validate_graph_decode.py PASS
TF_TASKS="piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa,boolq,lambada_openai,wikitext"   # social_iqa: HF dataset script unsupported by datasets 5.0 -> run separately

declare -A ARGS=(
  [tf_k24]="--tasks $TF_TASKS --num_fewshot 0 --engine_K 24 --T_eval 2048 --batch_rows 16"
  [boolq_k24]="--tasks boolq --num_fewshot 0 --engine_K 24 --T_eval 2048 --batch_rows 16"
  [siqa_k24]="--tasks social_iqa_pq --num_fewshot 0 --engine_K 24 --T_eval 2048 --batch_rows 16"
  [siqa_k64]="--tasks social_iqa_pq --num_fewshot 0 --engine_K 64 --T_eval 2048 --batch_rows 16"
  [tf_k64]="--tasks $TF_TASKS --num_fewshot 0 --engine_K 64 --T_eval 2048 --batch_rows 16"
  [gen_rc]="$GD --tasks swde,fda,squad_completion,triviaqa,nq_open,drop --num_fewshot 0 --ctx_len 2048 --batch_rows 32"
  [gen_ruler_s12]="$GD --tasks niah_single_1,niah_single_2 --ruler_lengths 1024,2048,4096,8192 --ruler_samples $RS --ctx_len 8448 --batch_rows 16"
  [gen_ruler_s3mk]="$GD --tasks niah_single_3,niah_multikey_1 --ruler_lengths 1024,2048,4096 --ruler_samples $RS --ctx_len 4352 --batch_rows 16"
)
ORDER=${PHASES:-"tf_k24 tf_k64 gen_rc gen_ruler_s12 gen_ruler_s3mk"}

log(){ echo "[$(date -u +%H:%M:%S) gdn2_evals_dist] $*"; }
gpus_free(){ local ok=1; for n in "${NODES[@]}"; do p=$(timeout 20 ssh -n "$n" 'nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l'); [ "${p:-1}" -ne 0 ] && ok=0; done; echo $ok; }

log "ckpt=$CKPT tok=$TOK ruler_samples=$RS run=$run nodes=${NODES[*]} phases=[$ORDER]"
[ "$(gpus_free)" = 1 ] || { if [ "${ALLOW_BUSY:-0}" = 1 ]; then log "WARNING: GPUs busy on some node (ALLOW_BUSY=1, continuing)"; else log "ABORT: GPUs busy on some node (set ALLOW_BUSY=1 to co-locate)"; exit 2; fi; }
for ph in $ORDER; do
  tag="${run}_${ph}"
  a=${ARGS[$ph]}
  log "phase $ph: $a"
  for k in $(seq 0 $((NN-1))); do
    n=${NODES[$k]}; lg="$R/analysis/gdn2dist_${tag}_n${k}.log"
    timeout 40 ssh -n -o ConnectTimeout=20 "$n" "NODE_RANK=$k MASTER=$MASTER PORT=$PORT NNODES=$NN HF_TOKEN=$HFTOK PY=${PY:-python} EVAL_ARCH=${EVAL_ARCH:-} GDN2LIT_CONFIG=${GDN2LIT_CONFIG:-gdn2_1.3B} setsid nohup bash $R/lm/eval/launch_torchrun.sh lm/eval/run_lm_eval.py --ckpt $CKPT --tok_dir $TOK --tag $tag $a $LIM --log_samples > $lg 2>&1 < /dev/null & sleep 1; echo launched-$k" || log "WARNING: ssh launch on $n failed"
  done
  sleep 30
  # wait for the job (all ranks) to exit on every node
  while :; do
    alive=0
    for n in "${NODES[@]}"; do c=$(timeout 20 ssh -n "$n" "pgrep -fc 'run_lm_eva[l].py.*--tag $tag'" 2>/dev/null); alive=$((alive + ${c:-0})); done   # [l]: never match the pgrep shell itself
    [ "$alive" -eq 0 ] && break
    sleep 30
  done
  out=$(ls -t "$R"/analysis/lmeval_${run}_*"${tag}"*.json 2>/dev/null | head -1)
  if [ -n "$out" ]; then log "phase $ph DONE -> $out"; else log "phase $ph FAILED (no json); see analysis/gdn2dist_${tag}_n0.log"; grep -E "Error|Traceback" "$R/analysis/gdn2dist_${tag}_n0.log" | tail -3; fi
  PORT=$((PORT+1))
done
log "all phases finished; collate with: python lm/eval/gdn2_collate.py $run"
