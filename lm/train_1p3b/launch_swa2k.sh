#!/bin/bash
# hero-fw-1p3b-swa2k-s1337: DM 1.3B / 100B FineWeb-Edu with the RAW-DIAGONAL operator (exact by
# nilpotency at K=C) + 2K sliding-window attention (GDN-2 SWA-hybrid convention: W tokens incl. self).
# Ladder (user 2026-09-08 "K8 to 50k, k16 to 140k and then k24"): K8 -> 50k, K16 -> 140k, K24 -> end,
# probes one rung up (K16 / K24 / K32) at 10% / 10% / 30% (mirrors the finished w3dp run's probe policy).
#
#   SMOKE=1 bash lm/train_1p3b/launch_swa2k.sh           # 1 node x 8 GPU, 60 steps, all rungs + K64 val
#   bash lm/train_1p3b/launch_swa2k.sh                   # 4 nodes x 8 GPU production launch (per-node torchrun)
set -uo pipefail
R=${R:-$(cd "$(dirname "$0")/../.." && pwd)}
cd "$R"
NODES=(${NODES:?set NODES="host0 host1 host2 host3" (passwordless ssh + shared repo mount)})
TOTAL=190650
export LADDER_K=${LADDER_K:-8,16,24,32}
export PROBE_MULT=${PROBE_MULT:-0}
export PROBE_FRAC=${PROBE_FRAC:-0.1,0.1,0.3}
export ENGINE_KWARGS=${ENGINE_KWARGS:-'{"graphs": true}'}
OP_ARGS="--raw_diag 1 --local_window ${LOCAL_WINDOW:-2048}"
if [ "${SMOKE:-0}" = 1 ]; then
  export RUN_NAME=${RUN_NAME:-swa2k-smoke-$(date +%m%d%H%M)}
  export NNODES=1 GLOBAL_BATCH=${GLOBAL_BATCH:-32} TOTAL_STEPS=${TOTAL_STEPS:-60}
  export PHASE_BOUNDS=${PHASE_BOUNDS:-0.3,0.6,2.0} PROBE_START=${PROBE_START:-5}
  export EXTRA_ARGS="$OP_ARGS --eval_interval 20 --tel_interval 20 --ckpt_interval 30 --snap_interval 30 --wandb_log false --wandb_tags smoke,swa2k ${EXTRA_ARGS:-}"
  n=${SMOKE_NODE:-${NODES[0]}}
  echo "[swa2k] SMOKE on $n: RUN_NAME=$RUN_NAME steps=$TOTAL_STEPS ladder=$LADDER_K bounds=$PHASE_BOUNDS probe_frac=$PROBE_FRAC $OP_ARGS"
  timeout 40 ssh -n -o ConnectTimeout=20 "$n" "cd $R && NODE_RANK=0 MASTER=$n PORT=${PORT:-29511} NNODES=1 RUN_NAME=$RUN_NAME GLOBAL_BATCH=$GLOBAL_BATCH TOTAL_STEPS=$TOTAL_STEPS LADDER_K=$LADDER_K PHASE_BOUNDS=$PHASE_BOUNDS PROBE_MULT=$PROBE_MULT PROBE_FRAC=$PROBE_FRAC PROBE_START=$PROBE_START ENGINE_KWARGS='$ENGINE_KWARGS' EXTRA_ARGS='$EXTRA_ARGS' setsid nohup bash $R/lm/train_1p3b/launch_hero_wave3d.sh > $R/out/_swa2k_smoke_n0.log 2>&1 < /dev/null & sleep 1; echo launched" || { rc=$?; [ $rc -eq 124 ] || echo "[swa2k] ssh launch on $n FAILED (rc=$rc)"; }
  echo "[swa2k] log: out/_swa2k_smoke_n0.log  metrics: out/$RUN_NAME/metrics.jsonl"
  exit 0
fi
export RUN_NAME=${RUN_NAME:-hero-fw-1p3b-swa2k-s1337}
export NNODES=${#NODES[@]} GLOBAL_BATCH=${GLOBAL_BATCH:-128} TOTAL_STEPS=$TOTAL
export PHASE_BOUNDS=${PHASE_BOUNDS:-0.26226,0.734329}       # 50000/190650, 140000/190650 (+ never-reached K32 rung)
export PHASE_BOUNDS="$PHASE_BOUNDS,2.0"
export EXTRA_ARGS="$OP_ARGS ${EXTRA_ARGS:-}"
MASTER=${NODES[0]}
echo "[swa2k] PRODUCTION: RUN_NAME=$RUN_NAME nodes=${NODES[*]} ladder=$LADDER_K bounds=$PHASE_BOUNDS probe_frac=$PROBE_FRAC $OP_ARGS"
for k in $(seq 0 $((NNODES-1))); do
  n=${NODES[$k]}
  timeout 40 ssh -n -o ConnectTimeout=20 "$n" "cd $R && NODE_RANK=$k MASTER=$MASTER PORT=${PORT:-29500} NNODES=$NNODES RUN_NAME=$RUN_NAME GLOBAL_BATCH=$GLOBAL_BATCH TOTAL_STEPS=$TOTAL_STEPS LADDER_K=$LADDER_K PHASE_BOUNDS=$PHASE_BOUNDS PROBE_MULT=$PROBE_MULT PROBE_FRAC=$PROBE_FRAC ENGINE_KWARGS='$ENGINE_KWARGS' EXTRA_ARGS='$EXTRA_ARGS' ${RESUME_PATH:+RESUME_PATH=$RESUME_PATH} setsid nohup bash $R/lm/train_1p3b/launch_hero_wave3d.sh > $R/out/_swa2k_n$k.log 2>&1 < /dev/null & sleep 1; echo launched-$k" || { rc=$?; [ $rc -eq 124 ] || echo "[swa2k] ssh launch on $n FAILED (rc=$rc)"; }
done
echo "[swa2k] logs out/_swa2k_n{0..3}.log  metrics out/$RUN_NAME/metrics.jsonl"
