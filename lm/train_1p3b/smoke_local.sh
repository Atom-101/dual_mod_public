#!/usr/bin/env bash
# Single-node smoke of the 1.3B hero recipe (the same trainer, engine and K-ladder; 60 steps, tiny batch).
# Verifies the wave3d engine builds, walks the rungs K8 -> K16 -> K24 and evaluates val/loss_exact at K=64.
#   GPUS=0,1,2,3,4,5,6,7 bash lm/train_1p3b/smoke_local.sh         # needs data/fwedu_100b/{fwedu_train,fwedu_val}.bin
set -u
R=${R:-$(cd "$(dirname "$0")/../.." && pwd)}; cd "$R"
GPUS=${GPUS:-0}; NG=$(echo "$GPUS" | tr ',' '\n' | grep -c .)
export CUDA_VISIBLE_DEVICES=$GPUS
export LADDER_K=8,16,24,32 PROBE_MULT=0 PROBE_FRAC=0.1,0.1,0.3 ENGINE_KWARGS='{"graphs": true}'
export RUN_NAME=${RUN_NAME:-swa2k-smoke} NNODES=1 NPROC=$NG GLOBAL_BATCH=${GLOBAL_BATCH:-$NG} TOTAL_STEPS=${TOTAL_STEPS:-60}
export PHASE_BOUNDS=0.3,0.6,2.0 PROBE_START=5
export EXTRA_ARGS="--raw_diag 1 --local_window 2048 --eval_interval 20 --tel_interval 20 --ckpt_interval 30 --snap_interval 30 --wandb_tags smoke ${EXTRA_ARGS:-}"
NODE_RANK=0 MASTER=${MASTER:-127.0.0.1} PORT=${PORT:-29511} bash lm/train_1p3b/launch_hero_wave3d.sh
