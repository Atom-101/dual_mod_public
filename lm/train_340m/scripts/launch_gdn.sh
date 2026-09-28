#!/usr/bin/env bash
# Gated DeltaNet 340M baseline (lm/train_340m/gdn340m_config.json: 4 heads x head_dim 256, gated; 334.0M non-embedding)
# on the same token stream and recipe. Needs flash-linear-attention (pip install flash-linear-attention==0.5.1,
# with transformers 4.57 and tilelang on Hopper; see README "GDN environment").
#   GPUS=0,1,2,3,4,5,6,7 bash lm/train_340m/scripts/launch_gdn.sh
set -u; source "$(dirname "$0")/_common.sh"
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/triton_cache_gdn}
RUN=${RUN:-PDN-GDN-s42}; EXTRA=${EXTRA:-}
[ "${SMOKE:-0}" = 1 ] && { RUN=${RUN}-smoke; EXTRA="$EXTRA --total_steps 30 --warmup_tokens 2e5 --global_batch $((8*NG)) --eval_interval 10 --ckpt_interval 20"; }
$PY -m torch.distributed.run --nproc_per_node=$NG --master_port=$PORT lm/train_340m/train_baseline.py \
  --run_name $RUN --arch gdn --recipe pdn --seed 42 --data_bin $DATA_BIN --wandb_log ${WANDB_LOG:-false} $EXTRA 2>&1 | tee out/$RUN.launch.log
