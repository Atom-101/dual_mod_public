#!/usr/bin/env bash
# Transformer++ 340M baseline (d=1024, 24 layers, 303.6M non-embedding) on the same token stream and recipe.
# Gradient-identical for any GPU count (global batch 256 is fixed by --recipe pdn; micro-batches accumulate).
#   GPUS=6,7 bash lm/train_340m/scripts/launch_tpp.sh
set -u; source "$(dirname "$0")/_common.sh"
RUN=${RUN:-PDN-TPP-s42}; EXTRA=${EXTRA:-}
[ "${SMOKE:-0}" = 1 ] && { RUN=${RUN}-smoke; EXTRA="$EXTRA --total_steps 30 --warmup_tokens 2e5 --global_batch $((8*NG)) --eval_interval 10 --ckpt_interval 20"; }
$PY -m torch.distributed.run --nproc_per_node=$NG --master_port=$PORT lm/train_340m/train_baseline.py \
  --run_name $RUN --arch vanilla340 --recipe pdn --seed 42 --data_bin $DATA_BIN --wandb_log ${WANDB_LOG:-false} $EXTRA 2>&1 | tee out/$RUN.launch.log
