#!/usr/bin/env bash
# DM flagship, 340M / 15.36B SlimPajama tokens: the paper's 340M DM row (val CE 2.4399 / test CE 2.4946).
# d=960, 26 layers, mult-res write operator, rank-120 (d/8) LoRA factors on the contextual and gate maps,
# k-ctx RMS norm, k-mod head, entry-norm clamp; DM-path lr 3e-4 ("sync" schedule) under trunk lr 4e-4.
# Recipe (--recipe pdn): global batch 256 x T2048 = 0.5M tok/step, 30k steps, 1024 warmup, cosine to 0.1x,
# AdamW(0.9,0.95) wd 0.01, chunk c=64 with the K-ladder of the WaveScan engine. 8 GPUs, ~40 h on H100.
#   GPUS=0,1,2,3,4,5,6,7 bash lm/train_340m/scripts/launch_dm_d960.sh            # paper run
#   NOKMOD=1 bash lm/train_340m/scripts/launch_dm_d960.sh                       # k-mod ablation arm
#   SMOKE=1 GPUS=0 bash lm/train_340m/scripts/launch_dm_d960.sh                 # 30 steps, 1 GPU
set -u; source "$(dirname "$0")/_common.sh"
RUN=${RUN:-PDN-DM-d960-mrk-lora8-sync-s42}; KM=1
[ "${NOKMOD:-0}" = 1 ] && { RUN=${RUN_NOKMOD:-PDN-DM-d960-mrk-lora8-sync-nokmod-s42}; KM=0; }
EXTRA=${EXTRA:-}
[ "${SMOKE:-0}" = 1 ] && { RUN=${RUN}-smoke; EXTRA="$EXTRA --total_steps 30 --warmup_tokens 2e5 --global_batch $((8*NG)) --eval_interval 10 --ckpt_interval 20 --val_batches 2 --eager_val_interval 1000000"; }
$PY -m torch.distributed.run --nproc_per_node=$NG --master_port=$PORT lm/train_340m/train_flagship.py \
  --run_name $RUN --data_bin $DATA_BIN \
  --recipe pdn --ladder k --lr 4e-4 --dm_lr 3e-4 --dm_accum 1 \
  --group_clip calibrated --clip_mult 2.0 --wd_doctrine fs1 \
  --skip_factor 20 --seed 42 --d_model 960 --n_layers 26 \
  --gate_rank 120 --ctx_rank 120 \
  --ctx_mod mult_res --kmod_vraw_heads $KM --fused_nonlin true \
  --k_ctx_norm rms --k_gain_init 1.75 --v_clamp_tau 150 \
  --auto_rewind --skip_free_sched --snap_interval 250 \
  --wandb_log ${WANDB_LOG:-false} $EXTRA 2>&1 | tee out/$RUN.launch.log
