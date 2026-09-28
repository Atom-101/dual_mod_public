#!/usr/bin/env bash
# 340M SlimPajama arms with the paper's seed (42) and shared token stream, then the evaluation battery.
# 8 GPUs, ~40 h per arm; the GDN arm needs the flash-linear-attention environment (PY_GDN).
#   HF_TOKEN=... bash scripts/repro_lm_340m.sh
set -u; cd "$(dirname "$0")/.."; export PYTHONPATH=$PWD
[ -f data/slimpj627_train.bin ] || python lm/data/pretokenize_slimpj627.py --target_tokens 15.5e9 --workers ${WORKERS:-32}
bash lm/train_340m/scripts/launch_dm_d960.sh
NOKMOD=1 bash lm/train_340m/scripts/launch_dm_d960.sh
bash lm/train_340m/scripts/launch_tpp.sh
PY=${PY_GDN:-python} bash lm/train_340m/scripts/launch_gdn.sh
for run in PDN-DM-d960-mrk-lora8-sync-s42 PDN-DM-d960-mrk-lora8-sync-nokmod-s42 PDN-TPP-s42 PDN-GDN-s42; do
  bash lm/eval/run_posttrain_suite.sh out/$run/ckpt_latest.pt ${GPUS:-0,1,2,3,4,5,6,7} $run; wait
  CKPT=out/$run/ckpt_latest.pt GPUS=${GPUS:-0,1,2,3,4,5,6,7} bash lm/eval/niah_340m.sh
done
