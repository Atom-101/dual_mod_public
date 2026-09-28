#!/usr/bin/env bash
# TinyStories 2x2 ablation (paper Table "tinystories", Fig. tinystories): five arms on one shared token stream,
# 300M tokens = 1,144 steps of 512 x 512-token sequences, eval every 100 steps. Exact eager sequential scan for
# the DM arms (attn_mode=sequential), so it is slow (~16 s/step on 2 B300 for the DM arms): about 5 h per DM arm.
#   python -m models.dualmod.data --data_dir data          # once: TinyStories -> data/{train,val}.bin (GPT-2 BPE)
#   ARM=A GPUS=0,1 bash tinystories/run_ablation.sh         # A: Transformer++ width-matched (d=512)
#   ARM=E ... | ARM=B ... | ARM=C ... | ARM=D ...           # E: param-matched TPP d=576; B: DM; C: v-mod only; D: k-mod only
set -u
R=${R:-$(cd "$(dirname "$0")/.." && pwd)}; cd "$R"; export PYTHONPATH=$R
PY=${PY:-python}; GPUS=${GPUS:-0,1}; NG=$(echo "$GPUS" | tr ',' '\n' | grep -c .)
ARM=${ARM:?set ARM=A|E|B|C|D}
COMMON="--batch_size 512 --tokens_target 300e6 --eval_interval 100 --eval_iters 8 --telemetry_interval 100 --log_interval 10 --ckpt_interval 200 --out_dir out --wandb_log ${WANDB_LOG:-false}"
[ "${SMOKE:-0}" = 1 ] && COMMON="--batch_size $((16*NG)) --tokens_target 2e5 --eval_interval 5 --eval_iters 2 --telemetry_interval 5 --log_interval 1 --ckpt_interval 10 --out_dir out --wandb_log false"
case $ARM in
  A) ARGS="--preset A --run_name A-vanilla --seed 1337" ;;
  E) ARGS="--preset A --d_model 576 --n_heads 9 --run_name E-vanilla576 --seed 1337" ;;
  B) ARGS="--preset B --run_name B-ginit0 --gate_bias_init 0 --seed 1337" ;;
  C) ARGS="--preset C --run_name TS0-C-ddp-s2001 --gate_bias_init 0 --seed 2001" ;;
  D) ARGS="--preset D --run_name TS0-D-ddp-s2001 --gate_bias_init 0 --seed 2001" ;;
  *) echo "unknown ARM $ARM"; exit 1 ;;
esac
[ "${SMOKE:-0}" = 1 ] && ARGS="$ARGS --run_name ts-smoke-$ARM"
CUDA_VISIBLE_DEVICES=$GPUS $PY -m torch.distributed.run --nproc_per_node=$NG --master_port=${PORT:-29610} -m tinystories.train $ARGS $COMMON
