#!/usr/bin/env bash
# A5 word problem, tagging protocol, T=64, evaluated in-distribution and at 2x / 4x length (paper Table "a5",
# Fig. "a5pos"). Every arm at d=768 (63m scale; looped models at d=1280 = w1280) and at its best learning rate.
#   GPU=0 bash formal_language/a5/run_a5_T64.sh dm1          # one arm; arms listed below
#   bash formal_language/a5/run_a5_T64.sh all                  # sequentially
set -u; source "$(dirname "$0")/../_env.sh"
A5="--task group --group a5 --format tagged --n_ops 64 --gen_factors 1,2,4 --steps 12000 --bsz 64"
arm(){ case $1 in
  dm1)   $PY $H $A5 --arch dualmod --ctx_mod mult_res --n_layers 1 --scale 63m --lr 4e-4 --seed ${SEED:-1338} --eval_dump ;;   # DM 1L 12M: 1.0 / 0.998 / 0.86
  dm9)   $PY $H $A5 --arch dualmod --ctx_mod mult_res --scale 63m --lr 4e-4 --curriculum --loss_guard --seed ${SEED:-50} --eval_dump ;;   # DM 9L 106M
  tpp)   $PY $H $A5 --arch vanilla --scale 63m --lr 1e-4 --seed ${SEED:-1343} --eval_dump ;;      # Transformer++ 9L 64M (8 seeds 1337-1344 were run; 5/8 form)
  gdn)   $PY_GDN $H $A5 --arch gdn --scale 63m --lr 1e-3 --curriculum --eval_dump ;;               # GDN 9L 56M
  lstm)  $PY $H $A5 --arch lstm --scale 63m --lr 1e-3 --curriculum --eval_dump ;;                  # LSTM 4L 19M
  fbt1)  $PY $H $A5 --arch fbt --n_layers 1 --scale 63m --lr 4e-4 --eval_dump ;;                   # full-bandwidth Tr. 1L 8M (exact sequential training, ~1.3 s/step)
  fbt1_32x) $PY $H --task group --group a5 --format tagged --n_ops 64 --gen_factors 1,2,4,8,16,32 --steps 12000 --bsz 64 --arch fbt --n_layers 1 --scale 63m --lr 4e-4 --eval_dump ;;
  ag)    $PY $H --task group --group a5 --format tagged --n_ops 64 --gen_factors 1,2,4 --steps 24000 --bsz 64 --scale w1280 --lr 1e-4 --curriculum \
           --arch looped --loop_layers 2 --loop_k 16 --loop_k_min 4 --eval_loop_ks 8,16,32,64 --eval_dump ;;           # iteration-agnostic loop, 59M (table row = eval K=32)
  kt)    $PY $H --task group --group a5 --format tagged --n_ops 64 --gen_factors 1,2,4 --steps 24000 --bsz 64 --scale w1280 --lr 1e-4 --curriculum \
           --arch looped --loop_layers 2 --loop_k 64 --loop_k_per_ops 4 --eval_dump ;;                                 # K proportional to length (Fan et al.), 59M
  hug)   $PY $H $A5 --scale w1280 --lr 1e-4 --curriculum --arch looped --loop_layers 2 --loop_k 128 --loop_huginn --loop_emb_scale \
           --eval_loop_ks 16,32,64,128 --seed ${SEED:-1337} --eval_dump ;;                                             # recurrent-depth (Geiping et al.), 82M, with the sqrt(d) embedding scale
  hug1)  $PY $H $A5 --scale w1280 --lr 1e-4 --curriculum --arch looped --loop_layers 1 --loop_k 128 --loop_huginn --loop_emb_scale \
           --eval_loop_ks 16,32,64,128 --seed ${SEED:-1337} --eval_dump ;;                                             # recurrent-depth, 1 block
  hug_noscale) $PY $H $A5 --scale w1280 --lr 1e-4 --curriculum --arch looped --loop_layers 2 --loop_k 128 --loop_huginn \
           --eval_loop_ks 16,32,64,128 --seed ${SEED:-1338} --eval_dump ;;                                             # the submitted-paper variant (no embedding scale)
  *) echo "unknown arm $1"; exit 1 ;; esac; }
if [ "${1:-}" = all ]; then for a in dm1 dm9 tpp gdn lstm fbt1 ag kt hug; do arm $a; done; else arm "${1:?arm}"; fi
