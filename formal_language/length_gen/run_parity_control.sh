#!/usr/bin/env bash
# Positive control for the loop harness (appendix): running parity with bits and prefix parities interleaved
# in the stream, train 32 bits, test to 4x. The looped transformer (both regimes) reaches 1.0 at 4x where the
# single-pass Transformer gets ~0.73, reproducing the length-generalization gain reported for loops on parity
# (Fan et al. supervise only the final parity; this format is ours).
set -u; source "$(dirname "$0")/../_env.sh"
P="--task parity --n_ops 32 --gen_factors 1,2,4 --scale 63m --steps 12000 --bsz 64 --seed ${SEED:-1337}"
case ${1:-all} in
  tpp) $PY $H $P --arch vanilla --lr 4e-4 ;;
  ag)  $PY $H $P --arch looped --loop_layers 2 --loop_k 64 --loop_k_min 16 --eval_loop_ks 32,64,128 --lr 1e-4 ;;
  kt)  $PY $H $P --arch looped --loop_layers 2 --loop_k 128 --loop_k_per_ops 1 --lr 1e-4 ;;
  all) for a in tpp ag kt; do bash "$0" $a; done ;;
esac
