#!/usr/bin/env bash
# Minimum layers to solve A5 at short lengths T in {5,10,15,20} (right half of the paper's Table "a5"):
# shallowest depth at which any seed reaches accuracy >= 0.95 in-distribution at 12k steps.
#   GPU=0 bash formal_language/a5/run_a5_minlayers.sh dm 10 1       # arm, T, layers
#   bash formal_language/a5/run_a5_minlayers.sh grid                  # the whole grid, sequentially
set -u; source "$(dirname "$0")/../_env.sh"
cell(){ a=$1; T=$2; L=$3
  C="--task group --group a5 --format tagged --scale 63m --lr 4e-4 --steps 12000 --seed ${SEED:-1337} --n_ops $T --gen_factors 1"
  case $a in
    dm)     $PY $H $C --arch dualmod --ctx_mod mult_res --n_layers $L ;;
    tpp)    $PY $H $C --arch vanilla --n_layers $L ;;
    lstm)   $PY $H $C --arch lstm --n_layers $L ;;
    gdn)    $PY_GDN $H $C --arch gdn --n_layers $L ;;
    fbt)    $PY $H $C --arch fbt --n_layers $L ;;
    loopag) $PY $H $C --arch looped --loop_layers $L --loop_k 16 --loop_k_min 4 --eval_loop_ks 16 ;;
    loopkt) $PY $H $C --arch looped --loop_layers $L --loop_k 64 --loop_k_per_ops 1 ;;
    hug)    $PY $H $C --arch looped --loop_layers $L --loop_k 128 --loop_huginn --loop_emb_scale --eval_loop_ks 16,32,64,128 ;;
  esac; }
if [ "${1:-}" = grid ]; then for T in 5 10 15 20; do for a in dm tpp lstm gdn fbt loopag loopkt hug; do for L in 1 2 3 4 6 9; do cell $a $T $L; done; done; done
else cell "${1:?arm}" "${2:?T}" "${3:?layers}"; fi
