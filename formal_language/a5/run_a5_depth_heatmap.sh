#!/usr/bin/env bash
# A5 accuracy over length x depth at d=768, 12k steps (appendix heat map, Fig. "a5depth"):
# DM 1..9 layers x T in {16,32,48,64} (3 seeds) and Transformer++ at lr 1e-4 on the same grid.
set -u; source "$(dirname "$0")/../_env.sh"
for T in 16 32 48 64; do for L in 1 2 3 4 6 9; do
  for s in 1337 1338 1339; do $PY $H --task group --group a5 --format tagged --n_ops $T --gen_factors 1,2,4 --scale 63m --steps 12000 --bsz 64 --arch dualmod --ctx_mod mult_res --n_layers $L --lr 4e-4 --seed $s; done
  $PY $H --task group --group a5 --format tagged --n_ops $T --gen_factors 1,2,4 --scale 63m --steps 12000 --bsz 64 --arch vanilla --n_layers $L --lr 1e-4
done; done
