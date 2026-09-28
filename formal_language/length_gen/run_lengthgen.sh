#!/usr/bin/env bash
# C-RASP length-generalization protocol (Yang et al.): train on lengths 2-50, evaluate at 1x..10x (500) and,
# for the 32x cells, to 1600 (paper Table "lengthgen", Fig. "lengthgen"). Seven languages: z2 (parity), z5, z10,
# a5, flipflop (with the separator), contains_a, contains_ab. 63m scale, 12k steps, 10-25 min per cell.
#   GPU=0 bash formal_language/length_gen/run_lengthgen.sh dm z2 [seed=1337]   # arms: dm dm_nope tpp tpp_nope lstm gdn fbt ag kt hug dm_logn
#   bash formal_language/length_gen/run_lengthgen.sh all                          # the full 9-arm x 7-language grid
set -u; source "$(dirname "$0")/../_env.sh"
cell(){ a=$1; lg=$2; s=${3:-1337}
  D="--task dfa --dfa $lg --n_ops 50 --len_min 2 --gen_factors ${GF:-1,2,4,6,8,10} --scale 63m --steps 12000 --bsz 64 --seed $s --eval_dump"
  [ "$lg" = flipflop ] && D="$D --dfa_sep"
  case $a in
    dm)      $PY $H $D --arch dualmod --ctx_mod mult_res --lr 4e-4 ;;
    dm_nope) $PY $H $D --arch dualmod --ctx_mod mult_res --lr 4e-4 --nope ;;
    dm_logn) $PY $H $D --arch dualmod --ctx_mod mult_res --lr 4e-4 --attn_logn_ref 50 ;;     # log-n logit scale (parity, 7 seeds 1337-1343)
    tpp)     $PY $H $D --arch vanilla --lr 4e-4 ;;
    tpp_nope) $PY $H $D --arch vanilla --lr 4e-4 --nope ;;
    lstm)    $PY $H $D --arch lstm --n_layers 4 --lr 4e-4 ;;
    gdn)     $PY_GDN $H $D --arch gdn --lr 4e-4 ;;
    fbt)     $PY $H $D --arch fbt --n_layers 1 --lr 4e-4 ;;
    ag)      $PY $H $D --arch looped --loop_layers 2 --loop_k 16 --loop_k_min 4 --eval_loop_ks 8,16,32,64 --lr 1e-4 ;;
    kt)      $PY $H $D --arch looped --loop_layers 2 --loop_k 256 --loop_k_per_ops 2 --lr 1e-4 ;;
    hug)     $PY $H $D --arch looped --loop_layers 2 --loop_k 128 --loop_huginn --loop_emb_scale --eval_loop_ks 16,32,64,128 --lr 1e-4 ;;
    *) echo "unknown arm $a"; exit 1 ;;
  esac; }
if [ "${1:-}" = all ]; then for lg in z2 z5 z10 a5 flipflop contains_a contains_ab; do for a in dm dm_nope tpp tpp_nope lstm gdn fbt ag kt hug; do cell $a $lg; done; done
else cell "${1:?arm}" "${2:?language}" "${3:-1337}"; fi
# 32x cells: GF=1,2,4,8,10,16,20,32 bash formal_language/length_gen/run_lengthgen.sh dm flipflop
