#!/usr/bin/env bash
# Circuit value problem (keyed XOR/MAJ gates, 12 inputs), n gates, evaluated per depth (paper Table "keyed_cvp"
# right, Fig. "cvpdepth", Fig. "cvpminlayers"). "Exact" = accuracy >= 0.95 at every depth.
# DM cells of n >= 32 use the WaveScan engine harness; baselines run eagerly. Warm chains: n -> 2n from the
# previous solver (the chain policy fired at FINAL accuracy >= 0.95).
#   GPU=0 bash formal_language/cvp/run_cvp.sh dm_cold 16 [layers=1]            # 30k steps, cold
#   GPU=0 bash formal_language/cvp/run_cvp.sh dm_chain <ckpt> 32 [steps=30000] [lr=4e-4]   # n=32 warm (60k steps for n >= 256)
#   GPU=0 bash formal_language/cvp/run_cvp.sh tpp 16 | lstm 16 | gdn 16 | fbt1 16 | ag 16 | kt 16 | hug1 16 | hug2 16
#   GPU=0 bash formal_language/cvp/run_cvp.sh chain <arm> <ckpt> 32           # baseline warm chain (early stop 0.995)
#   GPU=0 bash formal_language/cvp/run_cvp.sh minlayers <arm> 32 <layers>     # min-layers ladder cells
set -u; source "$(dirname "$0")/../_env.sh"
CV="--task cvp --cvp_format tagged --cvp_inputs 12 --gen_factors 1,2 --scale 63m --steps 30000 --bsz 64 --curriculum --eval_dump"
case ${1:?mode} in
  dm_cold)   n=${2:?n}; L=${3:-1}
             $PY $W --task cvp --ctx_mod mult_res --bsz 64 --cvp_inputs 12 --curriculum --steps 30000 --n_ops $n --scale 63m --lr 4e-4 --n_layers $L --seed ${SEED:-1339} ;;
  dm_cold_eager) n=${2:?n}; $PY $H $CV --lr 4e-4 --n_ops $n --arch dualmod --ctx_mod mult_res --n_layers ${3:-1} --seed ${SEED:-1339} ;;
  dm_chain)  ck=${2:?ckpt}; n=${3:?n}; st=${4:-30000}; lr=${5:-4e-4}
             $PY $W --task cvp --ctx_mod mult_res --bsz 64 --cvp_inputs 12 --steps $st --n_ops $n --scale 63m --n_layers 1 --lr $lr --seed ${SEED:-1339} --init_from "$ck" ;;
  tpp)   $PY $H $CV --lr 1e-4 --n_ops ${2:?n} --arch vanilla ;;                                    # Transformer++ 9L 64M
  lstm)  $PY $H $CV --lr 4e-4 --n_ops ${2:?n} --arch lstm ;;                                       # LSTM 4L 19M
  gdn)   $PY_GDN $H $CV --lr 4e-4 --n_ops ${2:?n} --arch gdn ;;                                    # GDN 9L 56M
  fbt1)  $PY $H $CV --lr 2e-4 --n_ops ${2:?n} --arch fbt --n_layers 1 ;;                           # full-bandwidth Tr. 1L 8M
  ag)    n=${2:?n}; $PY $H $CV --lr 1e-4 --n_ops $n --arch looped --loop_layers 2 --loop_k $((n*2)) --loop_k_min $n --eval_loop_ks $n,$((n*2)) ;;
  kt)    n=${2:?n}; $PY $H $CV --lr 1e-4 --n_ops $n --arch looped --loop_layers 2 --loop_k $((n*2)) --loop_k_per_ops 1 ;;
  hug1|hug2) n=${2:?n}; NL=${1#hug}
         $PY $H $CV --lr 1e-4 --n_ops $n --arch looped --loop_layers $NL --loop_k $((n*2)) --loop_huginn --loop_emb_scale --eval_loop_ks 16,32,64,128,$((n*2)) ;;
  chain) a=${2:?arm}; ck=${3:?ckpt}; n2=${4:?n}; B=64; [ $n2 -ge 512 ] && B=16; [ $n2 -ge 1024 ] && B=4
         C="--task cvp --cvp_format tagged --cvp_inputs 12 --gen_factors 1,2 --scale 63m --steps 30000 --bsz $B --curriculum --n_ops $n2 --init_from $ck --eval_dump --early_stop 0.995 --curriculum_end 2000"
         case $a in
           tpp)  $PY $H $C --lr 1e-4 --arch vanilla ;;
           fbt1) $PY $H $C --lr 2e-4 --arch fbt --n_layers 1 ;;
           lstm) $PY $H $C --lr 2e-4 --arch lstm ;;
           gdn)  $PY_GDN $H $C --lr 1e-4 --arch gdn ;;
           ag)   $PY $H $C --lr 1e-4 --arch looped --loop_layers 2 --loop_k $((n2*2)) --loop_k_min $n2 --eval_loop_ks $n2,$((n2*2)) ;;
           kt)   $PY $H $C --lr 1e-4 --arch looped --loop_layers 2 --loop_k $((n2*2)) --loop_k_per_ops 1 ;;
           hug1|hug2) $PY $H $C --lr 1e-4 --arch looped --loop_layers ${a#hug} --loop_k $((n2*2)) --loop_huginn --loop_emb_scale --eval_loop_ks 16,32,64,128,$((n2*2)) ;;
         esac ;;
  minlayers) a=${2:?arm}; n=${3:?n}; L=${4:?layers}
         C="--task cvp --cvp_format tagged --cvp_inputs 12 --gen_factors 1,2 --scale 63m --lr 4e-4 --steps 30000 --bsz 64 --curriculum --seed ${SEED:-1337} --n_ops $n"
         case $a in
           dm)   $PY $W --task cvp --ctx_mod mult_res --bsz 64 --cvp_inputs 12 --curriculum --steps 30000 --n_ops $n --scale 63m --lr 4e-4 --n_layers $L --seed ${SEED:-1337} ;;
           tpp)  $PY $H $C --arch vanilla --n_layers $L ;;   tpp1e4) $PY $H $C --arch vanilla --n_layers $L --lr 1e-4 ;;
           lstm) $PY $H $C --arch lstm --n_layers $L ;;
           gdn)  $PY_GDN $H $C --arch gdn --n_layers $L ;;
           fbt)  $PY $H $C --arch fbt --n_layers $L --lr 2e-4 ;;
         esac ;;
  *) echo "unknown mode $1"; exit 1 ;;
esac
