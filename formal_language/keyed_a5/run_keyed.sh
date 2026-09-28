#!/usr/bin/env bash
# Keyed A5: K interleaved A5 registers, D=8 operations per register, queried once at the end (paper Table
# "keyed_cvp", left). "Solved" = accuracy >= 0.95 at depth 8 (depth 1 is free retrieval).
# DM runs on the WaveScan engine harness (the eager scan OOMs past T ~ 2-5k); every baseline runs eagerly.
#   GPU=0 bash formal_language/keyed_a5/run_keyed.sh dm_cold          # DM 1L 12M cold at K=8 (root of every DM lineage)
#   GPU=0 bash formal_language/keyed_a5/run_keyed.sh dm_chain <ckpt> <K_from> <K_to>   # register ramp K -> 2K, warm from <ckpt>
#   GPU=0 bash formal_language/keyed_a5/run_keyed.sh tpp 8 | gdn 8 | lstm 8 | fbt1 8 | fbt9 8 | ag 8 | kt 8 | hug1 8 | hug2 8   # baselines cold at K
#   GPU=0 bash formal_language/keyed_a5/run_keyed.sh hug_chain <ckpt> <K_from> <K_to> [blocks]  # recurrent-depth ramp
set -u; source "$(dirname "$0")/../_env.sh"
kmax(){ k=$1; m=64; while [ $m -lt $k ]; do m=$((m*2)); done; echo $m; }   # K_max ladder 64/128/256/512/1024
bsz_for(){ k=$1; if [ $k -ge 1024 ]; then echo 2; elif [ $k -ge 512 ]; then echo 4; elif [ $k -ge 256 ]; then echo 8; else echo 64; fi; }
case ${1:?mode} in
  dm_cold)   $PY $W --task keyed --group a5 --ctx_mod mult_res --scale 63m --n_layers 1 --K 8 --K_max 64 --D 8 --lr 4e-4 --steps 12000 --bsz 64 \
               --loss_guard --seed ${SEED:-102} --tag keyed_ws_cold_K8_L1_63m_s${SEED:-102} ;;
  dm_chain)  ck=${2:?ckpt}; K=${3:?K_from}; K2=${4:?K_to}
             $PY $W --task keyed --group a5 --ctx_mod mult_res --D 8 --K $K2 --K_max $(kmax $K2) --lr 4e-4 --steps 12000 --bsz 64 --loss_guard \
               --key_shuffle --newkey_init mean --K_curriculum $K --scale 63m --n_layers 1 --seed ${SEED:-$((RANDOM%1000+200))} \
               --init_from "$ck" --tag keyed_ws_chain_K${K2}_from_$(basename "$ck" _dualmod.pt) ;;
  dm9_cold)  $PY $W --task keyed --group a5 --ctx_mod mult_res --scale w1280 --D 8 --K ${2:-16} --K_max 64 --lr 4e-4 --curriculum --loss_guard --steps 12000 --bsz 64 --seed ${SEED:-1337} ;;
  tpp|gdn|lstm|lstm303|fbt1|fbt9|ag|kt|hug1|hug2)
    K=${2:?K}; KM=$(kmax $K); B=$(bsz_for $K)
    C="--task keyed --group a5 --D 8 --K $K --K_max $KM --lr 1e-4 --curriculum --steps 12000 --bsz $B --seed ${SEED:-1337} --eval_dump"
    case $1 in
      tpp)  $PY $H $C --arch vanilla --scale w1280 ;;                                     # Transformer++ 9L 176M
      gdn)  $PY_GDN $H $C --arch gdn --scale w1280 ;;                                     # GDN 9L 161M
      lstm) $PY $H $C --arch lstm --scale w1280 ;;                                        # LSTM 4L 53M
      lstm303) $PY $H $C --arch lstm --scale w3072 ;;                                     # LSTM 4L 303M
      fbt1) $PY $H $C --arch fbt --n_layers 1 --scale 63m ;;                              # full-bandwidth Tr. 1L 8M
      fbt9) $PY $H $C --arch fbt --scale w1280 ;;                                         # full-bandwidth Tr. 9L 180M
      ag)   $PY $H $C --arch looped --loop_layers 2 --loop_k 64 --loop_k_min 16 --eval_loop_ks 16,32,64 --scale w1280 ;;
      kt)   $PY $H $C --arch looped --loop_layers 2 --loop_k 64 --loop_k_per_ops 4 --scale w1280 ;;
      hug1) $PY $H $C --arch looped --loop_layers 1 --loop_k 128 --loop_huginn --loop_emb_scale --eval_loop_ks 16,32,64 --scale w1280 --early_stop 0.995 ;;
      hug2) $PY $H $C --arch looped --loop_layers 2 --loop_k 128 --loop_huginn --loop_emb_scale --eval_loop_ks 16,32,64 --scale w1280 --early_stop 0.995 ;;
    esac ;;
  hug_chain) ck=${2:?ckpt}; K=${3:?K_from}; K2=${4:?K_to}; NL=${5:-1}; B=$(bsz_for $K2)
             $PY $H --task keyed --group a5 --D 8 --curriculum --steps 12000 --arch looped --lr 1e-4 --loop_k 128 --loop_huginn --loop_emb_scale \
               --eval_loop_ks 16,32,64 --early_stop 0.995 --key_shuffle --newkey_init mean --scale w1280 --loop_layers $NL \
               --K $K2 --K_max $(kmax $K2) --bsz $B --init_from "$ck" --K_curriculum $K --tag_suffix _from$K --eval_dump ;;
  gdn_chain) ck=${2:?ckpt}; K=${3:?K_from}; K2=${4:?K_to}
             $PY_GDN $H --task keyed --group a5 --D 8 --K $K2 --K_max $(kmax $K2) --scale w1280 --lr 1e-4 --steps 12000 --bsz $(bsz_for $K2) --arch gdn \
               --key_shuffle --newkey_init mean --K_curriculum $K --init_from "$ck" --eval_dump --early_stop 0.999 --seed ${SEED:-7} ;;
  fbt_chain) ck=${2:?ckpt}; K=${3:?K_from}; K2=${4:?K_to}; B=$(bsz_for $K2); [ $K2 -ge 32 ] && B=16; [ $K2 -ge 64 ] && B=8
             $PY $H --task keyed --group a5 --D 8 --K $K2 --K_max $(kmax $K2) --scale w1280 --lr 1e-4 --steps 12000 --bsz $B --arch fbt \
               --key_shuffle --newkey_init mean --K_curriculum $K --init_from "$ck" --eval_dump --early_stop 0.995 ;;
  *) echo "unknown mode $1"; exit 1 ;;
esac
