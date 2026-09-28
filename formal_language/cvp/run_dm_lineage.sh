#!/usr/bin/env bash
# One full DM 1-layer CVP lineage: cold n=16 (30k steps), warm chain 32/64/128 (30k) and 256/512/1024 (60k),
# then the 1024 completion at lr 2e-5 for 26k steps (the paper's certified 1024-gate cell). ~0.7 h at n=16
# to ~days at n=1024 on one B300 (T_fix 5168).
#   GPU=0 SEED=1339 bash formal_language/cvp/run_dm_lineage.sh [n_final=1024]
set -u; source "$(dirname "$0")/../_env.sh"; S=formal_language/cvp/run_cvp.sh; SEED=${SEED:-1339}
NF=${1:-1024}
SEED=$SEED bash $S dm_cold 16
ck=$(ls -t out/statetrack_ckpts/cvp_ws_63m_mult_res_L1_tagged_m12n16_*s${SEED}*_dualmod.pt | head -1)
n=16
while [ $((n*2)) -le $NF ]; do
  n2=$((n*2)); st=30000; [ $n2 -ge 256 ] && st=60000
  SEED=$SEED bash $S dm_chain "$ck" $n2 $st
  ck=$(ls -t out/statetrack_ckpts/cvp_ws_63m_mult_res_L1_tagged_m12n${n2}_*s${SEED}*_dualmod.pt | head -1)
  [ -f "$ck" ] || { echo "stage n=$n2 saved no checkpoint"; exit 1; }
  n=$n2
done
[ $NF -ge 1024 ] && SEED=$SEED bash $S dm_chain "$ck" 1024 26000 2e-5      # low-lr completion
echo "lineage complete: $ck"
