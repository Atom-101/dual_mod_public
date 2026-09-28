#!/usr/bin/env bash
# One full DM keyed-A5 lineage: cold K=8, then the register ramp 8 -> 16 -> ... -> 1024 (each stage warm from the
# previous solver; the watcher policy in the paper chained a stage once its FINAL accuracy was >= 0.95).
# ~minutes at K=8; ~40 h at K=1024 (T_fix 8208, ~12 s/step) on one B300.
#   GPU=0 SEED=102 bash formal_language/keyed_a5/run_dm_lineage.sh [K_final=1024]
set -u; source "$(dirname "$0")/../_env.sh"; S=formal_language/keyed_a5/run_keyed.sh
KF=${1:-1024}; SEED=${SEED:-102}
SEED=$SEED bash $S dm_cold
ck=out/statetrack_ckpts/keyed_ws_cold_K8_L1_63m_s${SEED}_dualmod.pt
K=8
while [ $((K*2)) -le $KF ]; do
  K2=$((K*2)); bash $S dm_chain "$ck" $K $K2
  ck=out/statetrack_ckpts/keyed_ws_chain_K${K2}_from_$(basename "$ck" _dualmod.pt)_dualmod.pt
  [ -f "$ck" ] || { echo "stage K=$K2 did not save a checkpoint (accuracy below the chain threshold?)"; exit 1; }
  K=$K2
done
echo "lineage complete: $ck"
