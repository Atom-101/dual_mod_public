#!/usr/bin/env bash
# Loop fixed-point diagnostic (paper Fig. "fixedpoints", Tables "loopstate"/"loopdeepest"): for a trained looped
# checkpoint, the relative state change ||x_k - x_{k-1}|| / ||x_k|| and the accuracy of the readout at every
# iterate k, in-distribution and out of distribution (2x length for A5/CVP, 2x depth for keyed A5).
# The nine paper checkpoints are the A5 / keyed / CVP runs of the three loop recipes from run_a5_T64.sh (ag, kt, hug),
# run_keyed.sh (ag 8, kt 8, hug2 8) and run_cvp.sh (ag 64, kt 32, hug2 64) with --eval_dump.
#   GPU=0 bash formal_language/loop_fixed_point/run_fixed_points.sh <ckpt> a5|keyed|cvp <outprefix>
set -u; source "$(dirname "$0")/../_env.sh"
ck=${1:?ckpt}; task=${2:?a5|keyed|cvp}; out=${3:-analysis/loop_fixed_point/$(basename "$ck" .pt)}
mkdir -p "$(dirname "$out")"
FP=formal_language/loop_fixed_point/loop_fixed_point.py
$PY $FP "$ck" --K 128 --B 128 --every > ${out}_in.txt
case $task in
  a5|cvp) $PY $FP "$ck" --K 128 --B 128 --every --n_ops ${N_OOD:-128} > ${out}_ood.txt ;;
  keyed)  $PY $FP "$ck" --K 128 --B 128 --every --D ${D_OOD:-16} > ${out}_ood.txt ;;
esac
echo "traces: ${out}_in.txt ${out}_ood.txt  (render: python figures/render/render_loop_fixedpoints_3x3.py --deepest)"
