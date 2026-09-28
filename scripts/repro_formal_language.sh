#!/usr/bin/env bash
# End-to-end reproduction of the formal-language results with the paper's seeds, sequentially on one GPU.
# Budget: several GPU-days (the keyed K=1024 and CVP n=1024 DM lineages dominate). Run pieces in parallel by
# calling the per-folder scripts directly; every cell is independent except the warm chains.
#   GPU=0 PY=python PY_GDN=<fla interpreter> bash scripts/repro_formal_language.sh
set -u; cd "$(dirname "$0")/.."; export GPU=${GPU:-0}
F=formal_language
# A5 word problem at T=64 (Table a5 left, Fig. a5pos) and the min-layers grid (Table a5 right)
bash $F/a5/run_a5_T64.sh all
bash $F/a5/run_a5_minlayers.sh grid
# keyed A5 (Table keyed_cvp left): DM lineage to K=1024, baselines cold at K=2..32, recurrent-depth ramp
bash $F/keyed_a5/run_dm_lineage.sh 1024
for K in 2 8 16 32; do for a in tpp gdn lstm lstm303 fbt1 fbt9 ag kt hug1 hug2; do bash $F/keyed_a5/run_keyed.sh $a $K; done; done
# CVP (Table keyed_cvp right, Fig. cvpdepth): DM lineage to 1024 gates, baselines cold at 16..128 and their chains
bash $F/cvp/run_dm_lineage.sh 1024
for n in 16 32 64 128; do for a in tpp lstm gdn fbt1 ag kt hug1 hug2; do bash $F/cvp/run_cvp.sh $a $n; done; done
# C-RASP length generalization (Table lengthgen, Fig. lengthgen), and the loop positive control
bash $F/length_gen/run_lengthgen.sh all
bash $F/length_gen/run_parity_control.sh all
echo "formal-language reproduction complete; FINAL lines in analysis/, checkpoints in out/statetrack_ckpts/"
