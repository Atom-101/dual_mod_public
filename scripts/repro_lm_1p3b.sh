#!/usr/bin/env bash
# 1.3B FineWeb-Edu DM run (GDN-2 recipe, 2K sliding window, seed 1337) on 4 x 8 GPUs, then the GDN-2 comparison battery.
#   NODES="host0 host1 host2 host3" bash scripts/repro_lm_1p3b.sh
set -u; cd "$(dirname "$0")/.."; export PYTHONPATH=$PWD; NODES=${NODES:?}
[ -f data/fwedu_100b/fwedu_train.bin ] || { python lm/data/tokenize_fwedu.py tokenize --file_lo 0 --file_hi 140 --workers ${WORKERS:-64}; python lm/data/tokenize_fwedu.py merge --budget_gb 100; }
SMOKE=1 bash lm/train_1p3b/launch_swa2k.sh                      # 60-step smoke on node 0 first
bash lm/train_1p3b/launch_swa2k.sh                              # production; watch out/hero-fw-1p3b-swa2k-s1337/metrics.jsonl
# after the run finishes (190,650 steps):
# CKPT=out/hero-fw-1p3b-swa2k-s1337/ckpt_latest.pt PHASES="tf_k24 siqa_k24" bash lm/eval/gdn2_evals_dist.sh
# CKPT=out/hero-fw-1p3b-swa2k-s1337/ckpt_latest.pt PHASES="gen_rc_jrt" bash lm/eval/gdn2_evals_gen.sh
# bash lm/eval/ruler/ruler_official_chain.sh
# python lm/eval/gdn2_collate.py hero-fw-1p3b-swa2k-s1337
