#!/usr/bin/env bash
# Official-RULER NIAH chain for the 1.3B models: download the essay haystack, generate the official
# prompts with NVIDIA's scripts (external/RULER @ c3f5e3b), then score each checkpoint through the
# wave3d engine (K=24, the training operator) with lm/eval/gdn2_evals_gen.sh.
#   NODES="host0 host1" bash lm/eval/ruler/ruler_official_chain.sh
set -u
R=${R:-$(cd "$(dirname "$0")/../../.." && pwd)}; cd "$R"
PY=${PY:-python}
ESSAYS=external/RULER/scripts/data/synthetic/json/PaulGrahamEssays.json
[ -s "$ESSAYS" ] || (cd external/RULER/scripts/data/synthetic/json && $PY download_paulgraham_essay.py)
bash lm/eval/ruler/ruler_official_gen.sh ${RULER_SAMPLES:-500}
ls external/RULER/data_official | wc -l
export ALLOW_BUSY=1
# third-party GDN-2 checkpoint (official lit_gpt code under external/gateddeltanet-2, its own venv):
# CKPT=models/gdn2_1p3b_thirdparty/model-100b.pth PY=<venv-gdn2 python> EVAL_ARCH=gdn2lit GDN2LIT_CONFIG=gdn2_1.3B \
#   PHASES="gen_ruler_off_s gen_ruler_off_mk" PORT=29810 bash lm/eval/gdn2_evals_gen.sh
for run in ${RUNS:-hero-fw-1p3b-swa2k-s1337}; do
  CKPT=out/$run/ckpt_latest.pt PHASES="gen_ruler_off_s gen_ruler_off_mk" PORT=${PORT:-29811} bash lm/eval/gdn2_evals_gen.sh
done
echo "OFFICIAL-RULER CHAIN DONE"
