# shared prologue for the formal-language reproduction scripts (sourced)
R=${R:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$R"; export PYTHONPATH=$R
PY=${PY:-python}            # DM / Transformer++ / LSTM / looped / FBT arms
PY_GDN=${PY_GDN:-$PY}       # interpreter with flash-linear-attention for --arch gdn
H=formal_language/harness/train_statetrack.py        # eager harness (every arch)
W=formal_language/harness/train_statetrack_ws.py     # WaveScan-engine harness (DM only; use for T > ~300)
export CUDA_VISIBLE_DEVICES=${GPU:-0}
mkdir -p analysis out/statetrack_ckpts
# Results: the harness prints one FINAL line per run (in-dist accuracy, per-position/depth bins, extrapolation)
# and writes analysis/statetrack_<tag>_<arch>.json; --eval_dump also saves out/statetrack_ckpts/<tag>_<arch>.pt.
