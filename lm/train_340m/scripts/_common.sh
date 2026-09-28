# shared prologue for the 340M launchers (sourced)
R=${R:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
cd "$R"
PY=${PY:-python}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
NG=$(echo "$GPUS" | tr ',' '\n' | grep -c .)
PORT=${PORT:-29829}
export CUDA_VISIBLE_DEVICES=$GPUS PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export PYTHONPATH=$R
# The token stream is read by memmap; stage it on local disk for throughput if you can:
#   cp data/slimpj627_train.bin /tmp/ && export DATA_BIN=/tmp/slimpj627_train.bin
DATA_BIN=${DATA_BIN:-data/slimpj627_train.bin}
mkdir -p out
