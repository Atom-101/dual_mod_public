#!/usr/bin/env bash
# Per-node torchrun launcher for distributed evals (mirrors lm/train_1p3b/launch_hero_wave3d.sh).
#   NODE_RANK=<k> MASTER=<node0-hostname> PORT=29650 NNODES=4 NPROC=8 bash lm/eval/launch_torchrun.sh <script.py> [args...]
set -euo pipefail
R=${R:-$(cd "$(dirname "$0")/../.." && pwd)}
cd "$R"
export PYTHONPATH=$R
export HF_HOME=${HF_HOME:-$R/.hf_cache}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6}
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=43200 TORCH_NCCL_ENABLE_MONITORING=0 TORCH_NCCL_ASYNC_ERROR_HANDLING=1   # long rank-imbalanced generation
unset LD_LIBRARY_PATH
NNODES=${NNODES:-4}; NPROC=${NPROC:-8}
NODE_RANK=${NODE_RANK:?}; MASTER=${MASTER:?}; PORT=${PORT:-29650}
exec "${PY:-python}" -m torch.distributed.run \
  --nnodes="$NNODES" --nproc_per_node="$NPROC" --node_rank="$NODE_RANK" \
  --rdzv_backend=c10d --rdzv_endpoint="$MASTER:$PORT" "$@"
