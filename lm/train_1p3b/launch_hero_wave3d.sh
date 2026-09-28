#!/usr/bin/env bash
# Per-node launcher for the DM 1.3B hero run on the wave3d LAYER-BATCHED 3D
# WAVEFRONT engine (4 nodes x 8 = 32 GPU), FULL FS STAIRCASE ladder:
#   LADDER_K=[8,16,32], PHASE_BOUNDS=[0.075,0.70]; 10% probes ONE rung up
#   (phase0 K8->K16, phase1 K16->K32, phase2 K32 flat, NO probes);
#   val/loss @ phase base K, val/loss_exact @ K64; gate_bias 0.0.
# Run ON EACH worker node with a distinct NODE_RANK (0..3); node 0 is master and
# owns wandb + checkpoints.
#
#   node 0:  NODE_RANK=0 MASTER=<node0 hostname> bash lm/train_1p3b/launch_hero_wave3d.sh
#   node 1:  NODE_RANK=1 MASTER=<node0 hostname> bash lm/train_1p3b/launch_hero_wave3d.sh
#   node 2:  NODE_RANK=2 MASTER=<node0 hostname> bash lm/train_1p3b/launch_hero_wave3d.sh
#   node 3:  NODE_RANK=3 MASTER=<node0 hostname> bash lm/train_1p3b/launch_hero_wave3d.sh
#
# Engine knobs pass straight through: ENGINE=vmap|x, ENGINE_KWARGS='<json>',
# ENGINE_COMPILE=none|refine|all (see engines/wave3d/wave3d.py make_engine).
# DEFAULTS = the PRODUCTION config validated by validate_wave3d.py: Wave3DX,
# all bodies compiled, CUDA-graph mode (captured per K on first use, ~30-50 s
# one-time each; defer_dw + cudnn_fe are the engine defaults). Override only
# for A/B (ENGINE=vmap ENGINE_COMPILE=none ENGINE_KWARGS='{}').
# Smoke overrides: NNODES / GLOBAL_BATCH / EXTRA_ARGS (e.g. EXTRA_ARGS='--total_steps
# 40 --warmup_steps 6 --phase_bounds 0.2,0.8 --wandb_tags smoke').
set -euo pipefail

R=${R:-$(cd "$(dirname "$0")/../.." && pwd)}
cd "$R"                                  # torchrun target is a repo-relative path
export PYTHONPATH=$R
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# system cuDNN 9.15 on LD_LIBRARY_PATH (~/.bashrc) shadows torch's bundled 9.20 -> cudnn_fe raises; use torch's
unset LD_LIBRARY_PATH

# TOTAL_STEPS: one pass over fwedu_train (24.4M chunks / 128 = 190,650) == GDN-2 recipe
# (max_tokens 1e11 / 524288 = 190,734 iters; README "0.5M" is nominal). 200000 was a transcription error.
# MULT_RES_LAMBDA 1.0 (user 2026-09-03: lambda does not help the K16 explosion). LR-GATED LADDER since the
# 15000 resume (2026-09-03): deep-K bases drift into a gradient cliff at high LR (K16@4e-4 in 160-600 steps,
# K12@4e-4 in ~900; 2000-step sweep: K12 holds at <=2e-4, K16 at <=1e-4, K8 heals the drift). So:
# base K8 -> K12 at step 60,000 (= 0.3147, LR 3.3e-4; user 2026-09-04 "should hold now, let it rewind if storms")
# -> K16 at step 95,000 (= 0.498293, LR 2.2e-4; user 2026-09-04 "Kill at 95k and switch to K16. Drifting too far":
# val@K12 vs exact@K64 gap drifted -0.003 -> -0.009 over 88-93k = K12 entrenchment); probes 2x base.
# -> K32 base + K64 probes at 141,000 (user 2026-09-06) stormed 27x in 3.5k steps (K64 probes blow up) ->
# user 2026-09-06 17:40: "kill and launch from 144k ckpt. Do a K24/32 run with 30% 32 probe": rung 3 = K24 base,
# probe = NEXT rung K32 (probe_mult 0), probe_frac 0.3; the K32 rung bound 2.0 is never reached.
NNODES=${NNODES:-4}
NPROC=${NPROC:-8}
NODE_RANK=${NODE_RANK:?set NODE_RANK 0..$((NNODES-1))}
MASTER=${MASTER:?set MASTER hostname (node 0)}
PORT=${PORT:-29500}
RUN_NAME=${RUN_NAME:-hero-fw-1p3b-w3d-s1337}
SEED=${SEED:-1337}
ENGINE=${ENGINE:-x}
ENGINE_KWARGS=${ENGINE_KWARGS:-'{"graphs": true}'}
ENGINE_COMPILE=${ENGINE_COMPILE:-all}
GLOBAL_BATCH=${GLOBAL_BATCH:-128}         # 32 GPU x micro_bs 4 x T4096 = 0.524M tok/step
EXTRA_ARGS=${EXTRA_ARGS:-}

"${PY:-python}" -m torch.distributed.run \
  --nnodes="$NNODES" --nproc_per_node="$NPROC" --node_rank="$NODE_RANK" \
  --rdzv_backend=c10d --rdzv_endpoint="$MASTER:$PORT" \
  lm/train_1p3b/train_hero_wave3d.py \
    --run_name "$RUN_NAME" --seed "$SEED" \
    --total_steps "${TOTAL_STEPS:-190650}" --global_batch "$GLOBAL_BATCH" --T 4096 \
    --lr 4e-4 --min_lr 4e-5 --warmup_steps 2000 --wd 0.1 --grad_clip 1.0 \
    --probe_start "${PROBE_START:-300}" \
    --ladder_k "${LADDER_K:-8,12,16,24,32}" --phase_bounds "${PHASE_BOUNDS:-0.3147,0.498293,0.739573,2.0}" --probe_mult "${PROBE_MULT:-0}" --probe_frac "${PROBE_FRAC:-0.3}" \
    --mult_res_lambda "${MULT_RES_LAMBDA:-1.0}" \
    ${RESUME_PATH:+--resume_path "$RESUME_PATH"} \
    --gate_bias_init 0.0 \
    --engine "$ENGINE" --engine_kwargs "$ENGINE_KWARGS" --engine_compile "$ENGINE_COMPILE" \
    --data_dir "$R/data/fwedu_100b" --train_split fwedu_train --val_split fwedu_val \
    --eval_interval 500 --tel_interval 1000 --ckpt_interval 1000 --snap_interval 2000 \
    --group_clip calibrated --clip_mult 2.0 \
    --auto_rewind true --rewind_window 20 --gn_wedge 5 --loss_wedge 5 \
    --rewind_lookback 150 --rewind_ring 8 --rewind_seed_stride 1000 --rewind_max 100 \
    --wandb_log ${WANDB_LOG:-false} --resume true $EXTRA_ARGS
