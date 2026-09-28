"""PRODUCTION trainer for the DM 1.3B hero run (FineWeb-Edu 100B) on the
LAYER-BATCHED 3D WAVEFRONT engine (engines/wave3d/wave3d.py), with the FULL FS
STAIRCASE K-ladder restored.

Derived from train_hero_chunked.py (the GraphedShared production trainer): the
ENGINE is swapped and the staircase ladder + probes are restored; EVERYTHING
else — the FS-pretrain skip rule, the calibrated probe-aware clip, the ENHANCED
storm/rewind governor, the 2-group wd doctrine, the recipe, the telemetry, the
checkpoint format — is carried over verbatim.

Engine (engines/wave3d/wave3d.py + wave3d_x.py, do NOT modify; constructed via
`make_engine` so its kwargs pass straight through --engine_kwargs '<json>'):
    eng  = make_engine(model, B, T, engine=<vmap|x>, C=64, **engine_kwargs)
    loss = eng.forward(idx, targets, K)      # under torch.autocast(bf16)
    loss.backward()                          # grads land on the ORIGINAL
                                             #   model.parameters() (per-block)
  K is a per-call int, time-segment checkpointing is internal. Both engines
  assert idx.shape == (B, T), so the val loader hands every rank exactly B_local
  seqs per batch (see below).

PRODUCTION ENGINE CONFIG (the defaults; validated by validate_wave3d.py):
    --engine x --engine_compile all --engine_kwargs '{"graphs": true}'
  (defer_dw and attn_impl=cudnn_fe are the Wave3DX defaults). CUDA-graph mode
  contract (see the wave3d_x.py header): graphs are captured PER K on first use
  (~30-50 s one-time each incl. compile) -> expect a slow first step at every
  new rung (K8 step 0, K16 first probe, K32 first probe) and again after a
  resume (fresh process). Captures are logged as GRAPH CAPTURE lines +
  engine/capture_K<K>_s. The eager no_grad validation forward (val/loss and the
  K64 exact reference) MUST run after backward+opt.step, never between a graphed
  forward and its backward (it would corrupt the shared caches). Memory: graph
  replays bypass the allocator, so train/step_peak_gb (max_allocated) UNDER-
  reports; train/gpu_reserved_gb (memory_reserved) is the honest footprint
  (~75 / 92 / 131 GB at K8 / K16 / K32, B4 T4096).
  ENGINE BUG GUARD (--cudnn_ws_gb, default 1 GiB): wave3d_attn.cudnn_fe's single
  growable workspace is re-allocated by the first bigger K, invalidating the
  address every earlier K's graphs baked -> replaying the base K after the first
  probe was an illegal memory access (S1/S2 smokes, repro in
  out/_w3d_trainer_scratch/repro_multik.py). build_engine pre-grows it once
  before any capture; remove once wave3d_attn.py sizes/keys it per K.

Schedule (FROZEN — the FS staircase, K64 probe DROPPED):
    LADDER_K = [8, 16, 32], PHASE_BOUNDS = [0.075, 0.70] (fractions of total
    optim steps)
    phase 0 (prog <  0.075)          base K8    probe K16
    phase 1 (0.075 <= prog < 0.70)   base K16   probe K32
    phase 2 (prog >= 0.70)           base K32   NO probes (final rung = K32 flat)
  Probes: step%10==3, step>=probe_start (default 300), ONE rung up, is_probe=True (separate gn/clip
  EMA buckets). val/loss at the CURRENT phase base K; val/loss_exact at K=64
  (exact by nilpotency at C=64; eng.forward under no_grad); val/base_minus_k64
  = the entrenchment guard. K / phase / is_probe logged EVERY step.

DDP: MANUAL (no DistributedDataParallel wrapper — the engine's custom autograd
path must not be wrapped). After backward, grads on model.parameters() are
flattened into a few large fp32 buckets, dist.all_reduce(AVG)'d and unflattened
BEFORE the grad-norm / skip / clip / step logic (exactly where the chunked
trainer all-reduced its engine.grads.flats). NCCL all-reduce output is
rank-identical -> identical gn -> identical skip/rewind decisions on all ranks.

Recipe (GDN-2 match): fused AdamW (fp32 moments), betas (0.9,0.95), wd 0.1
(2-D only; 1-D params incl. DM gains -> wd 0), LR 4e-4 cosine -> 4e-5 with 2000
warmup, 200000 steps, global batch 128 seq (32 ranks x micro_bs 4 x T4096 =
0.524M tok/optim step), gate_bias_init 0.0.

Checkpoint format: model.state_dict() of the ORIGINAL per-block model (+ opt,
step, applied, EMA buckets, tainted) -> existing eval tooling works unchanged.

Launch (per node; see lm/train_1p3b/launch_hero_wave3d.sh):
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=<repo root> \
  python -m torch.distributed.run --nnodes=4 --nproc_per_node=8 \
      --node_rank=$NODE_RANK --rdzv_backend=c10d --rdzv_endpoint=$MASTER:29500 \
      lm/train_1p3b/train_hero_wave3d.py --run_name hero-fw-1p3b-w3d-s1337 --seed 1337
"""

import argparse
import collections
import json
import math
import os
import shutil
import statistics
import sys
import threading
import time
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

R = os.environ.get("REPO_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, R)

from models.dualmod.data import PackedLoader
from models.dualmod.rope import apply_rope
from engines.wave3d.config_1p3b import build_config
from engines.wave3d.wave3d import make_engine
from models.dualmod.model import DualModLM

# ---- STAIRCASE K-LADDER + 10% anti-entrenchment PROBES (FROZEN) ---------------
# The FS staircase: the BASE K steps UP over training (the deep operator is
# entrenched progressively), with a 10% probe one rung DEEPER to keep the next
# operator's gradient alive (the fixed-K8 run ENTRENCHED — val@K8 drifted from
# val@K64-exact 0.0004 -> 1.07 once the probes were dropped and the deep grad
# path died). The K64 probe is DROPPED: the final rung is K32 FLAT (phase 2 has
# NO probes). sid_of returns the RUNG INDEX into LADDER_K, a PURE function of
# step so every DDP rank picks the same K (no collective stalls).
LADDER_K = [8, 16, 32]                  # base rungs (phase i -> LADDER_K[i])
PHASE_BOUNDS = [0.075, 0.70]            # prog fractions -> 3 phases (overridable
#                                         via --phase_bounds for SMOKE only)
K_EVAL_EXACT = 64                       # val/loss_exact reference (entrench guard)
PROBE_MULT = None                       # if set (e.g. 2): probe K = PROBE_MULT * base K (FS "2x rung");
PROBE_FRAC = 0.1                        # fraction of steps that are probes (0.1 = legacy step%10==3)
                                        # else probe = next ladder rung (only while one exists)
CHUNK_C = 64
CUDNN_WS_GB_DEFAULT = 1.0               # measured need: 18/37/73 MiB at K8/K16/K32 (K64 val
#                                         no larger); 1 GiB = 14x headroom (see build_engine)


def phase_of(step, total):
    """Phase index 0/1/2 from prog = step/total against PHASE_BOUNDS."""
    prog = step / max(1, total)
    return sum(1 for b in PHASE_BOUNDS if prog >= b)


def sid_of(step, total, probe_start):
    """(rung index into LADDER_K, is_probe). Base = the phase rung; 10% probe
    steps (step%10==3, step >= probe_start) run ONE rung UP with is_probe=True —
    only while a higher rung EXISTS (phase 2 = top rung K32 -> NO probes).
    probe_start is DECOUPLED from the LR warmup: FS gated probes at
    `step >= warmup_steps`, but FS's warmup was ~318 steps (0.5B tok). Our LR
    warmup is 2000 steps; gating probes on it let the K8 base ENTRENCH during
    warmup (val@K8 vs exact@K64 gap 0.17@1500 -> 0.37@2000) and the first probes
    at 2003 then hit a specialized model with ~40x grad norms -> STORM -> rewind
    loop (run hero-fw-1p3b-w3d-s1337, 2026-09-02). Default probe_start=300 =
    FS's absolute gate."""
    ph = phase_of(step, total)
    pf = PROBE_FRAC[min(ph, len(PROBE_FRAC) - 1)] if isinstance(PROBE_FRAC, (list, tuple)) else PROBE_FRAC
    if pf == 0.1:
        is_p = (step % 10 == 3)                       # legacy exact schedule (reproducible)
    else:                                             # exactly pf of steps, evenly spread
        is_p = int(step * pf) != int((step - 1) * pf)
    if step >= probe_start and is_p and (PROBE_MULT or ph + 1 < len(LADDER_K)):
        return ph, True
    return ph, False


def k_of(ph, is_probe):
    """K for (phase, is_probe): base rung, or its probe (PROBE_MULT*base, else next rung)."""
    if not is_probe:
        return LADDER_K[ph]
    return LADDER_K[ph] * PROBE_MULT if PROBE_MULT else LADDER_K[ph + 1]


def lr_at(step, total, warmup_steps, lr, min_lr):
    if step < warmup_steps:
        return lr * (step + 1) / max(1, warmup_steps)
    prog = (step - warmup_steps) / max(1, total - warmup_steps)
    prog = min(1.0, prog)      # skip-extension steps past `total` stay at min_lr (FS semantics)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * min(prog, 1.0)))


# ---- parallel-sweep forward WITH telemetry collect (verbatim from chunked) ------
# Mirrors the earlier wavefront engine's sweep_attn / block_fwd EXACTLY, but threads
# the model's `collect` dict into refine_kv on the CONVERGED sweep so the SAME
# per-layer §10 telemetry FS logs (g_k/g_v, ratio, vpp, ...) populates from the
# actual training operator (the K-sweep), not a separate sequential scan.

def sweep_attn_collect(blk, xhat, cos, sin, K, collect):
    a = blk.attn
    k_flat = a.wk(xhat)
    v_flat = a.project_v(xhat)
    q_r = apply_rope(a._split(a.wq(xhat)), cos, sin)
    K_cache = apply_rope(a._split(k_flat), cos, sin)
    V_cache = a._split(v_flat)
    o = None
    for s in range(K):
        o = F.scaled_dot_product_attention(q_r, K_cache, V_cache, is_causal=True)
        o_cat = a._merge(o)
        if s < K - 1:
            # collect ONLY on the last refine (converged sweep) -> one dict entry
            # per layer, matching FS's once-per-position refine telemetry.
            c = collect if s == K - 2 else None
            kpp, vpp = a.refine_kv(xhat, o_cat, k_flat, v_flat, collect=c)
            K_cache = apply_rope(a._split(kpp), cos, sin)
            V_cache = a._split(vpp)
    return a._merge(o)


def block_fwd_collect(blk, x, cos, sin, K, collect):
    xhat = blk.attn_norm(x)
    o_cat = sweep_attn_collect(blk, xhat, cos, sin, K, collect)
    x = x + blk.attn.wo(o_cat)
    x = x + blk.mlp(blk.mlp_norm(x))
    return x


@torch.no_grad()
def gate_stats_layerwise(model, x, K):
    """FS §10 telemetry (ported verbatim from train_flagship/train_hero_soft) —
    per-layer gate means + register/low fractions, ratio (ctx/raw) + eff ctx
    share, v''-tail (p999/max), ctx attn-logit tails (absent on the SDPA path),
    and k_gain (gamma) spread. Runs ONE eager parallel-sweep collect forward at
    the given K on a small fixed batch; no kernel/engine touch."""
    was_train = model.training
    model.eval()
    collect = [dict() for _ in model.blocks]
    cos, sin = model.rope_cos[:x.shape[1]], model.rope_sin[:x.shape[1]]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        xx = model.tok_emb(x)
        for i, blk in enumerate(model.blocks):
            xx = block_fwd_collect(blk, xx, cos, sin, K, collect[i])
    out = {}
    hd = model.blocks[0].attn.head_dim
    # --- v'' tail (clamp-tau gauge) ---
    vn_p999, vn_max = [], []
    for li, c in enumerate(collect):
        if "vpp" not in c:
            break
        v = torch.cat(c["vpp"], dim=1)
        vh = v.reshape(v.shape[0], v.shape[1], -1, hd)
        rn = vh.norm(dim=-1).flatten()
        vn_p999.append(torch.quantile(rn, 0.999).item())
        vn_max.append(rn.max().item())
        out[f"vtail/p999_L{li:02d}"] = vn_p999[-1]
    if vn_p999:
        out["vtail/p999_all_layers"] = float(np.mean(vn_p999))
        out["vtail/p999_max_layer"] = float(np.max(vn_p999))
        out["vtail/max_all_layers"] = float(np.max(vn_max))
    # --- ctx-read attn-logit tails (SDPA path does not expose these; skipped) ---
    al_p999, al_max = [], []
    for li, c in enumerate(collect):
        if "attn_logit_max" not in c:
            break
        s = torch.cat(c["attn_logit_max"])
        al_p999.append(torch.quantile(s, 0.999).item())
        al_max.append(s.max().item())
        out[f"attnlogit/p999_L{li:02d}"] = al_p999[-1]
    if al_p999:
        out["attnlogit/p999_all_layers"] = float(np.mean(al_p999))
        out["attnlogit/max_all_layers"] = float(np.max(al_max))
    # --- k_gain (gamma) gauge ---
    kg = [blk.attn.k_gain for blk in model.blocks if hasattr(blk.attn, "k_gain")]
    if kg:
        allg = torch.cat([g.detach().float().flatten() for g in kg])
        out["kgain/mean_all_layers"] = float(allg.mean())
        out["kgain/max_all_layers"] = float(allg.max())
        out["kgain/min_all_layers"] = float(allg.min())
        out["kgain/std_all_layers"] = float(allg.std())
        out["kgain/spread_all_layers"] = float(allg.max() / (allg.min() + 1e-6))
        for li, g in enumerate(kg):
            out[f"kgain/mean_L{li:02d}"] = float(g.detach().float().mean())
    # --- gate stats (g_k/g_v, register frac, fraclt03, ratio, eff ctx share) ---
    for tag in ("k", "v"):
        gm, rf, lo, ratios, effs = [], [], [], [], []
        for li, c in enumerate(collect):
            if f"g_{tag}" not in c:
                break
            g = torch.cat(c[f"g_{tag}"], dim=1).float()
            if f"ratio_{tag}" in c:
                r_ = torch.cat(c[f"ratio_{tag}"], dim=1).float().mean().item()
                g_ = g.mean().item()
                ratios.append(r_)
                effs.append((1 - g_) * r_ / (g_ + (1 - g_) * r_ + 1e-8))
            tok_mean = g.mean(dim=-1)
            gm.append(g.mean().item())
            rf.append(((1 - tok_mean) > 0.5).float().mean().item())
            lo.append((g < 0.3).float().mean().item())
            out[f"gates/g_{tag}_L{li:02d}"] = gm[-1]
            out[f"gates/rf_{tag}_L{li:02d}"] = rf[-1]
            out[f"gates/fraclt03_{tag}_L{li:02d}"] = lo[-1]
        if gm:
            out[f"telemetry/g_{tag}_mean_all_layers"] = float(np.mean(gm))
            out[f"telemetry/register_frac_{tag}_all_layers"] = float(np.mean(rf))
            out[f"telemetry/fraclt03_{tag}_all_layers"] = float(np.mean(lo))
        if ratios:
            out[f"telemetry/ratio_{tag}_mean"] = float(np.mean(ratios))
            out[f"telemetry/eff_ctx_share_{tag}"] = float(np.mean(effs))
    if was_train:
        model.train()
    return out


# ---- engine factory -----------------------------------------------------------
def build_engine(model, B, T, engine, engine_kwargs, compile_mode, cudnn_ws_gb=0.0):
    """Construct the wave3d engine through engines.wave3d.wave3d.make_engine, passing
    --engine_kwargs (json dict) STRAIGHT THROUGH so the engine's constructor may
    grow/rename knobs (attn_impl, sac, seg_steps, n_snapshots, compile flags, ...)
    without a trainer edit. C is pinned to 64 (the smooth c=64 operator; the
    K64 exact reference relies on it). compile_mode != 'none' -> eng.set_compile."""
    kw = dict(engine_kwargs)
    kw.setdefault("C", CHUNK_C)
    eng = make_engine(model, B, T, engine=engine, **kw)
    if compile_mode != "none":
        eng.set_compile(compile_mode)
    if cudnn_ws_gb > 0 and kw.get("attn_impl", "cudnn_fe") == "cudnn_fe" and kw.get("graphs"):
        # GRAPH-MODE GUARD (engine bug found by the S1/S2 smokes 2026-09-02):
        # wave3d_attn.cudnn_fe keeps ONE growable per-device workspace. The
        # captured graphs of a K bake its ADDRESS; the first bigger K (K16 after
        # K8, K32 after K16) grows it -> the old buffer is freed (and cudaFree'd by
        # torch.cuda.graph's empty_cache at the next capture) -> replaying the
        # OLDER K (back to base K after a probe) is an illegal memory access.
        # Pre-growing it to the max any K/K64-val needs BEFORE the first capture
        # makes the address stable for the whole run. Belongs in wave3d_attn.py
        # (per-K workspace, or size it from the largest plan); trainer-side until
        # the engine owner lands that.
        from engines.wave3d.wave3d_attn import KERNELS
        dev = model.tok_emb.weight.device
        KERNELS["cudnn_fe"]._workspace(dev, int(cudnn_ws_gb * 2 ** 30))
    return eng


# ---- manual DDP: bucketed all-reduce over model.parameters() grads --------------
class GradAllReducer:
    """Flat-bucket all-reduce (AVG) of param.grad over the world, in a FIXED
    parameter order -> rank-order-independent -> rank-IDENTICAL grads -> identical
    gn -> identical skip/rewind decisions. Grads are fp32 (fp32 params); a param
    that received no grad this step is materialized as zeros so every rank
    issues the same collectives."""

    def __init__(self, params, world, bucket_mb=256):
        self.params = list(params)
        self.world = world
        self.bucket_elems = bucket_mb * (1 << 20) // 4
        self.buckets = []                    # list of lists of param indices
        cur, n = [], 0
        for i, p in enumerate(self.params):
            if n + p.numel() > self.bucket_elems and cur:
                self.buckets.append(cur)
                cur, n = [], 0
            cur.append(i)
            n += p.numel()
        if cur:
            self.buckets.append(cur)

    @torch.no_grad()
    def __call__(self, scale=1.0):
        if self.world <= 1:
            return
        for idxs in self.buckets:
            grads = []
            for i in idxs:
                p = self.params[i]
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                grads.append(p.grad)
            flat = torch._utils._flatten_dense_tensors(grads)
            dist.all_reduce(flat, op=dist.ReduceOp.AVG)
            if scale != 1.0:
                flat.mul_(scale)                 # outlier ranks zeroed -> average over kept ranks
            for g, s in zip(grads, torch._utils._unflatten_dense_tensors(flat, grads)):
                g.copy_(s)


def main():
    global PHASE_BOUNDS, LADDER_K, PROBE_MULT, PROBE_FRAC
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--seed", type=int, default=1337)
    # ---- recipe (GDN-2 match) ----
    ap.add_argument("--total_steps", type=int, default=200000)
    ap.add_argument("--global_batch", type=int, default=128)   # seqs (0.5M tok @ T4096)
    ap.add_argument("--accum", type=int, default=1,
                    help="grad-accumulation micro-steps per optim step. The engine "
                         "runs micro_bs = global_batch//world//accum seqs/micro; "
                         "grads accumulate across accum micros (loss scaled 1/accum) "
                         "so the optim step is IDENTICAL to a single (accum x "
                         "micro_bs) batch.")
    ap.add_argument("--T", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--min_lr", type=float, default=4e-5)
    ap.add_argument("--warmup_steps", type=int, default=2000)  # ~1B tok
    ap.add_argument("--lr_rewarm_steps", type=int, default=0,
                    help="SWEEP/RUNG-ENTRY: after a resume, scale the schedule LR by a linear ramp from "
                         "lr_rewarm_start to lr_rewarm_end over this many steps (0 = off, production default)")
    ap.add_argument("--lr_rewarm_start", type=float, default=0.25)
    ap.add_argument("--lr_rewarm_end", type=float, default=1.0)
    ap.add_argument("--lr_cap", type=float, default=0.0,
                    help="RUNG-ENTRY: cap the (ramped) schedule LR at this absolute value (0 = off, "
                         "production default); the cosine takes over once it falls below the cap")
    ap.add_argument("--probe_start", type=int, default=300,
                    help="first step at which 10%% probes run (FS absolute gate ~318 steps); decoupled from --warmup_steps, see sid_of")
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--gate_bias_init", type=float, default=0.0,
                    help="override cfg gate_bias_init (config default 4.0; hero "
                         "doctrine = 0.0 for the from-scratch A5 cold formation)")
    # ---- engine ----
    # PRODUCTION engine config (validated by validate_wave3d.py --engine x --graphs 1):
    # Wave3DX, all bodies torch.compile'd, CUDA-graph mode (defer_dw + cudnn_fe are
    # the engine defaults). 'vmap' + none + {} is the A/B fallback (~15% slower eager).
    ap.add_argument("--engine", default="x",
                    help="wave3d.make_engine selector: 'x' = wave3d_x.Wave3DX (PRODUCTION), "
                         "'vmap' = engines.wave3d.wave3d.Wave3D (A/B fallback)")
    ap.add_argument("--engine_kwargs", default='{"graphs": true}',
                    help="json dict passed STRAIGHT THROUGH to the engine constructor "
                         "(e.g. '{\"graphs\":true,\"attn_impl\":\"cudnn_fe\",\"seg_steps\":8}'); "
                         "C=64 is added unless given. graphs=true = CUDA-graph mode "
                         "(captured per K on first use, ~30-50 s one-time each)")
    ap.add_argument("--engine_compile", default="all",
                    help="none | refine | all -> eng.set_compile(mode) after construction")
    ap.add_argument("--cudnn_ws_gb", type=float, default=CUDNN_WS_GB_DEFAULT,
                    help="graph mode + cudnn_fe: pre-grow the cudnn_fe workspace to this "
                         "many GiB BEFORE the first capture (address must never change "
                         "once a K's graphs bake it; see build_engine). 0 = off")
    ap.add_argument("--phase_bounds", default=None,
                    help="SMOKE ONLY: comma fractions overriding PHASE_BOUNDS "
                         f"(default {PHASE_BOUNDS}); the production schedule is FROZEN")
    ap.add_argument("--ladder_k", default=None,
                    help="DIAGNOSTIC ONLY (cliff sweep 2026-09-03): comma ints overriding LADDER_K "
                         f"(default None keeps {LADDER_K}); must have len(phase_bounds)+1 rungs")
    # ---- data ----
    ap.add_argument("--data_dir", default=f"{R}/data/fwedu_100b")
    ap.add_argument("--train_split", default="fwedu_train")
    ap.add_argument("--val_split", default="fwedu_val")
    # ---- cadence ----
    ap.add_argument("--eval_interval", type=int, default=500)
    ap.add_argument("--tel_interval", type=int, default=1000)
    ap.add_argument("--log_interval", type=int, default=1)
    ap.add_argument("--ckpt_interval", type=int, default=1000)   # ckpt_latest (model+opt)
    ap.add_argument("--snap_interval", type=int, default=2000)   # persistent model-only
    ap.add_argument("--val_batches", type=int, default=8,
                    help="val batches per eval; each is world x micro_bs seqs (the "
                         "engine asserts B == micro_bs, so every rank runs micro_bs)")
    # ---- stability: outlier batch-skip ----
    ap.add_argument("--skip_factor", type=float, default=5.0,
                    help="skip if gn > skip_factor x that batch-type's accepted EMA")
    ap.add_argument("--skip_grace", type=int, default=100)
    ap.add_argument("--gn_hard", type=float, default=1e4,
                    help="absolute gn ceiling -> SKIP (overflow rail)")
    ap.add_argument("--loss_cap_mult", type=float, default=2.0,
                    help="(unused in pretrain: no loss-based skip) kept for CLI parity")
    # ---- stability: probe-aware clip ----
    ap.add_argument("--group_clip", choices=["fixed", "calibrated"],
                    default="calibrated",
                    help="calibrated: per-(base/probe) clip at clip_mult x that "
                         "type's gn EMA (floor grad_clip until grace); fixed: grad_clip")
    ap.add_argument("--clip_mult", type=float, default=2.0)
    # ---- ENHANCED storm/rewind governor ----
    ap.add_argument("--auto_rewind", default="true")
    ap.add_argument("--rewind_window", type=int, default=20)
    ap.add_argument("--gn_wedge", type=int, default=5,
                    help="rewind if >= this many gn-overflow SKIPS in the window")
    ap.add_argument("--loss_wedge", type=int, default=5,
                    help="rewind if >= this many high-loss SKIPS in the window")
    ap.add_argument("--loss_win", type=int, default=20)
    ap.add_argument("--creep_persist", type=int, default=10)
    ap.add_argument("--rewind_lookback", type=int, default=150,
                    help="rewind target must be >= this many steps before onset")
    ap.add_argument("--rewind_ring", type=int, default=8,
                    help="# of named healthy rewind snapshots to retain")
    ap.add_argument("--rw_snap_interval", type=int, default=500,
                    help="cadence (steps) of healthy rewind-ring snapshots (model+opt)")
    ap.add_argument("--rewind_floor", type=int, default=0,
                    help="NEVER rewind below this step (seed-escape at the floor)")
    ap.add_argument("--rewind_seed_stride", type=int, default=1000)
    ap.add_argument("--rewind_max", type=int, default=100,
                    help="safety cap on total interventions (reloads) — abort beyond")
    ap.add_argument("--rewind_recover", type=int, default=0,
                    help="applied steps clean past a rewind target before the run "
                         "is 'recovered' and a fresh (newer) target may be chosen "
                         "on the next storm (0 => rewind_window)")
    # ---- misc ----
    ap.add_argument("--wandb_log", default="true")
    ap.add_argument("--wandb_id", default=None)
    ap.add_argument("--wandb_name", default=None, help="wandb display name (default run_name); out dir stays run_name")
    ap.add_argument("--wandb_tags", default="",
                    help="comma list of wandb tags (e.g. 'smoke')")
    ap.add_argument("--resume", default="true")
    ap.add_argument("--probe_frac", default="0.1",
                    help="fraction of steps run as probes (0.1 = legacy step%%10==3; user 2026-09-06: 0.3)")
    ap.add_argument("--probe_mult", type=int, default=None,
                    help="probe K = probe_mult * base K (FS 2x-rung probes); default: next ladder rung")
    ap.add_argument("--local_window", type=int, default=0,
                    help="sliding-window attention, W tokens incl. self (GDN-2 SWA convention; 0 = full causal). "
                         "Sets cfg.local_window (saved in model_cfg) AND the engine's local_window kwarg.")
    ap.add_argument("--raw_diag", type=int, default=0,
                    help="1: engine attends each token's own RAW k/v (exact by nilpotency at K=C; the module's "
                         "sequential scan). 0: legacy wave3d refined-diagonal operator. Saved in ckpt['engine_kwargs'].")
    ap.add_argument("--mult_res_lambda", type=float, default=None,
                    help="override cfg.mult_res_lambda (DM residual carry; 1.0 = config default, 0.97 = contractive)")
    ap.add_argument("--resume_path", default=None,
                    help="explicit checkpoint/snapshot to resume from (default: <out_dir>/ckpt_latest.pt)")
    ap.add_argument("--max_extra_steps", type=int, default=20000,
                    help="safety cap on the skip-extension past total_steps")
    ap.add_argument("--seed_offset", type=int, default=None,
                    help="override the data-order offset (batch index = step + seed_offset); "
                         "normally restored from the ckpt; needed once for pre-fix ckpts after a rewind")
    ap.add_argument("--gn_debug_ranks", default="",
                    help="SMOKE: comma list of ranks that print their post-allreduce "
                         "grad norm every step (all-reduce correctness check)")
    ap.add_argument("--rank_outlier_factor", type=float, default=20.0,
                    help="rank-local gn > factor*median(ranks) => that rank's grads are zeroed before the all-reduce")
    ap.add_argument("--rank_outlier_max", type=int, default=4,
                    help="if more than this many ranks are outliers, skip the step (model-level instability)")
    ap.add_argument("--gn_rank_diag", default="true",
                    help="on every SKIPPED step, all-gather each rank's LOCAL (pre-all-reduce) grad "
                         "norm and print the median + top-3 ranks: attributes a gn spike to a rank. "
                         "The 2026-09-02 K8 gn storm (hero-fw-1p3b-w3dp-s1337 steps 5329-5355: gn 17 -> "
                         "2e5 on FROZEN weights, loss and K16 probes normal, cured by an in-process "
                         "reload) was NOT reproducible from the engine, the weights or the batches "
                         "(validate_grad_state diagnostics), so "
                         "the next occurrence must be attributed per rank (one bad rank averaged "
                         "into 32 looks exactly like this).")
    args = ap.parse_args()

    if args.ladder_k:
        LADDER_K = [int(k) for k in args.ladder_k.split(",") if k.strip()]
    if args.phase_bounds:
        PHASE_BOUNDS = [float(b) for b in args.phase_bounds.split(",") if b.strip()]
    if args.probe_mult:
        PROBE_MULT = int(args.probe_mult)
    _pf = [float(x) for x in str(args.probe_frac).split(",") if x.strip()]
    PROBE_FRAC = _pf[0] if len(_pf) == 1 else _pf          # scalar, or one fraction per phase
    assert len(PHASE_BOUNDS) == len(LADDER_K) - 1, (PHASE_BOUNDS, LADDER_K)
    engine_kwargs = json.loads(args.engine_kwargs)
    assert isinstance(engine_kwargs, dict), "--engine_kwargs must be a json object"
    if args.raw_diag:
        engine_kwargs["raw_diag"] = True
    if args.local_window:
        engine_kwargs["local_window"] = int(args.local_window)

    # ---- DDP ----
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    master = rank == 0
    T = args.T
    assert args.global_batch % world == 0, "global_batch must divide world"
    seqs_per_rank = args.global_batch // world           # per rank per OPTIM step
    accum = max(1, args.accum)
    assert seqs_per_rank % accum == 0, \
        "global_batch//world must divide accum (each micro-step = micro_bs seqs)"
    B_local = seqs_per_rank // accum                     # engine micro-batch (micro_bs)
    grad_scale = 1.0 / accum
    tok_per_step = args.global_batch * T                 # unchanged (accum is invisible)
    gn_debug = rank in {int(r) for r in args.gn_debug_ranks.split(",") if r.strip()}
    gn_rank_diag = args.gn_rank_diag == "true" and world > 1

    # ---- staircase ladder summary ----
    stage_desc = " | ".join(
        f"phase{i}(base K{LADDER_K[i]}," +
        (f"probe K{k_of(i, True)})" if (PROBE_MULT or i + 1 < len(LADDER_K)) else "NO probe)")
        for i in range(len(PHASE_BOUNDS) + 1))

    # ---- model (fp32 params for true fp32 Adam moments; bf16 autocast fwd) ----
    torch.manual_seed(args.seed)
    cfg = build_config(seq_len=T)
    cfg.gate_init_seed = args.seed
    if args.mult_res_lambda is not None:
        # 2026-09-03: 1.0 -> 0.97 from step 14000. At lam=1.0 the K16/K32/K64 backward has
        # sharp sequence-specific blow-ups in layer-0's DM value-ctx path (w_vctx; 62x at K64 on
        # snap_rw_0014000) that drove the run into a gradient cliff after the K16 switch;
        # 0.97 (contractive carry) removes them on the same weights/batches at <=5e-4 loss cost.
        cfg.mult_res_lambda = float(args.mult_res_lambda)
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"[{args.run_name}] mult_res_lambda={getattr(cfg, 'mult_res_lambda', 1.0)}  v_clamp_tau={getattr(cfg, 'v_clamp_tau', None)}", flush=True)
    if args.gate_bias_init is not None:
        cfg.gate_bias_init = args.gate_bias_init
    if args.local_window:
        cfg.local_window = int(args.local_window)       # module (exact scan / eval) + engine agree
    model = DualModLM(cfg).cuda().float()
    n_params = sum(p.numel() for p in model.parameters())
    if master:
        print(f"[{args.run_name}] model {n_params/1e9:.4f}B  d{cfg.d_model} "
              f"L{cfg.n_layers} h{cfg.n_heads} T{T} | global_batch "
              f"{args.global_batch} ({seqs_per_rank}/rank x {world} = accum {accum} "
              f"x micro_bs {B_local}) = {tok_per_step/1e6:.3f}M tok/step\n"
              f"    ladder LADDER_K {LADDER_K} bounds {PHASE_BOUNDS}: {stage_desc} "
              f"(10% probe step%10==3, post-warmup) exact-ref K{K_EVAL_EXACT}",
              flush=True)

    # ---- optimizer: fused AdamW, 2-group wd doctrine (2-D wd, 1-D none) ----
    decay = [p for p in model.parameters() if p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.wd},
         {"params": nodecay, "weight_decay": 0.0}],
        lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps, fused=True)

    out_dir = f"{R}/out/{args.run_name}"
    latest = f"{out_dir}/ckpt_latest.pt"
    resume_src = args.resume_path if args.resume_path else latest   # explicit snapshot wins

    # ---- resume ----
    start_step, applied = 0, 0
    gn_ema = {False: None, True: None}
    gn_cnt = {False: 0, True: 0}
    clip_ema = {False: None, True: None}
    tainted = set()
    resume_seed_offset = 0                     # restored from ckpt below if present
    resume_anchor = None                       # (step, path) of the resumed snapshot, if any
    _resume_ck_wb_tick = 0
    if args.resume == "true" and os.path.exists(resume_src):
        ck = torch.load(resume_src, map_location="cpu", weights_only=False)
        _resume_ck_wb_tick = int(ck.get("wb_tick", 0) or 0)
        model.load_state_dict(ck["model"])
        if "optimizer" in ck:
            opt.load_state_dict(ck["optimizer"])
        start_step = ck["step"] + 1
        applied = ck.get("applied", ck["step"])
        gn_ema = ck.get("gn_ema", gn_ema)
        gn_cnt = ck.get("gn_cnt", gn_cnt)
        clip_ema = ck.get("clip_ema", clip_ema)
        tainted = set(ck.get("tainted", []))
        resume_seed_offset = int(ck.get("seed_offset", 0))
        if "seed_offset" not in ck and master:
            print(f"[{args.run_name}] WARNING: ckpt has no seed_offset (pre-fix ckpt); "
                  f"pass --seed_offset explicitly if a rewind had rotated it, else batches REPLAY", flush=True)
        if master:
            print(f"[{args.run_name}] RESUME step->{start_step} applied->{applied}",
                  flush=True)
        # governor rewind target right after a resume. ckpt_latest.pt is OVERWRITTEN every ckpt_interval,
        # so pin the anchor to a STABLE file (hardlink; same filesystem) — 2026-09-06: an anchor bound to
        # ckpt_latest silently reloaded a later, pre-storm state (143000 labelled 141000) in a storm loop.
        _anchor_path = resume_src
        if os.path.basename(resume_src) == "ckpt_latest.pt":
            _anchor_path = f"{out_dir}/anchor_{ck['step']:07d}.pt"
            if master and not os.path.exists(_anchor_path):
                try:
                    os.link(resume_src, _anchor_path)
                except OSError:
                    shutil.copy(resume_src, _anchor_path)
            if world > 1:
                dist.barrier()
        resume_anchor = (ck["step"], _anchor_path)

    # ---- engine (wave3d layer-batched 3D wavefront; K per call; graph mode
    #      captures per K lazily on the first forward at that K) ----
    engine = build_engine(model, B_local, T, args.engine, engine_kwargs,
                          args.engine_compile, args.cudnn_ws_gb)
    reducer = GradAllReducer(model.parameters(), world)
    if master:
        print(f"[{args.run_name}] engine {type(engine).__name__} (--engine {args.engine}) "
              f"C={CHUNK_C} B{B_local} T{T} kwargs={engine_kwargs} "
              f"compile={args.engine_compile} | ddp buckets {len(reducer.buckets)}",
              flush=True)

    # ---- data ----
    loader = PackedLoader(args.data_dir, args.train_split, T, args.global_batch,
                          data_seed=args.seed, rank=rank, world_size=world)
    # the engine asserts idx.shape == (B_local, T) -> every rank gets B_local seqs
    val_loader = PackedLoader(args.data_dir, args.val_split, T, world * B_local,
                              data_seed=args.seed, rank=rank, world_size=world)
    # small fixed telemetry batch (eager parallel-sweep collect forward)
    tel_arr = np.memmap(f"{args.data_dir}/{args.val_split}.bin", dtype=np.uint16,
                        mode="r")
    _Ttel = min(512, T)
    x_tel = torch.from_numpy(
        np.asarray(tel_arr[:4 * _Ttel], dtype=np.int64).reshape(4, _Ttel)).cuda()

    # ---- logging ----
    run, metrics_f = None, None
    _wb_tick = [0]                         # monotonic wandb step (train/step is the x-axis metric)
    _resume_wb_tick = [int(_resume_ck_wb_tick)]
    if master:
        os.makedirs(out_dir, exist_ok=True)
        metrics_f = open(f"{out_dir}/metrics.jsonl", "a")
        if args.wandb_log == "true":
            import wandb
            tags = [t for t in args.wandb_tags.split(",") if t.strip()] or None
            run = wandb.init(project="dualmod", name=(args.wandb_name or args.run_name),
                             id=(args.wandb_id or args.run_name), resume="allow",
                             tags=tags,
                             config={**cfg.to_dict(), "data": "fineweb-edu-100b",
                                     "global_batch": args.global_batch,
                                     "accum": accum, "micro_bs": B_local,
                                     "total_steps": args.total_steps,
                                     "ladder_K": LADDER_K, "phase_bounds": PHASE_BOUNDS,
                                     "k_eval_exact": K_EVAL_EXACT,
                                     "lr": args.lr,
                                     "warmup_steps": args.warmup_steps,
                                     "wd": args.wd, "seed": args.seed,
                                     "engine": f"wave3d:{args.engine}",
                                     "engine_kwargs": engine_kwargs,
                                     "engine_compile": args.engine_compile,
                                     "chunk_C": CHUNK_C})
            wandb.define_metric("train/step")
            wandb.define_metric("*", step_metric="train/step")   # rewinds make `step` non-monotonic
            # resume: continue past the server high-water mark. run.step is 0 after resume="allow" in
            # current wandb SDKs (2026-09-09: every point after a resume was dropped as non-monotonic),
            # so take the max of the ckpt-saved tick and the API's lastHistoryStep.
            _hw = int(getattr(run, "step", 0) or 0)
            try:
                _hw = max(_hw, int(wandb.Api().run(f"{run.entity}/{run.project}/{run.id}").lastHistoryStep or 0) + 1)
            except Exception as _e:
                print(f"[wandb] lastHistoryStep lookup failed: {_e}", flush=True)
            _wb_tick[0] = max(_hw, _resume_wb_tick[0])
            print(f"[wandb] tick resumes at {_wb_tick[0]}", flush=True)

    def log(d, step):
        if master:
            metrics_f.write(json.dumps({"step": step, **d}) + "\n")
            metrics_f.flush()
            if run is not None:
                d["train/step"] = step; _wb_tick[0] += 1; run.log(d, step=_wb_tick[0])

    # ---- async checkpoint writer (bounded to one in-flight snap; from hero) ----
    _writer = [None]

    def _cpu_snapshot(x):
        if torch.is_tensor(x):
            return x.detach().to("cpu", copy=True)
        if isinstance(x, dict):
            return {k: _cpu_snapshot(v) for k, v in x.items()}
        if isinstance(x, (list, tuple, set)):
            return type(x)(_cpu_snapshot(v) for v in x)
        return x

    def _bg_write(state_cpu, path, link_latest):
        tmp = path + ".tmp"
        torch.save(state_cpu, tmp)
        os.replace(tmp, path)
        if link_latest:
            try:
                if os.path.islink(latest) or os.path.exists(latest):
                    os.remove(latest)
                os.symlink(os.path.basename(path), latest)
            except OSError:
                pass

    def _async_save(state, path, link_latest=False):
        os.makedirs(out_dir, exist_ok=True)
        if _writer[0] is not None:
            _writer[0].join()
        snap = _cpu_snapshot(state)
        t = threading.Thread(target=_bg_write, args=(snap, path, link_latest),
                             daemon=True)
        t.start()
        _writer[0] = t

    def _full_state(step):
        return {"model": model.state_dict(), "optimizer": opt.state_dict(),
                "step": step, "applied": applied, "model_cfg": cfg.to_dict(),
                "engine_kwargs": engine_kwargs, "wb_tick": _wb_tick[0],
                "gn_ema": gn_ema, "gn_cnt": gn_cnt, "clip_ema": clip_ema,
                "tainted": sorted(tainted),
                "seed_offset": seed_offset,   # batch index = step + seed_offset (rewind rotates it)
                "tokens": int((step + 1) * tok_per_step), "arch": "dualmod"}

    def save_latest(step):
        if master:                          # atomic tmp-rename write to ckpt_latest.pt
            _async_save(_full_state(step), latest, link_latest=False)

    # ---- rewind ring (healthy warm-Adam snaps; saved only when window clean) ---
    rewind_ring = []                       # list of (step, path), asc
    if resume_anchor is not None:
        rewind_ring.append(resume_anchor)  # the resumed snapshot IS a valid rewind target
        if master:
            print(f"[{args.run_name}] rewind ring seeded with resume anchor {resume_anchor[0]}", flush=True)
    recent_skips = collections.deque(maxlen=args.rewind_window)
    gn_skip_win = collections.deque(maxlen=args.rewind_window)
    loss_skip_win = collections.deque(maxlen=args.rewind_window)
    loss_win = collections.deque(maxlen=args.loss_win)
    creep_streak = 0
    intervention_count = 0                 # total rewinds (cap rewind_max -> abort)
    # PINNED-TARGET rewind policy (coordinator spec): on a storm, rewind to the
    # nearest good NON-TAINTED ckpt N and rotate the seed. If it re-storms, keep
    # rewinding to the SAME N (new seed each time) — never regress to N-1, N-2
    # (the r1 death spiral). Once the run has run clean for `recover` applied
    # steps past N, un-pin so the NEXT storm may choose a fresher target.
    pinned_target = None                   # (step, path) currently pinned, or None
    pin_floor = resume_anchor              # highest reload target ever chosen; never regress below it
    clean_since_reload = 10 ** 9           # applied steps since the last reload
    _resident = {"step": None, "state": None}   # in-memory copy of N (memory->GPU
    #   restore on repeated same-N reloads -> no 39GB disk read thrash, the r1 fix)
    recover_after = args.rewind_recover or args.rewind_window
    loss_ema = None

    def snap_rewind(step):
        """Named healthy warm-Adam snapshot for the rewind ring (rank 0 writes;
        every rank tracks the step list identically)."""
        path = f"{out_dir}/snap_rw_{step:07d}.pt"
        if master:
            _async_save(_full_state(step), path, link_latest=False)
        # a FRESH clean snapshot at this step comes from a NEW trajectory: lift any taint that an
        # earlier trajectory's storm put on this step number (2026-09-06: a clean 144500 sat unused
        # while the governor replayed 144000-144500 because 144500 had been tainted once).
        tainted.discard(step)
        if step not in [s for s, _ in rewind_ring]:
            rewind_ring.append((step, path))
            while len(rewind_ring) > args.rewind_ring:
                old_step, old_path = rewind_ring.pop(0)
                if master and os.path.exists(old_path) and \
                        old_step not in [s for s, _ in rewind_ring] and \
                        (resume_anchor is None or old_path != resume_anchor[1]) and \
                        (pin_floor is None or old_path != pin_floor[1]):
                    try:
                        os.remove(old_path)
                    except OSError:
                        pass

    def reload_from(chosen_step, chosen_path):
        """Rewind to ckpt N. If N is already the resident in-memory copy, restore
        memory->GPU (no disk touch — avoids the r1 rapid-reload rendezvous-heartbeat
        starvation). Otherwise rank0 (per node local_rank 0) stages NFS->/tmp once,
        all ranks load from /tmp (no 32-proc NFS thundering herd), then cache N
        resident for the seed-spin retries that pin to the SAME N."""
        nonlocal loss_ema
        if _resident["step"] == chosen_step and _resident["state"] is not None:
            ck = _resident["state"]
        else:
            if master and _writer[0] is not None:
                _writer[0].join()           # ensure the target is fully on disk
            if world > 1:
                dist.barrier()
            rwtmp = f"/tmp/ra/rewind_{args.run_name}.pt"
            if local_rank == 0:
                os.makedirs("/tmp/ra", exist_ok=True)
                shutil.copy(chosen_path, rwtmp)
            if world > 1:
                dist.barrier()
            ck = torch.load(rwtmp, map_location="cpu", weights_only=False)
            _resident["step"], _resident["state"] = chosen_step, ck   # cache resident
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        st = {"gn_ema": ck.get("gn_ema", gn_ema), "gn_cnt": ck.get("gn_cnt", gn_cnt),
              "clip_ema": ck.get("clip_ema", clip_ema),
              "step": ck["step"], "applied": ck.get("applied", ck["step"])}
        loss_ema = None
        return st

    rewind_on = args.auto_rewind == "true"
    t0 = time.time()
    capt_seen = set()                      # engine.mem capture keys already logged
    skipped_total = 0
    rank_excl_total = 0
    # batch index = step + seed_offset. Rewinds rotate seed_offset (+rewind_seed_stride) for a
    # fresh batch order; it MUST survive a resume or the resumed run REPLAYS batches already
    # consumed under the rotated offset (bug found 2026-09-02: resume after a rewind at 5000
    # re-fed batches 8001-9000 that steps 7001-8000 had already trained on).
    seed_offset = args.seed_offset if args.seed_offset is not None else resume_seed_offset
    if master:
        print(f"[{args.run_name}] seed_offset={seed_offset} -> first batch index {start_step + seed_offset}", flush=True)
    step = start_step
    model.train()
    opt.zero_grad(set_to_none=True)

    # FS semantics: run until `applied` reaches total_steps, i.e. extend past the nominal
    # end by the number of skipped steps (those batches are consumed as epoch-2 replay at
    # min_lr, top rung, no probes). Safety cap so a skip cascade cannot run forever.
    while applied < args.total_steps and step < args.total_steps + args.max_extra_steps:
        if step == args.total_steps and master:
            print(f"[{args.run_name}] EXTENSION: step {step} == total_steps, applied {applied}; "
                  f"continuing until applied reaches {args.total_steps} (~{args.total_steps-applied} steps)", flush=True)
        sid, is_probe = sid_of(step, args.total_steps, args.probe_start)   # sid == phase
        K = k_of(sid, is_probe)             # base rung, or its probe (2x base / next rung)
        phase = phase_of(step, args.total_steps)
        lr_now = lr_at(step, args.total_steps, args.warmup_steps, args.lr, args.min_lr)
        if args.lr_rewarm_steps > 0:
            _t = min(1.0, max(0.0, (step - start_step) / max(1, args.lr_rewarm_steps)))
            lr_now *= args.lr_rewarm_start + (args.lr_rewarm_end - args.lr_rewarm_start) * _t
        if args.lr_cap > 0:
            lr_now = min(lr_now, args.lr_cap)
        for g in opt.param_groups:
            g["lr"] = lr_now

        t_step = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()             # per-step (=> per-K) peak
        x, y = loader.train_batch(step + seed_offset)   # [accum*micro_bs, T] this rank
        # GRAD ACCUMULATION over `accum` micro-steps of micro_bs seqs each: plain
        # autograd, so grads ACCUMULATE in param.grad (loss scaled 1/accum) and the
        # optim step is IDENTICAL to one (accum*micro_bs) batch. ONE all-reduce /
        # clip / opt.step per OPTIM step. LR/step-count are on OPTIM steps.
        loss_acc = 0.0
        for micro in range(accum):
            xm = x[micro * B_local:(micro + 1) * B_local]
            ym = y[micro * B_local:(micro + 1) * B_local]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss_t = engine.forward(xm, ym, K)       # base/probe K, per call
            _ts = os.environ.get("RANK_EXCL_TEST_STEP")      # smoke-only: fake a single-rank spike
            _gs = grad_scale                                  # local: must NOT persist across steps
            if _ts is not None and step == int(_ts) and rank == int(os.environ.get("RANK_EXCL_TEST_RANK", "3")):
                _gs = grad_scale * 1e4
            (loss_t * _gs).backward()
            loss_acc += float(loss_t.detach())
        loss_v = loss_acc / accum
        # ---- RANK-OUTLIER REJECTION (added 2026-09-02 after the single-rank spike storms) ----
        # Every skipped step of this run showed ONE rank with a 1e2-1e6x grad norm while the
        # other 31 were normal (a different rank each time; loss normal). A plain AVG all-reduce
        # lets that rank dominate; the 5x-EMA skip then either rejects the whole step or, once
        # the EMA has inflated from tracking spikes, lets a corrupted step THROUGH (steps
        # 14470-14500 -> storm @14521). Fix: all-gather the rank-local grad norms (tiny), compute
        # the IDENTICAL outlier mask on every rank (gn > factor*median, or non-finite); outlier
        # ranks zero their grads before the bucket all-reduce and the AVG is rescaled to the
        # kept ranks. If more than rank_outlier_max ranks are outliers the model itself is
        # unstable -> the step is skipped (rank_storm). All decisions are rank-identical.
        gn_local = torch.nn.utils.get_total_norm(
            [p.grad for p in model.parameters() if p.grad is not None]).detach().float()
        gl = None; n_out = 0; rank_storm = False
        if world > 1:
            gl = torch.zeros(world, device="cuda")
            dist.all_gather_into_tensor(gl, gn_local.reshape(1))          # rank-identical
            fin = torch.isfinite(gl)
            med = gl[fin].median() if bool(fin.any()) else torch.zeros((), device="cuda")
            outlier = (~fin) | (gl > args.rank_outlier_factor * med.clamp_min(1e-12))
            n_out = int(outlier.sum().item())
            if 0 < n_out <= args.rank_outlier_max:
                if bool(outlier[rank]):
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad.zero_()
                reducer(scale=world / (world - n_out))
                rank_excl_total += n_out
                if master:
                    idx = torch.nonzero(outlier).flatten().tolist()
                    print(f"[{args.run_name}] RANK-EXCL step {step}: {n_out} rank(s) "
                          + " ".join(f"r{i}={float(gl[i]):.3g}" for i in idx)
                          + f" vs median {float(med):.3f} -> zeroed, AVG x{world/(world-n_out):.3f}", flush=True)
            else:
                rank_storm = n_out > args.rank_outlier_max         # too many bad ranks => skip step
                reducer()                                           # manual DDP (rank-identical grads)
        else:
            reducer()
        # rank-identical loss for the (rank-identical) skip/rewind decision
        if world > 1:
            lt = torch.tensor(loss_v if math.isfinite(loss_v) else 1e9, device="cuda")
            dist.all_reduce(lt, op=dist.ReduceOp.AVG)
            loss_v = float(lt)

        # DEBUG storm injector (off in prod): force a loss spike over a window on
        # the FIRST trajectory only (intervention_count==0) so the run RECOVERS
        # after the rewind — exercises storm->wedge->pinned-rewind end to end.
        _st_at = int(os.environ.get("HW_STORM_AT", "-1"))
        _st_n = int(os.environ.get("HW_STORM_N", "0"))
        if _st_at >= 0 and intervention_count == 0 and _st_at <= step < _st_at + _st_n:
            loss_v = 1e3

        # ---- probe-aware grad clip (calibrated per base/probe EMA, or fixed) ----
        if args.group_clip == "calibrated" and clip_ema[is_probe] is not None \
                and gn_cnt[is_probe] >= args.skip_grace:
            # hard 1.0 ceiling (== GDN-2 grad_clip); calibration may only TIGHTEN
            # below grad_clip, never loosen above it.
            clip_thr = min(args.grad_clip, args.clip_mult * clip_ema[is_probe])
        else:
            clip_thr = args.grad_clip
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip_thr))
        if gn_debug:
            print(f"[rank {rank}] step {step} K{K} probe={int(is_probe)} "
                  f"gn_post_allreduce {gn:.8f}", flush=True)

        # ---- outlier batch-SKIP (non-finite/gn_hard/rel-EMA; no loss term) ----
        ema = gn_ema[is_probe]
        gn_overflow = (not math.isfinite(gn) or not math.isfinite(loss_v)
                       or gn > args.gn_hard or rank_storm)
        # FS PRETRAIN skip: the ONLY soft guard is gn > skip_factor * gn_ema — no
        # grace, no warmup gate, no loss term (FS pretrain never looked at loss).
        # gn stabilizes <0.5 in pretrain, so a real spike (e.g. 100) is >> 5*ema
        # and skipped; deadlock-safe because gn_ema tracks through soft-skips below.
        gn_rel = (ema is not None and gn > args.skip_factor * ema)
        loss_out = False   # no loss-based skip in pretrain
        skip = gn_overflow or gn_rel
        # wedge windows count NON-probe skips only (probes exempt)
        if not is_probe:
            gn_skip_win.append(1 if (gn_overflow or gn_rel) else 0)   # finite spikes count too (2026-09-02)
            loss_skip_win.append(1 if loss_out else 0)

        if skip:
            opt.zero_grad(set_to_none=True)
            skipped_total += 1
            if gn_local is not None:
                # skip is a rank-identical decision (gn/loss are all-reduced) -> safe collective
                gl = torch.zeros(world, device="cuda")
                dist.all_gather_into_tensor(gl, gn_local.reshape(1))
                if master:
                    top = torch.topk(gl, k=min(3, world))
                    print(f"[{args.run_name}] SKIP step {step} rank-local pre-allreduce gn: "
                          f"median {gl.median().item():.3f} max/median "
                          f"{(gl.max() / gl.median().clamp_min(1e-12)).item():.1f}x  top "
                          + " ".join(f"r{int(i)}={float(v):.3f}" for v, i in zip(top.values, top.indices)),
                          flush=True)
        else:
            opt.step()
            opt.zero_grad(set_to_none=True)
            applied += 1
            clean_since_reload += 1
            if pinned_target is not None and clean_since_reload >= recover_after:
                pinned_target = None        # recovered -> next storm picks a fresh target
                _resident["step"], _resident["state"] = None, None
        # gn/clip EMA track the grad regime on ANY finite, non-hard-overflow step
        # (applied OR soft-skipped) so a persistent gn_rel skip can't freeze the
        # EMA and deadlock; only genuine blowups (gn_overflow) are excluded.
        if not gn_overflow and math.isfinite(gn):
            # BOUNDED tracking (2026-09-02): a soft-skipped spike moves the EMAs by at most
            # skip_factor*ema per step, so a burst of spikes cannot inflate the threshold enough
            # to let corrupted steps through (14470-14500 were applied at 3-25x normal gn).
            g_upd = min(gn, args.skip_factor * ema) if (skip and ema is not None) else gn
            gn_ema[is_probe] = gn if ema is None else 0.98 * ema + 0.02 * g_upd
            gn_cnt[is_probe] += 1
            ce = clip_ema[is_probe]
            clip_ema[is_probe] = gn if ce is None else 0.98 * ce + 0.02 * g_upd
        if not skip and not is_probe and math.isfinite(loss_v):
            loss_ema = loss_v if loss_ema is None else 0.98 * loss_ema + 0.02 * loss_v
            loss_win.append(loss_v)

        # ---- creep detector (APPLIED-loss sustained rise) ----
        creep_now = False
        if len(loss_win) >= args.loss_win:
            w = list(loss_win)
            tt = len(w) // 3
            m1 = statistics.median(w[:tt]); m2 = statistics.median(w[tt:2 * tt])
            m3 = statistics.median(w[2 * tt:])
            creep_now = (m3 > m2 > m1) and (m3 - m1) > 0.03
        creep_streak = creep_streak + 1 if creep_now else 0
        creep = (creep_streak >= args.creep_persist
                 and step >= args.warmup_steps + args.loss_win)

        gn_wedge = sum(gn_skip_win) >= args.gn_wedge
        loss_wedge = sum(loss_skip_win) >= args.loss_wedge
        wedge = gn_wedge or loss_wedge or creep

        # ---- logging ----
        torch.cuda.synchronize()
        step_ms = round((time.perf_counter() - t_step) * 1000)
        step_peak_gb = round(torch.cuda.max_memory_allocated() / 1e9, 1)
        # graph replays bypass the allocator -> max_allocated UNDER-reports; the
        # honest footprint is memory_reserved (K8 ~75 / K16 ~92 / K32 ~131 GB)
        reserved_gb = round(torch.cuda.memory_reserved() / 1e9, 1)
        # CUDA-graph capture events (per K, first use; also at every probe rung and
        # after a resume): the engine records capture_K<K>_s -> log ONCE each.
        eng_mem = getattr(engine, "mem", None) or {}
        new_caps = {k: v for k, v in eng_mem.items()
                    if k.startswith("capture_K") and k.endswith("_s") and k not in capt_seen}
        for k, v in new_caps.items():
            capt_seen.add(k)
            kk = k[len("capture_"):-len("_s")]
            if master:
                print(f"[{args.run_name}] GRAPH CAPTURE {kk} at step {step}: {v:.1f}s "
                      f"(capture peak {eng_mem.get(f'{kk}_capture_peak_GB', 0):.1f}GB, "
                      f"reserved after {eng_mem.get(f'capture_{kk}_reserved_GB', 0):.1f}GB)",
                      flush=True)
        if master and (step % args.log_interval == 0):
            d = {"train/loss": loss_v, "train/lr": lr_now, "train/grad_norm": gn,
                 "train/sid": sid, "train/K": K, "train/phase": phase,
                 "train/is_probe": int(is_probe),
                 "train/clip_event": int(gn > clip_thr), "train/clip_thr": clip_thr,
                 "train/skipped": int(skip), "train/skipped_total": skipped_total, "train/rank_excl": n_out, "train/rank_excl_total": rank_excl_total,
                 "train/applied": applied, "train/tokens": int((step + 1) * tok_per_step),
                 "train/step_ms": step_ms, "train/mem_mode": 3,   # 3 = wave3d (segment ckpt)
                 "train/step_peak_gb": step_peak_gb,              # this step's (this K's) peak
                 "train/gpu_gb": round(torch.cuda.max_memory_reserved() / 1e9, 1),
                 "train/gpu_reserved_gb": reserved_gb}
            for k, v in new_caps.items():
                kk = k[len("capture_"):-len("_s")]
                d[f"engine/capture_{kk}_s"] = v
                d[f"engine/capture_{kk}_reserved_gb"] = eng_mem.get(f"capture_{kk}_reserved_GB", 0)
            if is_probe:
                d["probe/loss"] = loss_v
                d["probe/grad_norm"] = gn
                if gn_ema[False]:
                    d["probe/gn_ratio_vs_base"] = gn / max(gn_ema[False], 1e-9)
            else:
                d["base/grad_norm"] = gn
            log(d, step)
            if step % 50 == 0 or args.total_steps <= 200:
                print(f"[{args.run_name}] step {step}/{args.total_steps} ph{phase} K{K}"
                      f"{'(probe)' if is_probe else ''} loss {loss_v:.4f} gn {gn:.3f} "
                      f"clip {clip_thr:.2f} appl {applied} skip {skipped_total} "
                      f"{step_ms}ms peak {step_peak_gb}GB resv {reserved_gb}GB", flush=True)
            if skip:
                print(f"[{args.run_name}] SKIP step {step}: gn {gn:.2f} loss "
                      f"{loss_v:.4f} (probe={is_probe})", flush=True)

        # ---- eval (held-out CE): val/loss at the CURRENT-PHASE base K (train==
        #      eval), plus val/loss_exact at K=64 (exact operator reference; the
        #      base->K64 gap is the entrenchment guard the probes keep small) ----
        # GRAPH-MODE INVARIANT: val is an EAGER no_grad engine forward. It MUST run
        # here, AFTER backward + opt.step, never between a graphed forward and its
        # backward (an eager forward there corrupts the shared caches the bwd
        # graphs read). Same for gate_stats_layerwise (eager model forward).
        if step % args.eval_interval == 0 or step == args.total_steps - 1:
            base_K = LADDER_K[phase]                              # current phase base
            val_base = _val_ce(engine, model, val_loader, args.val_batches, world, base_K)
            val_k64 = _val_ce(engine, model, val_loader, args.val_batches, world, K_EVAL_EXACT)
            vd = {"val/loss": val_base, "val/loss_exact": val_k64,
                  "val/base_K": base_K, "val/base_minus_k64": val_base - val_k64}
            if step % args.tel_interval == 0:
                try:
                    vd.update(gate_stats_layerwise(model, x_tel, base_K))
                    model.train()
                except Exception as e:
                    if master:
                        print(f"[{args.run_name}] telemetry skipped: {e}", flush=True)
            log(vd, step)
            if master:
                print(f"[{args.run_name}] EVAL step {step}: val/loss(K{base_K}) "
                      f"{val_base:.4f}  val/loss_exact(K{K_EVAL_EXACT}) {val_k64:.4f}  "
                      f"gap {val_base-val_k64:+.4f}", flush=True)

        # ---- checkpoints ----
        if step % args.ckpt_interval == 0 and step > start_step:
            save_latest(step)
        if master and step % args.snap_interval == 0 and step > 0:
            _async_save({"model": model.state_dict(), "model_cfg": cfg.to_dict(),
                         "engine_kwargs": engine_kwargs,
                         "step": step, "applied": applied, "seed_offset": seed_offset,
                         "tokens": int((step + 1) * tok_per_step), "arch": "dualmod"},
                        f"{out_dir}/persist_{step:07d}.pt")

        # ==================== ENHANCED STORM / REWIND GOVERNOR ====================
        if rewind_on:
            recent_skips.append(1 if skip else 0)
            # healthy ring snap ONLY when the recent window is skip-free
            if step % args.rw_snap_interval == 0 and sum(recent_skips) == 0 \
                    and step >= args.rewind_floor:
                snap_rewind(step)

            if wedge:
                intervention_count += 1
                detected_step = step
                reason = ("gn-wedge" if gn_wedge else
                          "loss-wedge" if loss_wedge else "loss-creep")
                if intervention_count > args.rewind_max:
                    if master:
                        print(f"[{args.run_name}] REWIND ABORT: "
                              f"{intervention_count} interventions > "
                              f"{args.rewind_max} — persistent wedge", flush=True)
                    break

                # PINNED-TARGET policy: reuse the currently pinned target N if it is
                # still valid; else choose the NEAREST good NON-TAINTED snap >=
                # lookback and >= floor. On re-storm we do NOT regress to N-1 —
                # we keep reloading the SAME N and rotate the seed (r1 fix).
                if pinned_target is not None and pinned_target[0] not in tainted \
                        and pinned_target[0] >= args.rewind_floor:
                    chosen_step, chosen_path = pinned_target
                else:
                    # NEVER REGRESS (user 2026-09-06: "never let it go below the last good ckpt and keep
                    # looping seeds until you pass through"): the highest reload target ever chosen
                    # (pin_floor) is always eligible (lookback-exempt), never tainted, and nothing
                    # below it is considered. The pin moves FORWARD only when a newer snap qualifies.
                    cands = [(s, p) for (s, p) in rewind_ring
                             if (s <= step - args.rewind_lookback
                                 or (resume_anchor is not None and s == resume_anchor[0])
                                 or (pin_floor is not None and s == pin_floor[0]))
                             and s >= args.rewind_floor and s not in tainted
                             and (pin_floor is None or s >= pin_floor[0])]
                    if not cands and pin_floor is not None:
                        cands = [pin_floor]
                    # taint ring snaps on the abandoned (diverging) trajectory —
                    # everything AFTER the target we are about to pick, up to the
                    # storm (quarantine: these led into the storm, never reuse).
                    base = cands[-1][0] if cands else args.rewind_floor
                    for s, _ in rewind_ring:
                        if base < s <= step and not (pin_floor is not None and s == pin_floor[0]):
                            tainted.add(s)
                    pinned_target = cands[-1] if cands else None
                    chosen_step = pinned_target[0] if pinned_target else None
                    chosen_path = pinned_target[1] if pinned_target else None

                if pinned_target is not None:
                    if pin_floor is None or pinned_target[0] >= pin_floor[0]:
                        pin_floor = pinned_target          # highest target ever chosen = floor
                    # rotate the data offset ONLY when we actually reload: the model then
                    # re-walks the steps on different batches. Rotating on a "ride" (no
                    # reload, 2026-09-03: storms #2-#8 fired every ~5 steps) just SKIPS a
                    # stride of never-seen data per storm (9000 batches in 30 steps).
                    seed_offset += args.rewind_seed_stride
                    st = reload_from(chosen_step, chosen_path)
                    gn_ema.update(st["gn_ema"]); gn_cnt.update(st["gn_cnt"])
                    clip_ema.update(st["clip_ema"])
                    applied = st["applied"]
                    step = st["step"] + 1
                    clean_since_reload = 0
                    detail = f"reload@{chosen_step} (seed-rotate; pinned)"
                else:
                    # no valid target yet (early run / all tainted / below floor):
                    # ride it — rotate seed and retry the same step, no reload.
                    detail = "no valid target -> ride (seed-rotate only)"

                recent_skips.clear(); gn_skip_win.clear(); loss_skip_win.clear()
                loss_win.clear(); creep_streak = 0
                if master:
                    print(f"[{args.run_name}] STORM #{intervention_count} detected@{detected_step} "
                          f"({reason}) -> {detail}, continue@{step} seed_off={seed_offset} "
                          f"tainted={len(tainted)}", flush=True)
                    if run is not None:
                        _wb_tick[0] += 1
                        run.log({"train/rewind_count": intervention_count,
                                 "train/rewind_event": 1, "train/step": step,
                                 "train/rewind_target": chosen_step or -1}, step=_wb_tick[0])
                    with open(f"{out_dir}/rewind_events.jsonl", "a") as rf:
                        rf.write(json.dumps({
                            "intervention": intervention_count, "step": step,
                            "reason": reason, "target": chosen_step,
                            "seed_offset": seed_offset,
                            "tainted": sorted(tainted)}) + "\n")
                continue          # retry AT the (reloaded or held) step
        # ================== /ENHANCED STORM / REWIND GOVERNOR ====================
        step += 1

    if master:
        save_latest(args.total_steps - 1)
        if _writer[0] is not None:
            _writer[0].join()
        print(f"[{args.run_name}] DONE: {step} steps, skipped {skipped_total}, "
              f"interventions {intervention_count}, {time.time()-t0:.0f}s", flush=True)
        if metrics_f is not None:
            metrics_f.close()
        if run is not None:
            run.finish()
    if world > 1:
        dist.destroy_process_group()


@torch.no_grad()
def _val_ce(engine, model, val_loader, n_batches, world, K_eval):
    """Held-out CE via the SAME wave3d engine (eng.forward under no_grad, bf16
    autocast) at K_eval. Used for val/loss (current-phase base K, the train
    operator) and val/loss_exact (K=64, exact by nilpotency at C=64). Every rank
    feeds exactly micro_bs seqs per batch (the engine asserts B == micro_bs)."""
    was_train = model.training
    model.eval()
    tot, n = 0.0, 0
    for i in range(n_batches):
        x, y = val_loader.val_batch(i)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            l = engine.forward(x, y, K_eval)
        tot += float(l); n += 1
    if was_train:
        model.train()
    # The eager no_grad forward memoizes its attention outputs in engine._stash
    # (stash_attn; useless without a recompute) and leaves them ALIVE until the
    # next capture/eager forward: 177 GB after a B4/T4096 K64 val (measured).
    # Release them now; the graph replays use their own captured buffers.
    if hasattr(engine, "_stash"):
        engine._stash = {}
    v = tot / max(n, 1)
    if world > 1:
        vt = torch.tensor(v, device="cuda")
        dist.all_reduce(vt, op=dist.ReduceOp.AVG)
        v = float(vt)
    return v


if __name__ == "__main__":
    main()
