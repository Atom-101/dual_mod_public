"""Flagship DM run: SlimPajama 15B @ T=2048 (plans/flagship_run.md), 2-node
16xH100, global batch 768 seqs (1.573M tok/step, ~9537 steps).

Deviations from the plan, per launch decision 2026-07-08:
  - global batch 768 (48/GPU x 16) instead of fla's 256 — same LR/opt recipe.
  - no SlimPajama 2B smoke (P-1 skipped; ClimbMix schedule law assumed).
  - probe steps: 10% of steps (plan said 5%), at 2x current phase ratio,
    starting after LR warmup. sid is a pure function of step -> every rank
    picks the same schedule (no DDP stalls).

Schedule ladder (c=64 throughout, sids pre-captured at init):
  sid 0: K8   (ratio 0.125)  P1  0 -> 7.5% tokens
  sid 1: K16  (ratio 0.25)   P2  7.5% -> 70%
  sid 2: K32  (ratio 0.5)    P3  70% -> 100%
  sid 3: K64  (ratio 1.0)    probe target in P3; also the exact-eval sid

Telemetry: per-layer gate stats (g_k/g_v mean, register_frac, frac<0.3),
global grad-norm every step split probe-vs-base, clip events (clip stays at
1.0 — it was on for ClimbMix), outlier batch-skip on running (EMA) grad-norm
per batch type, per-layer grad norms at eval steps.

Checkpoints: model snaps every --snap_interval; full resume state (model+
opt+EMAs+step) every --ckpt_interval to ckpt_latest.pt (atomic, previous kept
as ckpt_prev.pt). --resume auto continues from ckpt_latest.pt; the data
stream is a pure function of step so it fast-forwards exactly.

Launch (per node; node_rank 0 gets wandb/master):
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  torchrun --nnodes=2 --nproc_per_node=8 --node_rank=$NR \
      --rdzv_backend=c10d --rdzv_endpoint=$MASTER:29500 \
      lm/train_340m/train_flagship.py --run_name PDN-DM-d960-mrk-lora8-sync-s42 --seed 42
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.dualmod.config import DualModConfig
from models.dualmod.model import DualModLM

from engines.wavescan.engine.graphs import WaveScanEngine
from engines.wavescan.engine.schedule import uniform_chunks

# Two schedule ladders (sid 0/1/2 = phase base, sid 3 = deepest probe):
#   k-ladder (FS1): c=64 fixed, K 8/16/32/64 — ratio grows via K (backward
#     tower grows with K; K64 probes caused the FS1 inf/NaN events)
#   c-ladder:       K=8 fixed, c 64/32/16/8 — same ratios 0.125/.25/.5/1.0
#     at ≤1/4 the refine FLOPs with the tower pinned at depth 8 (certified
#     free by the FS1 (c,K) grid: ratio 0.25 ≤3e-4 all run, 0.5 exact)
LADDERS = {"k": [(64, 8), (64, 16), (64, 32), (64, 64)],
           # c=8 is below the fused kernels' tl.dot M>=16 floor -> the
           # ratio-1.0 rung is c16/K16 (tower depth 16, proven-safe zone)
           "c": [(64, 8), (32, 8), (16, 8), (16, 16)]}
PHASE_BOUNDS = [0.075, 0.70]        # token-fraction entry of P2, P3
                                     # (mutable via --phase_bounds)
EXACT_SID = 3                        # ratio 1.0 -> exact by nilpotency


def phase_of(step, total):
    frac = step / total
    if frac < PHASE_BOUNDS[0]:
        return 0
    if frac < PHASE_BOUNDS[1]:
        return 1
    return 2


SCHEDULE_MODE = ["ladder"]          # "ladder" | "flat8_probes"


def sid_of(step, total, warmup_steps):
    """Schedule id for this step. Pure function of step -> rank-consistent."""
    if SCHEDULE_MODE[0] == "exact16":
        # c16/K16 exact (ratio 1.0 by nilpotency) from step 0: no ladder,
        # no ratio transitions (every wedge to date sat on one), no probes
        # needed (nothing deeper exists). Warmup happens AT the final
        # operator.
        return 0, False
    if SCHEDULE_MODE[0] == "flat8_probes":
        # contraction test: base 100% K8 (ratio 0.125); probes 5% K32 +
        # 5% K64 keep the deep-chain gradient path open (entrenchment law)
        if step >= warmup_steps and step % 20 == 3:
            return 2, True
        if step >= warmup_steps and step % 20 == 13:
            return 3, True
        return 0, False
    ph = phase_of(step, total)
    if step >= warmup_steps and step % 10 == 3:      # 10% probe steps
        return min(ph + 1, 3), True
    return ph, False


def lr_at(step, total, warmup_steps, lr=3e-4, min_lr=3e-5):
    if step < warmup_steps:
        return lr * (step + 1) / warmup_steps
    prog = (step - warmup_steps) / max(1, total - warmup_steps)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * prog))


def memmap_stream(path, seq_len, batch_size, rank, world, start_step=0,
                  seed=42, dtype=np.uint16):
    """Deterministic packed-chunk stream; per-epoch perm seeded seed+epoch so
    any step's batch is reproducible from (seed, step) — exact resume."""
    arr = np.fromfile(path, dtype=dtype)
    blk = seq_len + 1
    n_chunks = arr.shape[0] // blk
    steps_per_epoch = n_chunks // (batch_size * world)
    step = start_step
    while True:
        epoch = step // steps_per_epoch
        g = torch.Generator().manual_seed(seed + epoch)
        perm = torch.randperm(n_chunks, generator=g)
        s_ = step % steps_per_epoch
        while s_ < steps_per_epoch:
            idx = perm[(s_ * world + rank) * batch_size:
                       (s_ * world + rank + 1) * batch_size]
            rows = np.stack([arr[c * blk:(c + 1) * blk] for c in idx.tolist()])
            yield torch.from_numpy(rows.astype(np.int64))
            step += 1
            s_ += 1


@torch.no_grad()
def engine_val(engine, val):
    """Exact val via the ratio-1.0 schedule (nilpotent-exact): seconds, not
    the ~10min an eager scan costs at d1024/L24/T2048. Val rows are
    re-chunked to the engine's fixed B (the FS1 val tensor was 48-wide by
    coupling to B_local=48; PDN runs B_local=32). Cross-checked sparsely by
    eager_val (audits: engine-vs-eager agrees <1e-4)."""
    rows = val.reshape(-1, val.shape[-1])
    if rows.shape[-1] > engine.T + 1:      # short-T runs: val is chunked at 2048;
        rows = rows[:, :engine.T + 1]      # re-slice to the engine's T (+1 target)
    B = engine.B
    ls = []
    for i in range(rows.shape[0] // B):
        blk = rows[i * B:(i + 1) * B].cuda()
        l = engine.forward(blk[:, :-1].contiguous(), EXACT_SID,
                           targets=blk[:, 1:].contiguous())
        ls.append(l.item())
    return float(np.mean(ls))


@torch.no_grad()
def eager_val(model, val, n_batches=1, batch_rows=8, rows=16):
    """Sparse belt-and-suspenders cross-check of engine_val. The eager scan
    is latency-bound (~5-10 min per [*,2048] batch at L24/d1024) — keep this
    to ONE batch, few rows, wide interval."""
    model.eval()
    ls = []
    for i in range(n_batches):
        blk = val[i, :rows].cuda()
        x, y = blk[:, :-1], blk[:, 1:]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(x)
        tot, cnt = 0.0, 0
        for r0 in range(0, x.shape[0], batch_rows):
            yc = y[r0:r0 + batch_rows]
            tot += F.cross_entropy(
                logits[r0:r0 + batch_rows].float().flatten(0, 1),
                yc.reshape(-1), reduction="sum").item()
            cnt += yc.numel()
        ls.append(tot / cnt)
    return float(np.mean(ls))


def ctx_top_eig(model, xb, yb, params, n_iter=5):
    """Direction-free top eigenvalue of the loss Hessian restricted to the ctx
    ops (`params` = w_kctx/w_vctx), via HVP power iteration on the PLAIN model
    forward (create_graph double-backward) — the exact R4 instrument, run live.
    Independent of the engine. Returns the scalar eig (>0 up = sharpening).
    Caller runs this master-only at a coarse cadence and times it; on any
    failure it returns nan (best-effort telemetry, never fatal to training)."""
    was_train = model.training
    model.train()
    saved = [(p, p.requires_grad) for p in model.parameters()]
    try:
        for p, _ in saved:
            p.requires_grad_(False)
        for p in params:
            p.requires_grad_(True)
        vs = [torch.randn_like(p) for p in params]
        nrm = torch.sqrt(sum((v * v).sum() for v in vs))
        vs = [v / (nrm + 1e-12) for v in vs]
        eig = float("nan")
        for _ in range(n_iter):
            _, loss = model(xb, yb)
            grads = torch.autograd.grad(loss.float(), params, create_graph=True)
            gv = sum((g * v).sum() for g, v in zip(grads, vs))
            Hv = torch.autograd.grad(gv, params)
            eig = float(sum((h * v).sum() for h, v in zip(Hv, vs)))
            nrm = torch.sqrt(sum((h * h).sum() for h in Hv))
            vs = [(h / (nrm + 1e-12)).detach() for h in Hv]
        return eig
    finally:
        for p, r in saved:
            p.requires_grad_(r)
        if not was_train:
            model.eval()


@torch.no_grad()
def gate_stats_layerwise(model, x):
    """Per-layer gate telemetry (restored from the TinyStories runs, plus the
    distribution stats flagship_run.md §4 asks for since regfrac saturates)."""
    model.eval()
    collect = [dict() for _ in model.blocks]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        model(x, collect=collect)
    out = {}
    # --- drift arm 1: v'' tail telemetry (clamp tau source) ---------------
    # per-(position, head) L2 norm of the written value v''; the storm's
    # forward symptom and the quantity the clamp v'' <- v''*min(1, tau/||v''||)
    # would govern. We log the healthy p99.9 (tau candidate) + p999-of-p999
    # across layers and the global max. Uses the vpp already collected in
    # refine_kv (detached), so no extra forward and no kernel touch.
    hd = model.blocks[0].attn.head_dim
    vn_p999, vn_max = [], []
    for li, c in enumerate(collect):
        if "vpp" not in c:
            break
        v = torch.cat(c["vpp"], dim=1)                      # [B, T, d_kv]
        vh = v.reshape(v.shape[0], v.shape[1], -1, hd)      # [B,T,Hkv,hd]
        rn = vh.norm(dim=-1).flatten()                      # per (b,t,head)
        p999 = torch.quantile(rn, 0.999).item()
        vn_p999.append(p999)
        vn_max.append(rn.max().item())
        out[f"vtail/p999_L{li:02d}"] = p999
    if vn_p999:
        out["vtail/p999_all_layers"] = float(np.mean(vn_p999))
        out["vtail/p999_max_layer"] = float(np.max(vn_p999))
        out["vtail/max_all_layers"] = float(np.max(vn_max))
    # --- drift arm 1b: ctx-read attention-logit tails (QK-norm decision) ---
    al_p999, al_max = [], []
    for li, c in enumerate(collect):
        if "attn_logit_max" not in c:
            break
        s = torch.cat(c["attn_logit_max"])                  # per (pos,head)
        al_p999.append(torch.quantile(s, 0.999).item())
        al_max.append(s.max().item())
        out[f"attnlogit/p999_L{li:02d}"] = al_p999[-1]
    if al_p999:
        out["attnlogit/p999_all_layers"] = float(np.mean(al_p999))
        out["attnlogit/p999_max_layer"] = float(np.max(al_p999))
        out["attnlogit/max_all_layers"] = float(np.max(al_max))
    # --- K-norm gauge: k_gain (gamma) is the single scalar where read-
    # sharpening pressure concentrates. Monotone climb => the disease routing
    # around the governor (a finding). Logged per-layer mean + global max. ---
    kg = [blk.attn.k_gain for blk in model.blocks
          if hasattr(blk.attn, "k_gain")]
    if kg:
        allg = torch.cat([g.detach().float().flatten() for g in kg])
        out["kgain/mean_all_layers"] = float(allg.mean())
        out["kgain/max_all_layers"] = float(allg.max())
        # spread: does learned gamma rediscover the survivor's 7x head
        # specialization ([0.46, 3.13] on DM-d832)? std/min track it live.
        out["kgain/min_all_layers"] = float(allg.min())
        out["kgain/std_all_layers"] = float(allg.std())
        out["kgain/spread_all_layers"] = float(allg.max() / (allg.min() + 1e-6))
        for li, g in enumerate(kg):
            out[f"kgain/mean_L{li:02d}"] = float(g.detach().float().mean())
    for tag in ("k", "v"):
        gm, rf, lo = [], [], []
        ratios, effs = [], []
        for li, c in enumerate(collect):
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
        out[f"telemetry/g_{tag}_mean_all_layers"] = float(np.mean(gm))
        if ratios:
            out[f"telemetry/ratio_{tag}_mean"] = float(np.mean(ratios))
            out[f"telemetry/eff_ctx_share_{tag}"] = float(np.mean(effs))
        out[f"telemetry/register_frac_{tag}_all_layers"] = float(np.mean(rf))
        out[f"telemetry/fraclt03_{tag}_all_layers"] = float(np.mean(lo))
    return out


def layer_grad_norms(model, engine):
    out = {}
    for li, blk in enumerate(model.blocks):
        s = 0.0
        for p in blk.parameters():
            g = engine.grads.buf.get(id(p))
            if g is not None:
                s += g.float().pow(2).sum().item()
        out[f"gradnorm/L{li:02d}"] = math.sqrt(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--total_steps", type=int, default=9537)   # 15B tok @ 768x2048
    ap.add_argument("--global_batch", type=int, default=768)
    ap.add_argument("--warmup_tokens", type=float, default=0.5e9)
    ap.add_argument("--data_bin", default="data/slimpj_train.bin", help="packed uint16 token stream; stage to local disk (e.g. /tmp) for throughput")
    ap.add_argument("--eval_interval", type=int, default=250)
    ap.add_argument("--eager_val_interval", type=int, default=4000)
    ap.add_argument("--snap_interval", type=int, default=1000)
    # --- SAM (arm F) + curvature alarm (spectrum_optims Tier-0 greenlight) ----
    # All default-off => byte-identical to the committed path when unset. SAM
    # touches NO kernel: it replays engine.train_step twice and perturbs p.data
    # in-place on the ctx ops (w_kctx/w_vctx = the R4-measured set); the alarm
    # uses the separate plain model(x,y) double-backward (never the engine).
    ap.add_argument("--sam", action="store_true",
                    help="Sharpness-Aware Minimization on ctx ops only "
                         "(perturb w_kctx/w_vctx by rho*g/||g||, step on the "
                         "perturbed-point grad). ~2x train_step cost per SAM "
                         "step; probe batches exempt. NO kernel change.")
    ap.add_argument("--sam_rho", type=float, default=0.05,
                    help="SAM ascent radius (||perturbation||_2 over ctx ops)")
    ap.add_argument("--sam_every", type=int, default=1,
                    help="apply SAM every k non-probe steps (cost knob; "
                         "off-steps take a normal single step)")
    ap.add_argument("--curv_alarm_interval", type=int, default=0,
                    help="direction-free ctx-param top-Hessian-eig (power "
                         "iteration on plain model forward) every N steps, "
                         "master-only, timed. 0=off. The R4 dial as a live "
                         "leading indicator (threshold ~2x the survivor band).")
    ap.add_argument("--curv_alarm_iters", type=int, default=5)
    ap.add_argument("--curv_alarm_rows", type=int, default=4)
    ap.add_argument("--ckpt_interval", type=int, default=250)
    ap.add_argument("--skip_factor", type=float, default=5.0)
    ap.add_argument("--skip_grace_steps", type=int, default=100)
    ap.add_argument("--skip_free_sched", action="store_true",
                    help="index LR + c/K ladder by APPLIED updates (not raw"
                         " iters) and run until total_steps updates land, so"
                         " skipped batches don't waste LR-decay budget or"
                         " under-train. Off = current step-indexed behavior.")
    ap.add_argument("--wandb_log", default="false")
    ap.add_argument("--graphs", action="store_true",
                    help="capture CUDA graphs (B300-style); default off on H100")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--force_sid", type=int, default=None,
                    help="clamp every step (incl. probes) to this ladder rung"
                         " — storm-zone (c,K) diagnostics only")
    ap.add_argument("--wd_doctrine", choices=["pdn", "fs1"], default="pdn",
                    help="pdn = wd on all params (titan builder); fs1 = no wd"
                         " on 1-D params (gate biases + norms). fs1 under"
                         " recipe pdn transplants Adam state on resume")
    ap.add_argument("--ctx_mod", default="linear",
                    help="ctx-path nonlinearity for CONTINUATION runs"
                         " (fusion|mult_res). Forces non-fused compiled"
                         " cells; resume becomes strict=False with"
                         " optimizer-state transplant; branch params get a"
                         " 10x-wd hold-down group")
    ap.add_argument("--kmod_vraw_heads", type=int, default=0)
    ap.add_argument("--kmod_mode", choices=["raw", "gated_norm"],
                    default="raw")
    ap.add_argument("--v_norm", choices=["none", "rms"], default="none")
    ap.add_argument("--v_clamp_tau", type=float, default=0.0,
                    help="v'' entry-norm clamp radius (storm governor; 0=off). "
                         "Calibrate from a healthy run's vtail/p999.")
    ap.add_argument("--mult_res_lambda", type=float, default=1.0,
                    help="mult_res identity-carry leak (structural fix; "
                         "1.0=off, 0.97=arm). Only affects ctx_mod=mult_res.")
    ap.add_argument("--k_ctx_norm", choices=["none", "rms"], default="none",
                    help="K-norm: per-head RMS+gain on k_ctx (read governor).")
    ap.add_argument("--k_gain_init", type=float, default=1.75,
                    help="K-norm gain init (survivors' k_ctx RMS ~1.75).")
    ap.add_argument("--v_ctx_norm", choices=["none", "rms"], default="none")
    ap.add_argument("--gate_rank", type=int, default=0)
    ap.add_argument("--ctx_rank", type=int, default=0,
                    help="LoRA rank for the ctx ops (w_kctx/w_vctx/v_gate/v_map); "
                         "0=full. d/8 = lora-8.")
    ap.add_argument("--dm_accum", type=int, default=1,
                    help="hybrid recipe: accumulate the DUAL-MOD machinery "
                         "grads over N iters (effective batch N*global) and "
                         "step them every N iters at --dm_lr; trunk steps "
                         "every iter at --lr. 1 = off (single optimizer)")
    ap.add_argument("--dm_lr", type=float, default=None)
    ap.add_argument("--dm_min_lr", type=float, default=None,
                    help="absolute cosine floor for the DM machinery optimizer "
                         "(default 0.1*dm_lr). Set e.g. 1e-6 to cool the recall "
                         "circuit far below the trunk floor at end of training.")
    ap.add_argument("--cooldown_steps", type=int, default=0,
                    help="COOLDOWN ablation: resume, freeze everything except the "
                         "DM recall machinery (w_kctx/w_vctx/v_gate/v_map/w_gk/"
                         "w_gv/norm_ctx/gains) and train recall-only for N steps "
                         "at a linearly-annealed LR, then snapshot + stop. Tests "
                         "whether extra final-LR cooling recovers mqar. 0 = off.")
    ap.add_argument("--cooldown_lr_start", type=float, default=5e-5)
    ap.add_argument("--cooldown_lr_end", type=float, default=1e-6)
    ap.add_argument("--group_clip", choices=["fixed", "calibrated"],
                    default="fixed",
                    help="calibrated: per-group clip at clip_mult x that "
                         "group's own (probe/base) gn EMA, floor 1.0 until "
                         "calibrated (grace steps)")
    ap.add_argument("--clip_mult", type=float, default=2.0)
    ap.add_argument("--dm_all", action="store_true",
                    help="with --dm_accum N: accumulate ALL params (full-"
                         "model mega-batch, FS 768-convention); single "
                         "optimizer at --dm_lr")
    ap.add_argument("--schedule_mode", default="ladder",
                    choices=["ladder", "flat8_probes", "exact16"])
    ap.add_argument("--phase_bounds", default=None,
                    help="comma pair, token-fraction entry of P2,P3 "
                         "(hot-LR ladder stretch: keep K8 while lr is "
                         "above the wedge line)")
    ap.add_argument("--wandb_id", default=None,
                    help="override wandb run id (dead old-config runs pin "
                         "the id namespace; same id would resume them)")
    ap.add_argument("--seq_len", type=int, default=2048,
                    help="context length T (default 2048). Short-T diagnostic: "
                         "does the wedge need long T (chain depth ~ T/c)?")
    ap.add_argument("--d_model", type=int, default=1024)  # head_dim pinned 64;
    # NOTE fused kernels need d/2 == power of 2 (tl.arange) -> d in {512,1024}
    ap.add_argument("--n_layers", type=int, default=24)
    ap.add_argument("--fused_nonlin", default="true",
                    help="run PACKED_CTX_MODS variants (mult/mult_res/fusion)"
                         " through the FusedSweepCell fast path instead of the"
                         " compiled reference cell (gradient-audited via"
                         " analysis/kmod_parity_check.py --prod). true|false")
    ap.add_argument("--sat_tau", type=float, default=0.0,
                    help="saturating backward: per-row cap on inter-cell"
                         " gradient carry at tau x layer-entry norm"
                         " (stability_dynamics.md §8). 0 = off")
    ap.add_argument("--deterministic_bwd", action="store_true",
                    help="route the fused-sweep backward's float atomic_add "
                         "accumulators through a run-to-run bit-identical "
                         "reduction (separate deterministic path). Default off "
                         "= byte-identical to the historical non-det path.")
    ap.add_argument("--recipe", choices=["fs1", "pdn"], default="fs1",
                    help="fs1 = the completed flagship-space convention "
                         "(default, bit-identical to FS-*-s1337); pdn = the "
                         "preconditioned-deltanet paper pipeline "
                         "(plans/pdn_recipe.md): 0.5M-tok batches, 30k steps, "
                         "lr 4e-4 -> 0.1x, warmup 1024 steps, eps 1e-8, wd on "
                         "all params, non-finite-only skip, Mistral-32k data")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--eps", type=float, default=None)
    ap.add_argument("--ladder", choices=["k", "c"], default="k",
                    help="k = FS1 grow-K (default); c = c-anneal at fixed "
                         "K8 (same ratios, tower depth pinned)")
    ap.add_argument("--val_pt", default=None)
    ap.add_argument("--val_batches", type=int, default=0, help="use only the first N held-out batches (smoke runs); 0 = all")
    # --- chain_rho LR governor (dynamic storm-formation guard). The wedge is a
    # backward-chain artifact no structural forward guard reaches; this cuts LR
    # (both groups) transiently as chain_rho climbs and restores to cosine when
    # it recedes -> prevents the storm from FORMING. gov_mult = clip(1 -
    # alpha*log10(rho_ema/trig), floor, 1). Off by default (no-op). ---
    ap.add_argument("--lr_gov", action="store_true")
    ap.add_argument("--lr_gov_trig", type=float, default=20.0)
    ap.add_argument("--lr_gov_alpha", type=float, default=0.5)
    ap.add_argument("--lr_gov_floor", type=float, default=0.15)
    ap.add_argument("--lr_gov_beta", type=float, default=0.9)
    # --- AUTO-REWIND (checkpoint-hop) stability governor. The DM hot recipe
    # wedges: a chaotic backward blowup -> the outlier guard skips ~99% of
    # batches -> the run stalls forever. The wedge is a chaotic knife-edge
    # (bit-reproducible in a continuous run, but a RESUME from a pre-wedge snap
    # lands on a perturbed trajectory that DODGES it). So: on a sustained
    # wedge, reload the last healthy warm-Adam snapshot and PERTURB the data
    # stream (bump the loader seed -> different batches by construction) and
    # continue. Warm Adam (opt + opt_dm) is preserved on reload (re-init causes
    # a damaging transient). Fully OFF by default (no-op; default path
    # byte-identical). ---
    ap.add_argument("--auto_rewind", action="store_true",
                    help="master enable for the self-healing checkpoint-hop "
                         "wedge governor (off = default path untouched)")
    ap.add_argument("--rewind_window", type=int, default=20,
                    help="# of recent steps over which to measure skip rate")
    ap.add_argument("--rewind_skip_thresh", type=int, default=15,
                    help="skips within rewind_window >= this => sustained wedge")
    ap.add_argument("--rewind_lookback", type=int, default=150,
                    help="rewind to a healthy snap >= this many steps before "
                         "the current step (safely pre-onset)")
    ap.add_argument("--rewind_ring", type=int, default=5,
                    help="# of named rewind snapshots to retain")
    ap.add_argument("--rewind_seed_stride", type=int, default=1000,
                    help="seed offset added per rewind (distinct data stream)")
    ap.add_argument("--rewind_max", type=int, default=50,
                    help="safety cap on total rewinds (abort if exceeded)")
    args = ap.parse_args()
    if args.phase_bounds:
        PHASE_BOUNDS[:] = [float(x) for x in args.phase_bounds.split(",")]
    SCHEDULE_MODE[0] = args.schedule_mode
    # recipe defaults (only where the user didn't override)
    dflt = {k: ap.get_default(k) for k in ("global_batch", "total_steps",
                                           "skip_factor", "data_bin")}
    if args.recipe == "pdn":
        if args.global_batch == dflt["global_batch"]:
            args.global_batch = 256
        if args.total_steps == dflt["total_steps"]:
            args.total_steps = 30000
        if args.skip_factor == dflt["skip_factor"]:
            args.skip_factor = 1e30          # non-finite-only, their protocol
        if args.data_bin == dflt["data_bin"]:
            args.data_bin = "data/slimpj627_train.bin"
        if args.lr is None:
            args.lr = 4e-4
        if args.eps is None:
            args.eps = 1e-8
    if args.lr is None:
        args.lr = 3e-4
    if args.eps is None:
        args.eps = 1e-8
    min_lr = 0.1 * args.lr if args.recipe == "pdn" else 3e-5
    pdn_warmup = 1024

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if world > 1:
        import datetime
        dist.init_process_group("nccl",
                                timeout=datetime.timedelta(seconds=1800))
    master = rank == 0

    T = args.seq_len
    mcfg = DualModConfig(ctx_mod=args.ctx_mod,
                         kmod_vraw_heads=args.kmod_vraw_heads,
                         kmod_mode=args.kmod_mode, v_norm=args.v_norm,
                         v_ctx_norm=args.v_ctx_norm,
                         v_clamp_tau=args.v_clamp_tau,
                         mult_res_lambda=args.mult_res_lambda,
                         k_ctx_norm=args.k_ctx_norm,
                         k_gain_init=args.k_gain_init,
                         gate_rank=args.gate_rank,
                         ctx_rank=args.ctx_rank,
                         deterministic_bwd=args.deterministic_bwd,
                         d_model=args.d_model, n_layers=args.n_layers,
                         n_heads=args.d_model // 64,
                         max_seq_len=T, vocab_size=32000,
                         gate_bias_init=0.0, checkpoint_chunk=64)
    torch.manual_seed(args.seed)
    mcfg.gate_init_seed = args.seed
    model = DualModLM(mcfg).cuda()
    B_local = args.global_batch // world
    tok_per_step = args.global_batch * T
    warmup_steps = (1024 if args.recipe == "pdn"
                    else max(1, int(args.warmup_tokens / tok_per_step)))

    out_dir = os.path.join("out", args.run_name)
    start_step = 0
    resume_ck = None
    latest = os.path.join(out_dir, "ckpt_latest.pt")
    if args.resume and os.path.exists(latest):
        resume_ck = torch.load(latest, map_location="cpu", weights_only=False)
        if args.ctx_mod != "linear":
            missing, unexpected = model.load_state_dict(
                resume_ck["model"], strict=False)
            if rank == 0:
                print(f"CONT load: missing={len(missing)} (branch) "
                      f"unexpected={len(unexpected)}", flush=True)
        else:
            model.load_state_dict(resume_ck["model"])
        start_step = resume_ck["step"] + 1
        if master:
            print(f"resuming from step {start_step}", flush=True)

    # no-graphs: on H100 flagship shapes graphs ≡ no-graphs on speed
    # (h100_port.md §3/§4b) and 4-sid capture costs ~7.6GiB pools + capture
    # time; runtime sid switching is free without graphs. B48 + Adam fits at
    # ~64GiB vs ~74GiB captured.
    if args.schedule_mode == "exact16":
        schedules = [uniform_chunks(T, 16, 16)] * 4   # all sids = exact
    else:
        schedules = [uniform_chunks(T, c_, K_)
                     for c_, K_ in LADDERS[args.ladder]]
    from engines.wavescan.cells.scan_cell_fwd import PACKED_CTX_MODS
    nonlin = args.ctx_mod != "linear"
    # PACKED_CTX_MODS (incl. linear) run on the FusedSweepCell fast path; other
    # ctx_mod arms (glu/vmlp/kmlp/...) still fall back to compiled cells. The
    # gate pin for kmod_vraw_bound != none is not packed, so those also fall
    # back (mirrors the graphs.py _needs_verbatim routing).
    fused_ok = (args.fused_nonlin.lower() == "true"
                and args.ctx_mod in PACKED_CTX_MODS
                and getattr(mcfg, "kmod_vraw_bound", "none") == "none")
    # the v'' entry-norm clamp is implemented in the eager refine_kv path
    # (clamp_vpp) and the fused _refine2 kernel (fwd per-head norm reduction +
    # bwd ball-projection Jacobian; parity-validated in analysis/clamp_parity).
    # The only path that does NOT honor it is the FusedRefine linear fallback
    # (use_fused without fused_sweep) — refuse there so a clamp-less run can't
    # masquerade as the clamp arm.
    if (args.v_clamp_tau > 0 or args.k_ctx_norm == "rms") \
            and not fused_ok and args.graphs and not nonlin:
        raise SystemExit(
            "v_clamp_tau / k_ctx_norm are not honored on the FusedRefine "
            "linear path; use the fused_sweep path (default for "
            "PACKED_CTX_MODS) or eager cells (--graphs false).")
    engine = WaveScanEngine(model, schedules, B_local, T, autocast_bf16=True,
                            use_graphs=(args.graphs and (fused_ok or not nonlin)),
                            fused_sweep=fused_ok,
                            compile_cells=(nonlin and not fused_ok),
                            share_layer_pools=(args.graphs and not nonlin),
                            sat_tau=args.sat_tau)
    if master:
        print(f"engine path: fused_sweep={engine.fused_sweep} "
              f"compile_cells={engine.compile_cells} fused_ok={fused_ok}",
              flush=True)
        print(f"engine ready (graphs={args.graphs}); "
              f"params {sum(p.numel() for p in model.parameters())/1e6:.1f}M "
              f"(tied); warmup {warmup_steps} steps", flush=True)

    # no wd on 1-D params (norms + gate biases — doctrine) / wd 0.01 (fla)
    BRANCH = ("v_pre", "v_map", "v_gate", "v_comb", "k_pre")
    branch = [p for n, p in model.named_parameters()
              if any(t in n for t in BRANCH)]
    bids = {id(p) for p in branch}
    decay = [p for p in model.parameters()
             if p.dim() >= 2 and id(p) not in bids]
    nodecay = [p for p in model.parameters()
               if p.dim() < 2 and id(p) not in bids]
    opt_dm, dm_params = None, []
    if args.dm_accum > 1 or args.dm_lr is not None:
        # hybrid recipe: DM machinery on the proven FS temperature
        # (3e-4 @ 768-effective), trunk on titan heat (4e-4 @ 256)
        DM_NAMES = ("w_kctx", "w_vctx", "w_gk", "w_gv", "norm_ctx",
                    "v_gain", "ctx_gain", "k_gain") + BRANCH
        if args.dm_all:
            dm_params = list(model.parameters())
        else:
            dm_params = [p for n, p in model.named_parameters()
                         if any(t in n for t in DM_NAMES)]
        dmids = {id(p) for p in dm_params}
        t_decay = [p for p in model.parameters()
                   if p.dim() >= 2 and id(p) not in dmids]
        t_nodecay = [p for p in model.parameters()
                     if p.dim() < 2 and id(p) not in dmids]
        d_decay = [p for p in dm_params if p.dim() >= 2]
        d_nodecay = [p for p in dm_params if p.dim() < 2]
        trunk_params = t_decay + t_nodecay
        opt = None
        if trunk_params:
            opt = torch.optim.AdamW(
                [{"params": t_decay, "weight_decay": 0.01},
                 {"params": t_nodecay, "weight_decay": 0.0}],
                lr=args.lr, betas=(0.9, 0.95), eps=args.eps)
        opt_dm = torch.optim.AdamW(
            [{"params": d_decay, "weight_decay": 0.01},
             {"params": d_nodecay, "weight_decay": 0.0}],
            lr=args.dm_lr or args.lr, betas=(0.9, 0.95), eps=args.eps)
        if master:
            print(f"HYBRID recipe: trunk {sum(p.numel() for p in t_decay+t_nodecay)/1e6:.1f}M "
                  f"@ lr {args.lr} every iter; dm "
                  f"{sum(p.numel() for p in dm_params)/1e6:.1f}M @ lr "
                  f"{args.dm_lr or args.lr} every {args.dm_accum} iters "
                  f"(effective batch {args.dm_accum}x)", flush=True)
    elif args.recipe == "pdn" and args.wd_doctrine == "pdn":
        # titan builder: wd on ALL params
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                betas=(0.9, 0.95), weight_decay=0.01,
                                eps=args.eps)
    elif True:
        groups = [{"params": decay, "weight_decay": 0.01},
                  {"params": nodecay, "weight_decay": 0.0}]
        if branch:
            # diffusion hold-down: 10x wd on grown-branch params
            groups.append({"params": branch, "weight_decay": 0.1})
        opt = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.95),
                                eps=args.eps)
    # ---- COOLDOWN override: recall-only optimizer, everything else frozen ----
    # Gated by --cooldown_steps>0. Replaces whatever opt/opt_dm were built above
    # with a single fresh AdamW over the DM recall machinery; the rest of the
    # model is frozen (requires_grad=False). Fresh Adam is fine: at cooldown LRs
    # the moments re-warm in a few steps. Resume loads MODEL weights (above); the
    # optimizer-state load below is skipped in cooldown (params/groups differ).
    cooldown = args.cooldown_steps > 0
    if cooldown:
        RECALL = ("w_kctx", "w_vctx", "v_gate", "v_map", "w_gk", "w_gv",
                  "norm_ctx", "k_gain", "v_gain", "ctx_gain")
        # NOTE: do NOT set requires_grad=False on the frozen set — the fused
        # WaveScan backward differentiates w.r.t. tensors it assumes require
        # grad and errors otherwise. Freeze is achieved by only ever stepping
        # the recall optimizer: trunk grads are computed (engine re-zeros each
        # step) but never applied, so those weights never move.
        recall_params, froz = [], 0
        for n, p in model.named_parameters():
            if any(t in n for t in RECALL):
                recall_params.append(p)
            else:
                froz += p.numel()
        opt = torch.optim.AdamW(recall_params, lr=args.cooldown_lr_start,
                                betas=(0.9, 0.95), weight_decay=0.01,
                                eps=args.eps)
        opt_dm, dm_params, trunk_params = None, [], []
        if master:
            print(f"COOLDOWN: recall-only opt on "
                  f"{sum(p.numel() for p in recall_params)/1e6:.2f}M params, "
                  f"NOT stepping {froz/1e6:.1f}M (frozen); {args.cooldown_steps} "
                  f"steps, LR {args.cooldown_lr_start:.1e}->"
                  f"{args.cooldown_lr_end:.1e}", flush=True)
    gn_ema = {False: None, True: None}   # keyed by is_probe
    gn_cnt = {False: 0, True: 0}
    dm_buf = {id(p): torch.zeros_like(p, dtype=torch.float32)
              for p in dm_params}
    dm_count = 0
    clip_ema, clip_cnt = {}, {}
    if cooldown:
        pass  # cooldown uses the fresh recall-only opt built above (no load)
    elif resume_ck is not None and "optimizer" not in resume_ck:
        # model-only snap as continuation base (the pristine full ckpt was
        # rotated away): fresh Adam on a mature trunk — expect a re-warmup
        # transient for the first few hundred steps; guard rides it
        if master:
            print("CONT: base has no optimizer state — fresh Adam "
                  "(re-warmup transient expected)", flush=True)
    elif resume_ck is not None:
        opt_sd = resume_ck["optimizer"]
        if len(opt_sd["param_groups"]) == 1 and len(opt.param_groups) == 2:
            # doctrine change mid-run: transplant single-group Adam state
            # into the decay/nodecay split by parameter identity (ckpt group
            # order == model.parameters() order)
            id2old = {id(p): i for i, p in enumerate(model.parameters())}
            new_state, new_groups, gi = {}, [], 0
            for g_new in opt.param_groups:
                idxs = []
                for p in g_new["params"]:
                    old = opt_sd["state"].get(id2old[id(p)])
                    if old is not None:
                        new_state[gi] = old
                    idxs.append(gi)
                    gi += 1
                ng = {k: v for k, v in opt_sd["param_groups"][0].items()
                      if k != "params"}
                ng["params"] = idxs
                ng["weight_decay"] = g_new["weight_decay"]
                new_groups.append(ng)
            opt_sd = {"state": new_state, "param_groups": new_groups}
            if master:
                print(f"WD-DOCTRINE transplant: 1 group -> 2 "
                      f"({len(new_state)} param states carried)", flush=True)
        elif len(opt_sd["param_groups"]) == 2 and len(opt.param_groups) == 3:
            # continuation: 2-group fs1 ckpt -> +branch group. Shared params
            # keep relative order within decay/nodecay; branch starts fresh
            new_state, new_groups, gi, old_gi = {}, [], 0, 0
            for gidx, g_new in enumerate(opt.param_groups):
                idxs = []
                for p in g_new["params"]:
                    if gidx < 2:
                        old = opt_sd["state"].get(old_gi)
                        if old is not None:
                            new_state[gi] = old
                        old_gi += 1
                    idxs.append(gi)
                    gi += 1
                ng = {k: v for k, v in opt_sd["param_groups"][0].items()
                      if k != "params"}
                ng["params"] = idxs
                ng["weight_decay"] = g_new["weight_decay"]
                new_groups.append(ng)
            opt_sd = {"state": new_state, "param_groups": new_groups}
            if master:
                print(f"CONT transplant: trunk state carried "
                      f"({len(new_state)}), branch fresh", flush=True)
        opt.load_state_dict(opt_sd)
        if opt_dm is not None and "optimizer_dm" in resume_ck:
            # the DM optimizer (where the storm lives) must ride too, else
            # resume = fresh DM-Adam -> re-warmup transient (large 1st step,
            # bad first val). Faithful only if the ckpt carries optimizer_dm.
            opt_dm.load_state_dict(resume_ck["optimizer_dm"])
            if master:
                print("resumed opt_dm (DM Adam) state", flush=True)
        elif opt_dm is not None and master:
            print("WARN: no optimizer_dm in ckpt -> fresh DM-Adam "
                  "(re-warmup transient)", flush=True)
        gn_ema = resume_ck.get("gn_ema", gn_ema)
        gn_cnt = resume_ck.get("gn_cnt", gn_cnt)

    stream = memmap_stream(args.data_bin, T, B_local, rank, world,
                           start_step=start_step)
    if args.val_pt is None:
        args.val_pt = ("data/slimpj627_val.pt" if args.recipe == "pdn"
                       else "data/slimpj_val.pt")
    val = torch.load(args.val_pt)                     # [21, 48, 2049]
    if args.val_batches:
        val = val[:args.val_batches]
    if val.shape[-1] > T + 1:                          # short-T runs: val is
        val = val[..., :T + 1].contiguous()            # chunked at 2048 -> slice
    x_tel = val[0, :8, :-1].cuda()                     # to T (fixes engine/eager/tel)

    run, metrics_f = None, None
    if master:
        os.makedirs(out_dir, exist_ok=True)
        metrics_f = open(os.path.join(out_dir, "metrics.jsonl"), "a")
        if args.wandb_log == "true":
            import wandb
            cfg_w = {**mcfg.to_dict(), "data": "slimpajama-15b",
                     "global_batch": args.global_batch,
                     "total_steps": args.total_steps,
                     "schedule": "c64 K8->16->32, probe 2x 10%",
                     "seed": args.seed, "nodes": 2}
            # NOTE: wandb rewind (resume_from=f"{id}?_step={n}") would let a
            # post-crash replay overwrite the poisoned tail, but it's
            # private-preview (400 from the API on this account). With plain
            # resume, replayed steps below the old max are dropped by the
            # monotonic-step check — metrics.jsonl keeps the full record.
            run = wandb.init(project="dualmod", name=args.run_name,
                             id=args.wandb_id or args.run_name,
                             resume="allow", config=cfg_w)

    def log(d, step):
        if master:
            metrics_f.write(json.dumps({"step": step, **d}) + "\n")
            metrics_f.flush()
            if run is not None:
                run.log(d, step=step)

    def save_full(step):
        tmp = latest + ".tmp"
        torch.save({"model": model.state_dict(),
                    **({"optimizer": opt.state_dict()} if opt is not None else {}),
                    **({"optimizer_dm": opt_dm.state_dict()} if opt_dm is not None else {}),
                    "step": step, "applied": applied, "model_cfg": mcfg.to_dict(),
                    "gn_ema": gn_ema, "gn_cnt": gn_cnt}, tmp)
        if os.path.exists(latest):
            os.replace(latest, os.path.join(out_dir, "ckpt_prev.pt"))
        os.replace(tmp, latest)

    t0 = time.time()
    skipped_total = 0
    consec = {False: [], True: []}
    rho_ema, gov_mult = None, 1.0            # chain_rho LR governor state
    # --- auto-rewind state (all no-op unless --auto_rewind). recent_skips,
    # rewind_ring and reload_count are maintained IDENTICALLY on every rank
    # (the skip decision is bit-synced across ranks, and the ring step/path are
    # a pure function of step) so the trigger + chosen snapshot are rank-
    # identical and every rank reloads the same file. ---
    import collections
    recent_skips = collections.deque(maxlen=args.rewind_window)
    rewind_ring = []                         # list of (step, path), asc by step
    reload_count = 0
    # Loop is a while (not a for) so a rewind can reset `step` backward. With
    # --auto_rewind OFF nothing in the body touches `step` and the trailing
    # `step += 1` makes this byte-identical to `for step in range(...)`.
    step = start_step
    applied = (resume_ck.get("applied", start_step)
               if resume_ck is not None else 0)
    cd_start_applied = applied   # cooldown counts accepted steps from here
    # SAM (arm F) perturb-set = ctx ops ONLY (w_kctx/w_vctx = the R4 coordinate);
    # empty unless --sam. Alarm batch is a fixed val slice reused each fire so
    # the curvature dial is comparable step-to-step (master builds it lazily).
    sam_params = ([p for n, p in model.named_parameters()
                   if p.requires_grad and ("w_kctx" in n or "w_vctx" in n)]
                  if (args.sam or args.curv_alarm_interval > 0) else [])
    if master and args.sam:
        print(f"SAM armed: rho={args.sam_rho} every={args.sam_every} on "
              f"{len(sam_params)} ctx tensors "
              f"({sum(p.numel() for p in sam_params)/1e6:.1f}M params); "
              f"NO rewind, NO lr_gov -> pure storm-prevention test", flush=True)
    curv_alarm_x = curv_alarm_y = None
    curv_warned = False
    while (applied if args.skip_free_sched else step) < args.total_steps:
        if cooldown and (applied - cd_start_applied) >= args.cooldown_steps:
            if master:
                torch.save({"model": model.state_dict(),
                            "step": step, "applied": applied,
                            "model_cfg": mcfg.to_dict()},
                           os.path.join(out_dir, f"snap_cooldown_{step:06d}.pt"))
                print(f"COOLDOWN done: {applied - cd_start_applied} accepted "
                      f"steps, saved snap_cooldown_{step:06d}.pt", flush=True)
            break
        sidx = applied if args.skip_free_sched else step
        sid, is_probe = sid_of(sidx, args.total_steps, warmup_steps)
        if args.force_sid is not None:
            # diagnostic override: clamp base AND probe batches to one rung
            # (storm-zone (c,K) A/B tests; probes off so the rung is pure)
            sid, is_probe = args.force_sid, False
        if cooldown:
            prog = min(1.0, (applied - cd_start_applied)
                       / max(1, args.cooldown_steps))
            cd_lr = (args.cooldown_lr_start
                     + (args.cooldown_lr_end - args.cooldown_lr_start) * prog)
            for g in opt.param_groups:
                g["lr"] = cd_lr
        else:
            if opt_dm is not None:
                dml = args.dm_lr or args.lr
                dm_floor = (args.dm_min_lr if args.dm_min_lr is not None
                            else 0.1 * dml)
                for g in opt_dm.param_groups:
                    g["lr"] = lr_at(sidx, args.total_steps, warmup_steps,
                                    lr=dml, min_lr=dm_floor) * gov_mult
            for g in (opt.param_groups if opt is not None else []):
                g["lr"] = lr_at(sidx, args.total_steps, warmup_steps,
                                lr=args.lr, min_lr=min_lr) * gov_mult
        blk = next(stream).cuda(non_blocking=True)
        x, y = blk[:, :-1].contiguous(), blk[:, 1:].contiguous()

        t_step0 = time.perf_counter()
        do_sam = (args.sam and not is_probe and sam_params
                  and step % args.sam_every == 0)
        sam_sharp = float("nan")
        if do_sam:
            # 1st fwd/bwd at theta -> base grads (the ascent direction)
            loss = engine.train_step(x, y, sid).clone()
            engine.finish_grads(world_size=world)
            with torch.no_grad():
                gsq = sum((p.grad.float() ** 2).sum()
                          for p in sam_params if p.grad is not None)
                gnorm = float(gsq.sqrt()) + 1e-12
                scale = args.sam_rho / gnorm
                gbase = [(p.grad.detach().float().clone()
                          if p.grad is not None else None) for p in sam_params]
                for p, gb in zip(sam_params, gbase):
                    if gb is not None:
                        p.data.add_(gb.to(p.dtype), alpha=scale)   # ascend to theta+eps
            # 2nd fwd/bwd at theta+eps -> grads the optimizer will actually use
            # (engine re-zeros its grad buffers each train_step, so this is a
            # clean replacement, not an accumulation onto the base grads)
            loss = engine.train_step(x, y, sid).clone()
            engine.finish_grads(world_size=world)
            with torch.no_grad():
                # free along-gradient sharpness: <ghat,(g'-g)>/rho, ghat=g/||g||.
                # = ghat^T H ghat to O(rho) — SAM's own curvature probe (the
                # chain_rho-squared coordinate sampled along the gradient).
                dot = sum(((p.grad.float() - gb) * gb).sum()
                          for p, gb in zip(sam_params, gbase)
                          if p.grad is not None and gb is not None)
                sam_sharp = float(dot) / (gnorm * args.sam_rho)
                for p, gb in zip(sam_params, gbase):        # restore theta
                    if gb is not None:
                        p.data.sub_(gb.to(p.dtype), alpha=scale)
        else:
            loss = engine.train_step(x, y, sid).clone()
            engine.finish_grads(world_size=world)
        step_ms = (time.perf_counter() - t_step0) * 1000.0
        if args.lr_gov and step >= warmup_steps:
            # update the chain_rho governor for the NEXT step's LR (1-step lag).
            # rho_ema smooths lone spikes; log-scaled cut tracks an exponential
            # climb and auto-releases to cosine (gov_mult->1) as rho recedes.
            rn = engine.chain_rho_max.item()
            if not math.isfinite(rn):
                rn = 1e30
            b = args.lr_gov_beta
            rho_ema = rn if rho_ema is None else b * rho_ema + (1 - b) * rn
            gov_mult = min(1.0, max(args.lr_gov_floor, 1.0 - args.lr_gov_alpha
                                    * math.log10(max(rho_ema, 1e-9) / args.lr_gov_trig)))
        if opt_dm is None:
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        else:
            # hybrid: trunk clips per iter on its OWN norm; DM accumulates
            # UNCLIPPED (FS semantics = one clip of the mega-batch mean,
            # applied at the mega step below; guard-skip voids wild iters)
            def _thr(key):
                e, c = clip_ema.get(key), clip_cnt.get(key, 0)
                if args.group_clip == "calibrated" and e is not None \
                        and c >= args.skip_grace_steps:
                    return args.clip_mult * e
                return 1.0
            gn_t = (torch.nn.utils.clip_grad_norm_(
                        trunk_params, _thr(("t", is_probe))).item()
                    if trunk_params else 0.0)
            with torch.no_grad():
                gn_d = torch.norm(torch.stack(
                    [p.grad.norm() for p in dm_params
                     if p.grad is not None])).item()
            gn = (gn_t ** 2 + gn_d ** 2) ** 0.5

        # outlier batch-skip: EMA per batch type; identical grad bytes on all
        # ranks (bitwise_sync) -> identical gn -> identical decision.
        # Non-finite gn/loss ALWAYS skips: NaN fails every > comparison, so
        # without this a NaN backward walks straight into Adam (this exact
        # failure killed the run at step 6943 — K64 probe backward NaN'd,
        # skip stayed False, weights poisoned).
        ema = gn_ema[is_probe]
        skip = (not math.isfinite(gn) or not math.isfinite(loss.item())
                or (ema is not None
                    and gn_cnt[is_probe] >= args.skip_grace_steps
                    and gn > args.skip_factor * ema))
        # deadlock breaker (learned on FS-E: a genuine gn level-shift freezes
        # the accepted-only EMA and skips everything): 5 consecutive FINITE
        # skips of the same batch type re-warm that type's EMA
        if skip and math.isfinite(gn) and math.isfinite(loss.item()):
            consec[is_probe].append(gn)
            if len(consec[is_probe]) >= 5:
                lo, hi = min(consec[is_probe][-5:]), max(consec[is_probe][-5:])
                # re-warm ONLY on a genuine level shift (flat gns, like the
                # FS-E 0.27->1.5 episode); wild swings (bomb storms, engine
                # corruption) must NEVER re-warm — PDN-DM-s42's breaker
                # re-warmed to 1e4 amid a 1e3-1e15 storm and blinded the guard
                if hi < 2.0 * lo and gn < 50.0 * gn_ema[is_probe]:
                    # flat AND near the old level: FS-E's genuine shift was
                    # 5.6x; tower storms run 100-1000x and must never re-warm
                    # even when coincidentally flat
                    if master:
                        print(f"SKIP-BREAKER step {step}: ema[{is_probe}] "
                              f"{gn_ema[is_probe]:.3f} -> {gn:.3f}", flush=True)
                    gn_ema[is_probe], skip = gn, False
                    consec[is_probe] = []
        elif not skip:
            consec[is_probe] = []
        if skip:
            if opt is not None:
                opt.zero_grad(set_to_none=False)
            skipped_total += 1
            if opt_dm is not None:
                # pollution guard: a skipped iter voids the accumulation
                for p in dm_params:
                    dm_buf[id(p)].zero_()
                dm_count = 0
        else:
            applied += 1
            if opt is not None:
                opt.step()
            if opt_dm is not None:
                for p in dm_params:
                    if p.grad is not None:
                        dm_buf[id(p)] += p.grad
                dm_count += 1
                if dm_count == args.dm_accum:
                    for p in dm_params:
                        if p.grad is not None:
                            p.grad.copy_(dm_buf[id(p)]).div_(args.dm_accum)
                    # single clip of the mega-batch mean (FS convention);
                    # calibrated per-group threshold when enabled
                    torch.nn.utils.clip_grad_norm_(
                        dm_params, _thr(("d", is_probe)))
                    opt_dm.step()
                    for p in dm_params:
                        dm_buf[id(p)].zero_()
                    dm_count = 0
            gn_ema[is_probe] = gn if ema is None else 0.98 * ema + 0.02 * gn
            gn_cnt[is_probe] += 1
            if opt_dm is not None:
                for key, gval in ((("t", is_probe), gn_t),
                                  (("d", is_probe), gn_d)):
                    e = clip_ema.get(key)
                    clip_ema[key] = gval if e is None else 0.98 * e + 0.02 * gval
                    clip_cnt[key] = clip_cnt.get(key, 0) + 1
        if opt is not None:
            opt.zero_grad(set_to_none=False)
        if opt_dm is not None:
            opt_dm.zero_grad(set_to_none=False)

        lr_now = (opt if opt is not None else opt_dm).param_groups[0]["lr"]
        d = {"train/loss": loss.item(), "train/lr": lr_now,
             "train/grad_norm": gn, "train/sid": sid,
             "train/phase": phase_of(step, args.total_steps),
             "train/clip_event": int(gn > 1.0), "train/skipped": int(skip),
             "train/skipped_total": skipped_total,
             "train/applied": applied,
             "train/tokens": (step + 1) * tok_per_step,
             "train/wall_s": time.time() - t0,
             "train/step_ms": step_ms}
        if args.sam:
            d["train/sam_active"] = int(do_sam)
            if do_sam:
                d["train/sam_sharp"] = sam_sharp   # free along-grad curvature
        if args.sat_tau > 0:
            d["train/sat_events"] = int(engine.sat_events.item())
            d["train/sat_rows_zeroed"] = int(engine.sat_rows_zeroed.item())
            rho = engine.chain_rho_max.item()
            d["train/chain_rho_max"] = rho if math.isfinite(rho) else 1e30
        if args.lr_gov:
            d["train/lr_gov_mult"] = gov_mult
            d["train/rho_ema"] = float(rho_ema) if rho_ema is not None else 0.0
        if is_probe:
            d["probe/loss"] = loss.item()
            d["probe/grad_norm"] = gn
            if gn_ema[False] is not None:
                d["probe/gn_ratio_vs_base"] = gn / max(gn_ema[False], 1e-9)
        else:
            d["base/grad_norm"] = gn
        if skip:
            if master:
                ema_s = f"{gn_ema[is_probe]:.2f}" if gn_ema[is_probe] is not None else "-"
                print(f"SKIP step {step}: gn {gn:.2f} vs ema "
                      f"{ema_s} (probe={is_probe})", flush=True)
            # skip decision is rank-identical (bitwise_sync grads), so every
            # rank reaches this collective. rank-0-only debug_scan was a blind
            # spot: residue/origin on ranks 1..N-1 reported as NONE
            if not math.isfinite(gn) or gn > 1e3:
                bad = engine.debug_scan()
                lg = getattr(engine, "local_gn", float("nan"))
                if world > 1:
                    gathered = [None] * world
                    torch.distributed.all_gather_object(
                        gathered, (torch.distributed.get_rank(), lg, bad[:6]))
                else:
                    gathered = [(0, lg, bad[:6])]
                if master:
                    wild = [(r, f"{g:.3g}", b) for r, g, b in gathered
                            if b or not math.isfinite(g) or g > 1e3]
                    print(f"DEBUG-SCAN step {step}: corrupted="
                          f"{bad[:12] or 'NONE(transient)'} "
                          f"per-rank-wild={wild or 'NONE'}", flush=True)
        log(d, step)
        if master and step % 50 == 0:
            print(f"step {step}/{args.total_steps} loss {loss.item():.4f} "
                  f"gn {gn:.2f} sid {sid} ({(time.time()-t0)/3600:.2f} h)",
                  flush=True)

        if master and (step % args.eval_interval == 0
                       or step == args.total_steps - 1):
            d = {"val/loss": engine_val(engine, val)}
            if (args.curv_alarm_interval > 0 and sam_params
                    and step % args.curv_alarm_interval == 0):
                # direction-free R4 dial, master-only + timed. Best-effort:
                # any failure logs nan once and never disturbs training.
                try:
                    if curv_alarm_x is None:
                        nb = min(args.curv_alarm_rows, val.shape[1])
                        # short T: the L26 create_graph scan is ~per-iter-bound,
                        # so cap tokens to keep the dial to ~minutes not ~20min
                        ct = min(257, val.shape[-1])
                        cb = val[0, :nb, :ct].cuda()
                        curv_alarm_x = cb[:, :-1].contiguous()
                        curv_alarm_y = cb[:, 1:].contiguous()
                    _t = time.perf_counter()
                    d["curv/ctx_lambda_max"] = ctx_top_eig(
                        model, curv_alarm_x, curv_alarm_y, sam_params,
                        n_iter=args.curv_alarm_iters)
                    d["curv/alarm_ms"] = (time.perf_counter() - _t) * 1000.0
                except Exception as e:
                    if not curv_warned:
                        print(f"curv-alarm disabled (fired once): {e}",
                              flush=True)
                        curv_warned = True
                    d["curv/ctx_lambda_max"] = float("nan")
            if step % (2 * args.eval_interval) == 0:
                d.update(gate_stats_layerwise(model, x_tel))
            if (step % args.eager_val_interval == 0
                    and engine.B <= val.shape[1]):
                # same-batch pair: engine vs eager on val[0] full 48 rows —
                # directly comparable (audits: should agree to ~1e-4).
                # Skipped when B_local > val rows (engine buffers are static)
                nb = min(48, engine.B)
                d["val/loss_eager_xcheck"] = eager_val(model, val, rows=nb)
                with torch.no_grad():
                    d["val/loss_engine_xcheck"] = engine.forward(
                        val[0, :engine.B, :-1].contiguous().cuda(), EXACT_SID,
                        targets=val[0, :engine.B, 1:].contiguous().cuda()).item()
            log(d, step)

        if master and step % args.ckpt_interval == 0 and step > start_step:
            save_full(step)
        if master and (step % args.snap_interval == 0
                       or step == args.total_steps - 1
                       or phase_of(step, args.total_steps) !=
                       phase_of(min(step + 1, args.total_steps - 1),
                                args.total_steps)):
            torch.save({"model": model.state_dict(),
                        **({"optimizer": opt.state_dict()} if opt is not None else {}),
                        **({"optimizer_dm": opt_dm.state_dict()}
                           if opt_dm is not None else {}),
                        "step": step, "applied": applied,
                        "model_cfg": mcfg.to_dict(),
                        "gn_ema": gn_ema, "gn_cnt": gn_cnt},
                       os.path.join(out_dir, f"snap_{step:06d}.pt"))

        # ============================ AUTO-REWIND ========================
        # Self-healing checkpoint-hop wedge governor. Entirely gated behind
        # --auto_rewind; when off, `step` is untouched here and the trailing
        # `step += 1` reproduces `for step in range(...)` exactly.
        if args.auto_rewind:
            recent_skips.append(1 if skip else 0)
            # --- healthy-snapshot ring: named warm-Adam snaps every
            # snap_interval, ONLY when the recent window is skip-free (never
            # ring-save mid-wedge). Written by master, tracked on all ranks. ---
            if step % args.snap_interval == 0 and sum(recent_skips) == 0:
                rw_path = os.path.join(out_dir, f"snap_rw_{step:06d}.pt")
                # always refresh the file (a rewind can revisit this step with a
                # new trajectory), but keep the ring keyed by unique step so a
                # prune never os.remove()s a path a duplicate entry still needs.
                if master:
                    torch.save(
                        {"model": model.state_dict(),
                         **({"optimizer": opt.state_dict()}
                            if opt is not None else {}),
                         **({"optimizer_dm": opt_dm.state_dict()}
                            if opt_dm is not None else {}),
                         "step": step, "applied": applied,
                         "model_cfg": mcfg.to_dict(),
                         "gn_ema": gn_ema, "gn_cnt": gn_cnt}, rw_path)
                if step not in [s for s, _ in rewind_ring]:
                    rewind_ring.append((step, rw_path))
                    while len(rewind_ring) > args.rewind_ring:
                        old_step, old_path = rewind_ring.pop(0)
                        if master and os.path.exists(old_path) and \
                                old_step not in [s for s, _ in rewind_ring]:
                            os.remove(old_path)
            # --- wedge detection + rewind. Trigger + chosen snap are rank-
            # identical (recent_skips + rewind_ring identical on every rank). ---
            n_skips = sum(recent_skips)
            if n_skips >= args.rewind_skip_thresh and rewind_ring:
                cands = [(s, p) for (s, p) in rewind_ring
                         if s <= step - args.rewind_lookback]
                chosen_step, chosen_path = (cands[-1] if cands
                                            else rewind_ring[0])
                reload_count += 1
                if reload_count > args.rewind_max:
                    if master:
                        print(f"[AUTO-REWIND] ABORT: reload_count "
                              f"{reload_count} > rewind_max {args.rewind_max} "
                              f"at step {step} — persistent wedge, giving up "
                              f"to avoid an infinite rewind loop.", flush=True)
                    break
                new_seed = args.seed + args.rewind_seed_stride * reload_count
                detected_step = step
                rho_ema_val = float(rho_ema) if rho_ema is not None else 0.0
                if world > 1:
                    dist.barrier()
                ck = torch.load(chosen_path, map_location="cpu",
                                weights_only=False)
                model.load_state_dict(ck["model"])
                if opt is not None and "optimizer" in ck:
                    opt.load_state_dict(ck["optimizer"])
                if opt_dm is not None and "optimizer_dm" in ck:
                    opt_dm.load_state_dict(ck["optimizer_dm"])
                gn_ema = ck["gn_ema"]
                gn_cnt = ck["gn_cnt"]
                # rebuild the data stream at the reloaded step with a bumped
                # seed -> a perturbed (dodging) trajectory by construction.
                step = ck["step"] + 1
                applied = ck.get("applied", ck["step"])
                stream = memmap_stream(args.data_bin, T, B_local, rank, world,
                                       start_step=ck["step"] + 1, seed=new_seed)
                # reset guard/loop/accum state (mirror the skip-path void)
                recent_skips.clear()
                rho_ema, gov_mult = None, 1.0
                consec = {False: [], True: []}
                for p in dm_params:
                    dm_buf[id(p)].zero_()
                dm_count = 0
                if master:
                    print(f"[AUTO-REWIND #{reload_count}] step "
                          f"{detected_step} WEDGED ({n_skips}/"
                          f"{args.rewind_window} skips, rho_ema="
                          f"{rho_ema_val}) -> reload snap@{ck['step']} (lost "
                          f"{detected_step - ck['step']} steps), "
                          f"new_seed={new_seed}", flush=True)
                    if run is not None:
                        run.log({"train/rewind_count": reload_count,
                                 "train/rewind_event": 1}, step=detected_step)
                    with open(os.path.join(out_dir, "rewind_events.jsonl"),
                              "a") as rf:
                        rf.write(json.dumps({
                            "reload_num": reload_count,
                            "detected_step": detected_step,
                            "reload_from_step": ck["step"],
                            "lost_steps": detected_step - ck["step"],
                            "skips_in_window": n_skips,
                            "rewind_window": args.rewind_window,
                            "rho_ema": rho_ema_val,
                            "new_seed": new_seed,
                            "wall_s": time.time() - t0}) + "\n")
                # restart the loop AT the reloaded step (its batch fetched next);
                # `continue` intentionally skips the trailing `step += 1`.
                continue
        # ========================== /AUTO-REWIND =========================
        step += 1

    if master:
        save_full(args.total_steps - 1)
        print(f"done: {args.total_steps} steps in "
              f"{(time.time()-t0)/3600:.2f} h, skipped {skipped_total}",
              flush=True)
        metrics_f.close()
        if run is not None:
            run.finish()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
