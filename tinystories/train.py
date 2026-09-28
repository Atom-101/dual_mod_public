"""Training driver (plan.md §8, §11).

Runs:
  A (baseline):   python -m tinystories.train --preset A
  B (dual-mod):   python -m tinystories.train --preset B
  C (value-only): python -m tinystories.train --preset C

Both runs use the identical token stream (fixed data_seed), pure next-token
cross-entropy, AdamW(0.9, 0.95), wd 0.1 (none on norms/gate biases/embeddings),
lr 3e-4 cosine → 3e-5, 1% warmup, grad clip 1.0, bf16 autocast with fp32
softmax/gate-sigmoid inside the attention. Metrics go to wandb and to
out/<run>/metrics.jsonl (used by scripts/t6_overlay.py).
"""

import argparse
import contextlib
import json
import math
import os
import pickle
import time
from dataclasses import fields


@contextlib.contextmanager
def _nullctx():
    yield

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from models.dualmod.config import DualModConfig, TrainConfig
from models.dualmod.data import PackedLoader
from models.dualmod.model import DualModLM
from models.dualmod.telemetry import collect_telemetry, grad_norm_ratios

PRESETS = {
    "A": dict(attn_mode="vanilla", enable_key_mod=False, enable_value_mod=False,
              run_name="A-vanilla"),
    "B": dict(attn_mode="sequential", enable_key_mod=True, enable_value_mod=True,
              run_name="B-dualmod"),
    "C": dict(attn_mode="sequential", enable_key_mod=False, enable_value_mod=True,
              run_name="C-valuemod"),
    "D": dict(attn_mode="sequential", enable_key_mod=True, enable_value_mod=False,
              run_name="D-keymod"),
    # Placement-matched capacity controls (vanilla SDPA, +12.58M in the attention
    # sublayer, residual stream untouched). F1 keeps KV cache identical to A/B.
    "F1": dict(attn_mode="vanilla", enable_key_mod=False, enable_value_mod=False,
               n_heads=32, n_kv_heads=8, head_dim_override=64,
               run_name="F1-qwiden-gqa"),
    "F2": dict(attn_mode="vanilla", enable_key_mod=False, enable_value_mod=False,
               n_heads=20, n_kv_heads=0, head_dim_override=64,
               run_name="F2-widemha"),
    # F3: B's architecture exactly (param/placement/gate/init identical), but the
    # context branch reads x̂ instead of RMSNorm(o) — deletes the contextual content.
    # No recurrence → parallel SDPA-speed run.
    "F3": dict(attn_mode="static", enable_key_mod=True, enable_value_mod=True,
               run_name="F3-staticbranch"),
    # G-A: UGI gate-bias init on both paths (per-dim spread at mean 0.5). Tests
    # whether the SHAPE of the init matters at fixed level vs scalar bias-0.
    "G-A": dict(attn_mode="sequential", enable_key_mod=True, enable_value_mod=True,
                gate_bias_init_k="ugi", gate_bias_init_v="ugi", run_name="G-A-ugi"),
    # G-B: solution-shaped init — committed keys (13 raw / 13 ctx / 6 frontier per head)
    # + band values U[0.2,0.8] + reserved-address W_Kctx init. Mean 0.5 (matches G0/G-A).
    "G-B": dict(attn_mode="sequential", enable_key_mod=True, enable_value_mod=True,
                gate_bias_init_k="committed", gate_bias_init_v="band", run_name="G-B-shaped"),
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=list(PRESETS), default=None)
    for cfg_cls in (DualModConfig, TrainConfig):
        for f in fields(cfg_cls):
            t = f.type if isinstance(f.type, type) else type(f.default)
            if t is bool:
                ap.add_argument(f"--{f.name}", type=lambda s: s.lower() in ("1", "true", "yes"),
                                default=None)
            else:
                ap.add_argument(f"--{f.name}", type=type(f.default), default=None)
    ap.add_argument("--resume", type=str, default=None)
    return ap.parse_args()


def build_configs(args):
    over = dict(PRESETS[args.preset]) if args.preset else {}
    mcfg_kw, tcfg_kw = {}, {}
    for f in fields(DualModConfig):
        v = getattr(args, f.name, None)
        if v is not None:
            mcfg_kw[f.name] = v
        elif f.name in over:
            mcfg_kw[f.name] = over[f.name]
    for f in fields(TrainConfig):
        v = getattr(args, f.name, None)
        if v is not None:
            tcfg_kw[f.name] = v
        elif f.name in over:
            tcfg_kw[f.name] = over[f.name]
    return DualModConfig(**mcfg_kw), TrainConfig(**tcfg_kw)


def make_optimizer(model, tcfg):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # wd on weight matrices only; none on norms, biases (gates), embeddings
        if p.dim() >= 2 and "tok_emb" not in n and "lm_head" not in n:
            decay.append(p)
        else:
            no_decay.append(p)
    groups = [{"params": decay, "weight_decay": tcfg.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=tcfg.lr, betas=(tcfg.beta1, tcfg.beta2),
                             fused=True)


def lr_at(step, total_steps, tcfg):
    warmup = max(1, int(tcfg.warmup_frac * total_steps))
    if step < warmup:
        return tcfg.lr * (step + 1) / warmup
    prog = (step - warmup) / max(1, total_steps - warmup)
    return tcfg.min_lr + 0.5 * (tcfg.lr - tcfg.min_lr) * (1 + math.cos(math.pi * prog))


@torch.no_grad()
def eval_val_loss(model, loader, tcfg, autocast_ctx):
    model.eval()
    losses = []
    for i in range(tcfg.eval_iters):
        x, y = loader.val_batch(i)
        with autocast_ctx():
            _, loss = model(x, targets=y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def ddp_setup():
    """Returns (rank, local_rank, world_size, is_ddp). Honors torchrun env vars."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size, True
    return 0, 0, 1, False


def main():
    args = parse_args()
    mcfg, tcfg = build_configs(args)
    # tie the structured-gate-init permutation seed to the training seed unless set
    if getattr(args, "gate_init_seed", None) is None:
        mcfg.gate_init_seed = tcfg.seed
    rank, local_rank, world_size, is_ddp = ddp_setup()
    master = rank == 0
    device = f"cuda:{local_rank}" if is_ddp else tcfg.device
    torch.manual_seed(tcfg.seed)
    torch.cuda.manual_seed_all(tcfg.seed)

    out_dir = os.path.join(tcfg.out_dir, tcfg.run_name)
    if master:
        os.makedirs(out_dir, exist_ok=True)

    raw_model = DualModLM(mcfg).to(device)
    model = raw_model
    if is_ddp:
        # static_graph: our params are reused many times across the sequential scan
        # under activation checkpointing; the default dynamic bucket-rebuild path is
        # incompatible with that (negative-dim crash in _rebuild_buckets).
        model = DDP(raw_model, device_ids=[local_rank], static_graph=True)
    if tcfg.compile and mcfg.attn_mode == "vanilla":
        model = torch.compile(model)

    breakdown = raw_model.param_breakdown()
    tokens_per_step = tcfg.batch_size * tcfg.grad_accum * mcfg.max_seq_len
    total_steps = int(math.ceil(tcfg.tokens_target / tokens_per_step))
    if master:
        print(f"run={tcfg.run_name} mode={mcfg.attn_mode} key_mod={mcfg.enable_key_mod} "
              f"value_mod={mcfg.enable_value_mod} ddp={is_ddp} world_size={world_size}")
        print(f"params: base={breakdown['base']:,} dualmod_extra={breakdown['dualmod_extra']:,} "
              f"total={breakdown['total']:,}")
        print(f"global tokens/step={tokens_per_step:,} total_steps={total_steps:,} "
              f"target={tcfg.tokens_target:.3g} tokens")

    train_loader = PackedLoader(tcfg.data_dir, "train", mcfg.max_seq_len,
                                tcfg.batch_size, tcfg.data_seed, device,
                                rank=rank, world_size=world_size)
    val_loader = PackedLoader(tcfg.data_dir, "val", mcfg.max_seq_len,
                              tcfg.batch_size, tcfg.data_seed, device,
                              rank=rank, world_size=world_size)
    with open(os.path.join(tcfg.data_dir, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    function_mask = torch.from_numpy(meta["function_mask"])
    x_telemetry, _ = val_loader.val_batch(0)  # fixed batch for §10

    optimizer = make_optimizer(model, tcfg)
    start_step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        raw_model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        start_step = ck["step"] + 1
        print(f"resumed from {args.resume} at step {start_step}")

    use_bf16 = (mcfg.dtype == "bf16")
    autocast_ctx = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if use_bf16 \
        else (lambda: torch.autocast("cuda", enabled=False))

    run = None
    metrics_f = None
    if master:
        if tcfg.wandb_log:
            import wandb
            run = wandb.init(project=tcfg.wandb_project, name=tcfg.run_name,
                             config={**mcfg.to_dict(), **tcfg.to_dict(),
                                     "total_steps": total_steps,
                                     "world_size": world_size, **breakdown},
                             dir=out_dir)
        metrics_f = open(os.path.join(out_dir, "metrics.jsonl"), "a")

    def log(d, step):
        if not master:
            return
        d = {"step": step, "tokens": step * tokens_per_step, **d}
        metrics_f.write(json.dumps({k: v for k, v in d.items()
                                    if isinstance(v, (int, float, str))}) + "\n")
        metrics_f.flush()
        if run is not None:
            import wandb
            payload = {k: (wandb.Histogram(v) if isinstance(v, np.ndarray) else v)
                       for k, v in d.items() if k not in ("step",)}
            run.log(payload, step=step)

    model.train()
    t0 = time.time()
    running_loss, running_n = 0.0, 0
    for step in range(start_step, total_steps):
        lr = lr_at(step, total_steps, tcfg)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        micro_loss = 0.0
        for micro in range(tcfg.grad_accum):
            x, y = train_loader.train_batch(step * tcfg.grad_accum + micro)
            # only sync grads on the last micro-step (DDP averages once per step)
            sync = (not is_ddp) or (micro == tcfg.grad_accum - 1)
            sync_ctx = _nullctx() if sync else model.no_sync()
            with sync_ctx:
                with autocast_ctx():
                    _, loss = model(x, targets=y)
                (loss / tcfg.grad_accum).backward()
            micro_loss += loss.item() / tcfg.grad_accum

        telemetry_step = (master and mcfg.attn_mode != "vanilla"
                          and step % tcfg.telemetry_interval == 0)
        extra = grad_norm_ratios(raw_model) if telemetry_step else {}
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        optimizer.step()

        running_loss += micro_loss
        running_n += 1
        if step % tcfg.log_interval == 0:
            dt = time.time() - t0
            toks = (step + 1 - start_step) * tokens_per_step
            log({"train/loss": running_loss / running_n, "train/lr": lr,
                 "train/grad_norm": gnorm.item(),
                 "train/tok_per_s": toks / dt, "train/wall_s": dt}, step)
            running_loss, running_n = 0.0, 0

        if master and (step % tcfg.eval_interval == 0 or step == total_steps - 1):
            vl = eval_val_loss(model, val_loader, tcfg, autocast_ctx)
            log({"val/loss": vl}, step)
            print(f"step {step:6d}/{total_steps} val_loss {vl:.4f} "
                  f"lr {lr:.2e} elapsed {time.time()-t0:.0f}s", flush=True)

        if telemetry_step:
            scalars, hists = collect_telemetry(raw_model, x_telemetry, function_mask)
            log({**scalars, **extra, **hists}, step)

        if master and ((step > 0 and step % tcfg.ckpt_interval == 0)
                       or step == total_steps - 1):
            torch.save({"model": raw_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": step, "model_cfg": mcfg.to_dict(),
                        "train_cfg": tcfg.to_dict()},
                       os.path.join(out_dir, "ckpt.pt"))

    if master:
        if metrics_f is not None:
            metrics_f.close()
        if run is not None:
            run.finish()
        print(f"done: {total_steps} steps, {total_steps * tokens_per_step:,} tokens, "
              f"{time.time()-t0:.0f}s wall")
    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
