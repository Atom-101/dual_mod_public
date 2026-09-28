"""Flagship baseline arms (plans/flagship_run.md §2) on the IDENTICAL token
stream + recipe as FS-DM-s1337 (seed 42 stream is a pure function of step;
768-seq global batch per step is byte-identical regardless of world size).

  --arch ematch  vanilla DualModLM widened to DM params:
                 d1216/19h/L24/mlp3520 = 489.1M (DM 487.4M, Δ+0.34%)
  --arch gdn     fla GatedDeltaNet-340M, published config verbatim
                 (lm/train_340m/gdn340m_config.json, 366.8M) — needs flash-linear-attention

Same AdamW(0.9,0.95) wd0.01 (no wd on 1-D), lr 3e-4 cosine→3e-5, warmup
0.5B tok, clip 1.0, EMA outlier batch-skip with non-finite ALWAYS skipped
(identical stability protocol to the DM arm — comparability outranks).
Micro-batched grad accumulation (gradient-identical), torch DDP.

Launch (single node, 8 GPUs, B_local=96):
  torchrun --nproc_per_node=8 lm/train_340m/train_baseline.py \
      --run_name FS-E-s1337 --arch ematch --seed 1337
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


def lr_at(step, total, warmup_steps, lr=3e-4, min_lr=3e-5):
    if step < warmup_steps:
        return lr * (step + 1) / warmup_steps
    prog = (step - warmup_steps) / max(1, total - warmup_steps)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * prog))


def memmap_stream(path, seq_len, batch_size, rank, world, start_step=0,
                  seed=42, dtype=np.uint16):
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


def build_model(arch, seed):
    torch.manual_seed(seed)
    if arch == "ematch":
        from models.dualmod.config import DualModConfig
        from models.dualmod.model import DualModLM
        cfg = DualModConfig(attn_mode="vanilla", d_model=1216, n_heads=19,
                            n_layers=24, max_seq_len=2048, vocab_size=32000,
                            mlp_hidden_override=3520)
        model = DualModLM(cfg)
        return model, cfg.to_dict(), "dualmod"
    if arch == "vanilla340":
        # T++ 340M-class control: plain attention, d1024/L24 = 336.4M
        from models.dualmod.config import DualModConfig
        from models.dualmod.model import DualModLM
        cfg = DualModConfig(attn_mode="vanilla", d_model=1024, n_heads=16,
                            n_layers=24, max_seq_len=2048, vocab_size=32000)
        model = DualModLM(cfg)
        return model, cfg.to_dict(), "dualmod"
    else:
        from fla.models import GatedDeltaNetConfig, GatedDeltaNetForCausalLM
        cfg = GatedDeltaNetConfig(**{k: v for k, v in json.load(
            open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gdn340m_config.json"))).items()
            if k not in ("model_type", "architectures")})
        model = GatedDeltaNetForCausalLM(cfg)
        return model, cfg.to_dict(), "fla"


def loss_fn(kind, model, blk):
    """blk [b, T+1]. Returns mean next-token CE over T targets — identical
    target mapping for both kinds."""
    if kind == "dualmod":
        x, y = blk[:, :-1].contiguous(), blk[:, 1:].contiguous()
        _, loss = model(x, targets=y)
        return loss
    # fla causal LM shifts labels internally: input=labels=blk gives
    # predictions blk[1:] from blk[:-1] — same mapping
    out = model(input_ids=blk, labels=blk)
    return out.loss


@torch.no_grad()
def common_val(kind, model, val, micro=8):
    model.eval()
    ls = []
    for i in range(val.shape[0]):
        blk = val[i].cuda()
        sub = []
        for r0 in range(0, blk.shape[0], micro):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                l = loss_fn(kind, model, blk[r0:r0 + micro])
            sub.append(l.item())
        ls.append(float(np.mean(sub)))
    model.train()
    return float(np.mean(ls))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--arch", choices=["ematch", "gdn", "vanilla340"],
                required=True)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--total_steps", type=int, default=None)
    ap.add_argument("--global_batch", type=int, default=None)
    ap.add_argument("--warmup_tokens", type=float, default=0.5e9)
    ap.add_argument("--data_bin", default=None)
    ap.add_argument("--micro", type=int, default=8)
    ap.add_argument("--eval_interval", type=int, default=250)
    ap.add_argument("--snap_interval", type=int, default=1000)
    ap.add_argument("--ckpt_interval", type=int, default=250)
    ap.add_argument("--skip_factor", type=float, default=None)
    ap.add_argument("--skip_grace_steps", type=int, default=100)
    ap.add_argument("--wandb_log", default="false")
    ap.add_argument("--wandb_id", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--recipe", choices=["fs1", "pdn"], default="fs1",
                    help="fs1 = the completed flagship-space convention "
                         "(default, bit-identical to FS-*-s1337); pdn = the "
                         "preconditioned-deltanet paper pipeline "
                         "(plans/pdn_recipe.md): 0.5M-tok batches, 30k steps, "
                         "lr 4e-4 -> 0.1x, warmup 1024 steps, eps 1e-8, wd on "
                         "all params, non-finite-only skip, Mistral-32k data")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--val_pt", default=None)
    ap.add_argument("--eps", type=float, default=None)
    args = ap.parse_args()
    # recipe defaults (None-sentinel: an explicitly passed value always
    # sticks, even if it coincides with the other recipe's default — the
    # old ==-argparse-default test silently swapped TPP-llamapdn's
    # --data_bin /tmp/slimpj_train.bin for the Mistral bin)
    rd = ({"global_batch": 256, "total_steps": 30000, "skip_factor": 1e30,
           "data_bin": "data/slimpj627_train.bin", "lr": 4e-4, "eps": 1e-8}
          if args.recipe == "pdn" else
          {"global_batch": 768, "total_steps": 9537, "skip_factor": 5.0,
           "data_bin": "data/slimpj_train.bin", "lr": 3e-4, "eps": 1e-8})
    for k, v in rd.items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    min_lr = 0.1 * args.lr if args.recipe == "pdn" else 3e-5
    pdn_warmup = 1024

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if world > 1:
        dist.init_process_group("nccl")
    master = rank == 0

    T = 2048
    model, cfg_dict, kind = build_model(args.arch, args.seed)
    model = model.cuda()
    B_local = args.global_batch // world
    tok_per_step = args.global_batch * T
    warmup_steps = (1024 if args.recipe == "pdn"
                    else max(1, int(args.warmup_tokens / tok_per_step)))

    out_dir = os.path.join("out", args.run_name)
    latest = os.path.join(out_dir, "ckpt_latest.pt")
    start_step = 0
    resume_ck = None
    if args.resume and os.path.exists(latest):
        resume_ck = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_state_dict(resume_ck["model"])
        start_step = resume_ck["step"] + 1
        if master:
            print(f"resuming from step {start_step}", flush=True)

    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank])
    raw = model.module if world > 1 else model
    if master:
        print(f"{args.arch}: {sum(p.numel() for p in raw.parameters())/1e6:.1f}M "
              f"params | B_local {B_local} micro {args.micro} | warmup "
              f"{warmup_steps}", flush=True)

    if args.recipe == "pdn":       # titan builder: wd on ALL params
        opt = torch.optim.AdamW(raw.parameters(), lr=args.lr,
                                betas=(0.9, 0.95), weight_decay=0.01,
                                eps=args.eps)
    else:
        decay = [p for p in raw.parameters() if p.dim() >= 2]
        nodecay = [p for p in raw.parameters() if p.dim() < 2]
        opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01},
                                 {"params": nodecay, "weight_decay": 0.0}],
                                lr=args.lr, betas=(0.9, 0.95), eps=args.eps)
    gn_ema, gn_cnt = None, 0
    if resume_ck is not None:
        opt.load_state_dict(resume_ck["optimizer"])
        gn_ema = resume_ck.get("gn_ema")
        gn_cnt = resume_ck.get("gn_cnt", 0)

    stream = memmap_stream(args.data_bin, T, B_local, rank, world,
                           start_step=start_step)
    val = torch.load(args.val_pt if args.val_pt else
                     ("data/slimpj627_val.pt" if args.recipe == "pdn"
                      else "data/slimpj_val.pt"))

    run, metrics_f = None, None
    if master:
        os.makedirs(out_dir, exist_ok=True)
        metrics_f = open(os.path.join(out_dir, "metrics.jsonl"), "a")
        if args.wandb_log == "true":
            import wandb
            run = wandb.init(project="dualmod", name=args.run_name,
                             id=args.wandb_id or args.run_name,
                             resume="allow",
                             config={"arch": args.arch, **cfg_dict,
                                     "global_batch": args.global_batch,
                                     "total_steps": args.total_steps,
                                     "seed": args.seed})

    def log(d, step):
        if master:
            metrics_f.write(json.dumps({"step": step, **d}) + "\n")
            metrics_f.flush()
            if run is not None:
                run.log(d, step=step)

    def save_full(step):
        tmp = latest + ".tmp"
        torch.save({"model": raw.state_dict(), "optimizer": opt.state_dict(),
                    "step": step, "model_cfg": cfg_dict, "arch": args.arch,
                    "gn_ema": gn_ema, "gn_cnt": gn_cnt}, tmp)
        if os.path.exists(latest):
            os.replace(latest, os.path.join(out_dir, "ckpt_prev.pt"))
        os.replace(tmp, latest)

    model.train()
    t0 = time.time()
    skipped_total = 0
    consec_skips = []
    acc = max(1, B_local // args.micro)
    for step in range(start_step, args.total_steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.total_steps, warmup_steps,
                            lr=args.lr, min_lr=min_lr)
        blk = next(stream).cuda(non_blocking=True)
        lsum = 0.0
        for m_ in range(acc):
            mb = blk[m_ * args.micro:(m_ + 1) * args.micro]
            ctx = model.no_sync() if (world > 1 and m_ < acc - 1) else \
                torch.enable_grad()
            with ctx:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    l = loss_fn(kind, model, mb)
                (l / acc).backward()
            lsum += l.item() / acc
        gn = torch.nn.utils.clip_grad_norm_(raw.parameters(), 1.0).item()

        skip = (not math.isfinite(gn) or not math.isfinite(lsum)
                or (gn_ema is not None and gn_cnt >= args.skip_grace_steps
                    and gn > args.skip_factor * gn_ema))
        # deadlock breaker: the EMA only updates on accepted steps, so a
        # genuine gn level-shift freezes it and every batch skips forever
        # (hit on FS-E at step 1017: gn drifted 0.27->1.5, all skipped).
        # N consecutive FINITE skips at a stable level = new normal.
        if skip and math.isfinite(gn):
            consec_skips.append(gn)
            if len(consec_skips) >= 5:
                lo, hi = min(consec_skips[-5:]), max(consec_skips[-5:])
                if hi < 2.0 * lo:     # re-warm only on a FLAT level shift
                    if master:
                        print(f"SKIP-BREAKER step {step}: re-warming ema "
                              f"{gn_ema:.3f} -> {gn:.3f}", flush=True)
                    gn_ema, skip = gn, False
                    consec_skips = []
        elif not skip:
            consec_skips = []
        if skip:
            skipped_total += 1
        else:
            opt.step()
            gn_ema = gn if gn_ema is None else 0.98 * gn_ema + 0.02 * gn
            gn_cnt += 1
        opt.zero_grad(set_to_none=True)

        log({"train/loss": lsum, "train/lr": opt.param_groups[0]["lr"],
             "train/grad_norm": gn, "train/clip_event": int(gn > 1.0),
             "train/skipped": int(skip), "train/skipped_total": skipped_total,
             "train/tokens": (step + 1) * tok_per_step,
             "train/wall_s": time.time() - t0}, step)
        if skip and master:
            ema_s = f"{gn_ema:.2f}" if gn_ema is not None else "-"
            print(f"SKIP step {step}: gn {gn:.2f} vs ema {ema_s}", flush=True)
        if master and step % 50 == 0:
            print(f"step {step}/{args.total_steps} loss {lsum:.4f} gn {gn:.2f} "
                  f"({(time.time()-t0)/3600:.2f} h)", flush=True)

        if master and (step % args.eval_interval == 0
                       or step == args.total_steps - 1):
            log({"val/loss": common_val(kind, raw, val)}, step)
        if master and step % args.ckpt_interval == 0 and step > start_step:
            save_full(step)
        if master and (step % args.snap_interval == 0
                       or step == args.total_steps - 1):
            torch.save({"model": raw.state_dict(), "step": step,
                        "model_cfg": cfg_dict, "arch": args.arch},
                       os.path.join(out_dir, f"snap_{step:06d}.pt"))

    if master:
        save_full(args.total_steps - 1)
        print(f"done: {args.total_steps} steps in {(time.time()-t0)/3600:.2f} h,"
              f" skipped {skipped_total}", flush=True)
        metrics_f.close()
        if run is not None:
            run.finish()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
