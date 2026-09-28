"""Recall-focused MQAR post-training of the PRETRAINED flagship checkpoints
(DM / E-match / GDN), then acc-vs-pairs sweep. This is the discriminating
recall column: zero-shot floors everyone; post-training measures whether the
trained model can be adapted to high-load recall — fixed-state models hit
their capacity wall, attention-based should not.

Protocol: VALUE-POSITION masked CE (identical objective all arms) on
synthetic KV sequences in Llama-2 token space, T=512, n_pairs ~ U{8..110},
short (default 1500 steps @ bsz 24, lr 1e-4 cosine). v1 used full-NTP at
lr 2e-5: both arms learned only the format (loss -> the no-recall
theoretical floor ~2.45, acc 0 at every load) — the recall signal is 1.6%
of NTP loss mass. Masked loss at lr 1e-4 is what the from-scratch runs
solved the task with. Eval: value-position exact-match acc at pairs
{8,16,32,64,100}, 256 held-out seqs each.

  DM:   python lm/eval/posttrain_mqar.py --ckpt out/FS-DM-s1337/snap_009536.pt
  GDN:  python lm/eval/posttrain_mqar.py --ckpt out/FS-GDN-s1337/snap_009536.pt
Writes analysis/mqar_post_<run>.json
"""

import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lm.eval.mqar import build_batch, run as mqar_eval

T = 512
EVAL_PAIRS = (8, 16, 32, 64, 100)


def load_arm(ckpt):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    arch = ck.get("arch", "dualmod")
    if arch == "gdn":
        from fla.models import GatedDeltaNetConfig, GatedDeltaNetForCausalLM
        cfg = GatedDeltaNetConfig(**{k: v for k, v in ck["model_cfg"].items()
                                     if k not in ("model_type", "architectures")})
        model = GatedDeltaNetForCausalLM(cfg)
        kind = "gdn"
    else:
        from models.dualmod.config import DualModConfig
        from models.dualmod.model import DualModLM
        cfg = DualModConfig(**ck["model_cfg"])
        kind = "vanilla" if cfg.attn_mode == "vanilla" else "dualmod"
        if kind == "dualmod":
            cfg.attn_mode = "sequential"
        model = DualModLM(cfg)
    model.load_state_dict(ck["model"])
    model = model.cuda()
    model.arch = "gdn" if kind == "gdn" else "dualmod"
    return model, kind, ck


def make_train_batch(bsz, gen):
    """Variable-load MQAR rows + value-position mask, [bsz, T+1]."""
    rows, masks = [], []
    for _ in range(bsz):
        p = int(torch.randint(8, 111, (1,), generator=gen))
        q = min(8, p)
        x, tgt_pos, _ = build_batch(p, q, T + 1, 1,
                                    int(torch.randint(0, 2 ** 31, (1,), generator=gen)))
        m = torch.zeros(T + 1, dtype=torch.bool)
        m[torch.tensor(tgt_pos[0]) + 1] = True   # +1: tgt_pos is the EQ index,
        # the VALUE sits one right; v2 marked EQ and trained P(EQ|key)=const
        # (loss 0.000 instantly, recall untouched)
        rows.append(x[0]); masks.append(m)
    return torch.stack(rows), torch.stack(masks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--bsz", type=int, default=24)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    model, kind, ck = load_arm(args.ckpt)
    run_name = os.path.basename(os.path.dirname(args.ckpt))
    print(f"{run_name} ({kind}): "
          f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.0)
    gen = torch.Generator().manual_seed(args.seed)

    def sweep(tag):
        model.eval()
        res = {}
        for p in EVAL_PAIRS:
            res[f"pairs_{p}"] = mqar_eval(model, p, min(8, p), T + 1, 256,
                                          9999, "cuda")
        print(f"{tag}: " + " ".join(f"p{p}={res[f'pairs_{p}']:.3f}"
                                    for p in EVAL_PAIRS), flush=True)
        return res

    pre = sweep("pre ")
    for step in range(args.steps):
        lr = args.lr * 0.5 * (1 + math.cos(math.pi * step / args.steps))
        for g in opt.param_groups:
            g["lr"] = lr
        blk, m = make_train_batch(args.bsz, gen)
        blk, m = blk.cuda(), m.cuda()
        x = blk[:, :-1].contiguous()
        model.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if kind == "gdn":
                logits = model(input_ids=x).logits
            else:
                logits, _ = model(x)
        pos = m.nonzero(as_tuple=True)          # value positions in blk coords
        import torch.nn.functional as F
        loss = F.cross_entropy(logits[pos[0], pos[1] - 1].float(), blk[pos])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 100 == 0:
            print(f"step {step}: loss {loss.item():.4f}", flush=True)
        if step in (399, 799):
            sweep(f"@{step+1}")

    post = sweep("post")
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out = os.path.join(repo, "analysis", f"mqar_post_{run_name}.json")
    json.dump({"ckpt": args.ckpt, "kind": kind, "steps": args.steps,
               "pre": pre, "post": post}, open(out, "w"), indent=1)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
