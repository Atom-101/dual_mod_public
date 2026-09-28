"""Held-out SlimPajama loss for a checkpoint: eager exact scan mean-CE on
data/slimpj_val.pt (the common val tensor), row-chunked fp32 CE.
Comparable to the in-run engine val (agreement ~1e-4 per audits).

python lm/eval/heldout_loss.py --ckpt out/FS-DM-s1337/snap_007000.pt --n_batches 8
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
import torch.nn.functional as F

from models.dualmod.config import DualModConfig
from models.dualmod.model import DualModLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n_batches", type=int, default=8)
    ap.add_argument("--val", default="data/slimpj_val.pt")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    arch = ck.get("arch", "dualmod")
    if arch == "gdn":
        from fla.models import GatedDeltaNetConfig, GatedDeltaNetForCausalLM
        cfg = GatedDeltaNetConfig(**{k: v for k, v in ck["model_cfg"].items()
                                     if k not in ("model_type", "architectures")})
        model = GatedDeltaNetForCausalLM(cfg).cuda().eval()
    else:
        cfg = DualModConfig(**ck["model_cfg"])
        if cfg.attn_mode != "vanilla":
            cfg.attn_mode = "sequential"
        model = DualModLM(cfg).cuda().eval()
    model.load_state_dict(ck["model"])
    val = torch.load(args.val)

    ls = []
    with torch.no_grad():
        for i in range(min(args.n_batches, val.shape[0])):
            blk = val[i].cuda()
            x, y = blk[:, :-1], blk[:, 1:]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if arch == "gdn":
                    logits = model(input_ids=x).logits
                else:
                    logits, _ = model(x)
            tot, cnt = 0.0, 0
            for r0 in range(0, x.shape[0], 8):
                yc = y[r0:r0 + 8]
                tot += F.cross_entropy(
                    logits[r0:r0 + 8].float().flatten(0, 1),
                    yc.reshape(-1), reduction="sum").item()
                cnt += yc.numel()
            ls.append(tot / cnt)
            print(f"batch {i}: {ls[-1]:.4f}", flush=True)
    mean = float(np.mean(ls))
    print(f"heldout mean CE {mean:.4f}  ppl {np.exp(mean):.2f} "
          f"({len(ls)} batches x {val.shape[1]} rows)", flush=True)
    run = os.path.basename(os.path.dirname(args.ckpt))
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    split = os.path.splitext(os.path.basename(args.val))[0]
    out = os.path.join(repo, "analysis",
                       f"heldout_{run}_{ck.get('step', -1)}_{split}.json")
    json.dump({"ckpt": args.ckpt, "step": ck.get("step", -1),
               "mean_ce": mean, "per_batch": ls}, open(out, "w"), indent=1)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
