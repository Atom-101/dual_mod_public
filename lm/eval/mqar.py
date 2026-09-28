"""MQAR (multi-query associative recall) eval for DualModLM checkpoints —
zoology-style synthetic: N key-value pairs presented once, then M queries;
score = fraction of queried values predicted exactly (greedy argmax).

Vocab: keys/values drawn from mid-range Llama-2 token ids (plain word-piece
tokens); format uses single-token keys and values so the metric is pure
recall, not tokenization luck. Deterministic per (seed, config).

python lm/eval/mqar.py --ckpt out/FS-DM-s1337/ckpt_latest.pt \
    --pairs 8,16,32,64 --seq_len 2048 --n_seq 64
Writes analysis/mqar_<run>_<step>.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from models.dualmod.config import DualModConfig
from models.dualmod.model import DualModLM

SEP, EQ, Q = 29892, 29922, 29973     # ",", "=", "?" llama-2 word pieces


def build_batch(n_pairs, n_queries, seq_len, n_seq, seed, vocab_lo=3000,
                vocab_hi=28000):
    g = torch.Generator().manual_seed(seed)
    xs, tgt_pos, tgt_tok = [], [], []
    for s in range(n_seq):
        ids = torch.randperm(vocab_hi - vocab_lo, generator=g)[:2 * n_pairs] + vocab_lo
        keys, vals = ids[:n_pairs], ids[n_pairs:]
        seq = [1]                                   # BOS
        for k, v in zip(keys, vals):
            seq += [k.item(), EQ, v.item(), SEP]
        qi = torch.randperm(n_pairs, generator=g)[:n_queries]
        pos, toks = [], []
        for q in qi:
            seq += [Q, keys[q].item(), EQ]
            pos.append(len(seq) - 1)                # predict value AT this pos
            toks.append(vals[q].item())
            seq += [vals[q].item(), SEP]
        assert len(seq) <= seq_len, "config too large for seq_len"
        seq += [SEP] * (seq_len - len(seq))         # pad tail (causal-safe)
        xs.append(seq)
        tgt_pos.append(pos)
        tgt_tok.append(toks)
    return (torch.tensor(xs, dtype=torch.long),
            tgt_pos, tgt_tok)


@torch.no_grad()
def run(model, n_pairs, n_queries, seq_len, n_seq, seed, device, rows=32):
    x, tgt_pos, tgt_tok = build_batch(n_pairs, n_queries, seq_len, n_seq, seed)
    hit = tot = 0
    for r0 in range(0, n_seq, rows):
        xb = x[r0:r0 + rows].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=device == "cuda"):
            if getattr(model, "arch", "dualmod") == "gdn":
                logits = model(input_ids=xb).logits
            else:
                logits, _ = model(xb)
        pred = logits.float().argmax(-1)
        for i in range(xb.shape[0]):
            for p, t in zip(tgt_pos[r0 + i], tgt_tok[r0 + i]):
                # tgt_pos is the EQ index; logits AT the EQ position predict
                # the value token (off-by-one fixed: pred[p-1] scored the key
                # position, i.e. P(next|key)=EQ — a constant, always ~0 acc)
                hit += int(pred[i, p].item() == t)
                tot += 1
    return hit / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pairs", default="8,16,32,64")
    ap.add_argument("--queries", type=int, default=8)
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--n_seq", type=int, default=64)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    arch = ck.get("arch", "dualmod")
    if arch == "gdn":
        from fla.models import GatedDeltaNetConfig, GatedDeltaNetForCausalLM
        cfg = GatedDeltaNetConfig(**{k: v for k, v in ck["model_cfg"].items()
                                     if k not in ("model_type", "architectures")})
        model = GatedDeltaNetForCausalLM(cfg).to(args.device).eval()
    else:
        cfg = DualModConfig(**ck["model_cfg"])
        if cfg.attn_mode != "vanilla":
            cfg.attn_mode = "sequential"
        model = DualModLM(cfg).to(args.device).eval()
    model.load_state_dict(ck["model"])
    model.arch = arch

    res = {}
    for n in [int(p) for p in args.pairs.split(",")]:
        acc = run(model, n, min(args.queries, n), args.seq_len, args.n_seq,
                  args.seed, args.device)
        res[f"pairs_{n}"] = acc
        print(f"MQAR pairs={n:3d}: acc {acc:.3f}", flush=True)

    run_name = os.path.basename(os.path.dirname(args.ckpt))
    out = f"analysis/mqar_{run_name}_{ck.get('step', -1)}.json"
    json.dump({"ckpt": args.ckpt, "step": ck.get("step", -1),
               "queries": args.queries, "n_seq": args.n_seq,
               "seed": args.seed, "results": res}, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
