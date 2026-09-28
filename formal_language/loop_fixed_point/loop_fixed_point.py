"""Fixed-point diagnostic for the looped (UT-style) baseline: run a trained LoopedLM for K iterations and report, per
iteration k, the relative state change ||x_k - x_{k-1}|| / ||x_k|| (mean over tokens) and the masked accuracy of the
readout applied to x_k. A collapse to a fixed point shows as the change -> 0 while accuracy stays flat.
Usage: python formal_language/loop_fixed_point/loop_fixed_point.py <ckpt.pt> --K 64 (task read from ckpt: cvp | group | keyed; keyed adds acc at the deepest register depth D)"""
import os
import argparse, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from formal_language.harness.train_statetrack import SCALES
from models.dualmod.config import DualModConfig
from models.looped.looped_lm import LoopedLM

ap = argparse.ArgumentParser(); ap.add_argument("ckpt"); ap.add_argument("--K", type=int, default=64)
ap.add_argument("--B", type=int, default=128)
ap.add_argument("--n_ops", type=int, default=0, help="override sequence size (A5 length / CVP gates); 0 = the training size. Larger = out-of-distribution input")
ap.add_argument("--D", type=int, default=0, help="keyed: override ops per register (depth); 0 = training D")
ap.add_argument("--every", action="store_true", help="print every iteration k (default: k<=8, multiples of 8, K)")
a = ap.parse_args()
ck = torch.load(a.ckpt, map_location="cpu", weights_only=False); args = ck["args"]; sd = ck["model"]
vocab = sd["tok_emb.weight"].shape[0]; s = SCALES[args["scale"]]
cfg = DualModConfig(attn_mode="vanilla", d_model=s["d"], n_layers=1, n_heads=s["h"], max_seq_len=4096,
                    vocab_size=vocab, gate_bias_init=0.0, checkpoint_chunk=0, use_rope=not args.get("nope", False))
model = LoopedLM(cfg, n_unique=args["loop_layers"], k_max=args["loop_k"], huginn=bool(args.get("loop_huginn", False)), bp_iters=int(args.get("loop_bp", 0)), emb_scale=bool(args.get("loop_emb_scale", False))).cuda().eval()   # step index clamps at k_max beyond training K (as in the harness eval sweep)
missing, unexpected = model.load_state_dict(sd, strict=False); print("load: missing", len(missing), "unexpected", len(unexpected))
rng = np.random.default_rng(0)
if args["task"] == "cvp":
    from formal_language.harness.statetrack_gen import make_cvp_batch
    tok, tgt, msk, de = make_cvp_batch(rng, a.B, a.n_ops or args["n_ops"], m=args["cvp_inputs"], fmt=args["cvp_format"], n_max=args["max_ops"],
                                      n_topo=16, m_max=args["cvp_inputs"])
    de = np.asarray(de)
    if de.shape == msk.shape: deep = torch.from_numpy((de == de[msk.astype(bool)].max()) & msk.astype(bool)).cuda()
    else:                                                        # [B, n_q] sidecar: queries in row order
        dm = np.zeros_like(msk, dtype=bool); dmax = de.max()
        for b in range(msk.shape[0]):
            pos = np.flatnonzero(msk[b]); dm[b, pos[:len(de[b])]] = (de[b][:len(pos)] == dmax)
        deep = torch.from_numpy(dm).cuda()
elif args["task"] == "group":
    from formal_language.harness.statetrack_gen import make_group_tagged_batch, cayley
    mul, n = cayley(args["group"]); tok, tgt, msk = make_group_tagged_batch(rng, a.B, a.n_ops or args["n_ops"], mul, n)[:3]
    T_ = tok.shape[1]; deep = torch.from_numpy(msk.astype(bool) & (np.arange(T_)[None, :] >= (7 * T_) // 8)).cuda()   # deepest eighth of positions
elif args["task"] == "keyed":
    from formal_language.harness.statetrack_gen import make_keyed_group_batch, cayley
    mul, n = cayley(args["group"]); km = args.get("K_max") or args["K"]
    Dv = a.D or args["D"]
    tok, tgt, msk, depth = make_keyed_group_batch(rng, a.B, mul, n, args["K"], Dv, args.get("rho", 0.0), K_max=km)
    deep = torch.from_numpy(depth == Dv).cuda()          # deepest-register queries
else:
    raise SystemExit("task not supported")
deep = locals().get("deep", None)
x = torch.from_numpy(tok).long().cuda(); tg = torch.from_numpy(tgt).long().cuda(); mk = torch.from_numpy(msk).bool().cuda()
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    _, _, outs = model(x, K=a.K, return_all=True)
    print(f"{'k':>3s} {'rel_change':>11s} {'acc':>7s}" + ("" if deep is None else f" {'acc_D':>7s}"))
    prev = None
    for k, xk in enumerate(outs, 1):
        xk = xk.float(); ch = float("nan") if prev is None else ((xk - prev).norm(dim=-1) / (xk.norm(dim=-1) + 1e-6)).mean().item()
        xr = model.coda(xk.to(outs[0].dtype), model.rope_cos[:xk.shape[1]], model.rope_sin[:xk.shape[1]]).float() if model.huginn else xk   # Huginn: readout goes through the coda block, as in training
        pred = model.lm_head(model.norm_f(xr)).argmax(-1); acc = (pred[mk] == tg[mk]).float().mean().item()
        extra = "" if deep is None else f" {(pred[mk & deep] == tg[mk & deep]).float().mean().item():7.3f}"
        if a.every or k <= 8 or k % 8 == 0 or k == a.K: print(f"{k:3d} {ch:11.3e} {acc:7.3f}{extra}")
        prev = xk
