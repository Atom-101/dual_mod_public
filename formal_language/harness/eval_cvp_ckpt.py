"""Re-evaluate a CVP DM checkpoint in EAGER mode (no engine): python formal_language/harness/eval_cvp_ckpt.py <ckpt> [--no_kmod] [--rel]"""
import os, sys, numpy as np, torch; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from formal_language.harness.train_statetrack_ws import build_dm
from formal_language.harness.statetrack_gen import make_cvp_batch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False); a = ck["args"]; sd = ck["model"]
n, m, L = a["n_ops"], a["cvp_inputs"], a.get("n_layers", 0); nmax = sd["tok_emb.weight"].shape[0] - 8 - m - 4   # layout from the vocab
model = build_dm(a["scale"], sd["tok_emb.weight"].shape[0], 512, a["ctx_mod"], L, key_mod=("--no_kmod" not in sys.argv)).cuda().eval()
miss, unexp = model.load_state_dict(sd, strict=False); print("missing", len(miss), "unexpected", len(unexp))
tok, tgt, msk, depth = make_cvp_batch(np.random.default_rng(7), 128, n, m=m, n_max=nmax, m_max=m, n_topo=16, addr=("rel" if "--rel" in sys.argv else "abs"))
x = torch.from_numpy(tok).long().cuda(); tg = torch.from_numpy(tgt).long().cuda(); mk = torch.from_numpy(msk).bool().cuda()
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    out = model(x); lg = out[0] if isinstance(out, tuple) else out
pred = lg.argmax(-1); ok = (pred[mk] == tg[mk]); d = depth[msk]
print(f"eager acc {ok.float().mean().item():.4f}; by depth:", {k: round(ok[torch.from_numpy(d==k).cuda()].float().mean().item(),3) for k in (1,2,4,8,12,16,24,32) if (d==k).any()})
