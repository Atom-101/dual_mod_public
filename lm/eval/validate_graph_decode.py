"""Equivalence test: CUDA-graph static-cache decode vs the eager model.decode_step scan.
  CUDA_VISIBLE_DEVICES=0 python lm/eval/validate_graph_decode.py --ckpt <ckpt> [--n 8 --L 1024 --gen 64]
Prints per-position teacher-forced logit diffs + argmax agreement, free-run continuation
match rate, and PASS/FAIL (argmax agreement >= 99.9%, max|dlogit| < 0.1, >= 7/8 continuations identical)."""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from lm.eval.lm_eval_adapter import DualModEval
from lm.eval.graph_decode import GraphDecoder


def sample_prompts(tok, n, L, seed=0):
    """Real text from the FineWeb-Edu val stream (deterministic, no network); else wikitext; else random."""
    import numpy as np, os
    vb = os.environ.get("VAL_BIN", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "fwedu_100b", "fwedu_val.bin"))
    if os.path.exists(vb):
        arr = np.memmap(vb, dtype=np.uint16, mode="r")
        g = torch.Generator().manual_seed(seed)
        starts = torch.randint(0, len(arr) - L - 1, (n,), generator=g).tolist()
        x = torch.tensor(np.stack([arr[s:s + L] for s in starts]).astype("int64"))
        x[:, 0] = tok.bos_token_id
        return x, "fwedu_val"
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = " ".join(t for t in ds["text"] if t.strip())
        ids = tok(text, add_special_tokens=False)["input_ids"]
        g = torch.Generator().manual_seed(seed)
        starts = torch.randint(0, len(ids) - L - 1, (n,), generator=g).tolist()
        return torch.tensor([[tok.bos_token_id] + ids[s:s + L - 1] for s in starts]), "wikitext"
    except Exception as e:  # noqa
        g = torch.Generator().manual_seed(seed)
        x = torch.randint(3, 32000, (n, L), generator=g); x[:, 0] = tok.bos_token_id
        return x, f"random ({type(e).__name__})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--tok_dir", default="data/llama2_tok")
    ap.add_argument("--n", type=int, default=8); ap.add_argument("--L", type=int, default=1024)
    ap.add_argument("--gen", type=int, default=64); ap.add_argument("--ctx_len", type=int, default=2048)
    ap.add_argument("--cache_dtype", default="bf16", choices=["bf16", "fp32"])
    a = ap.parse_args()
    lm = DualModEval(a.ckpt, batch_rows=a.n, device="cuda", tok_dir=a.tok_dir, ctx_len=a.ctx_len)
    m = lm.model.eval()
    x, src = sample_prompts(lm.tok, a.n, a.L)
    x = x.cuda()
    B, L = x.shape
    print(f"prompts: {src}, B={B} L={L}")
    # ---- eager teacher-forced logits at every position -------------------------------------
    t0 = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        caches = m.decode_init(); eag = []
        for p in range(L):
            eag.append(m.decode_step(x[:, p:p + 1], p, caches).float())
    eag = torch.cat(eag, 1); torch.cuda.synchronize(); t_eager = time.time() - t0
    # ---- graph teacher-forced -------------------------------------------------------------
    cdt = torch.float32 if a.cache_dtype == "fp32" else torch.bfloat16
    dec = GraphDecoder(m, B, ((L + a.gen + 255) // 256) * 256, dtype=cdt).capture()
    t0 = time.time(); grp = []; dec.reset()
    for p in range(L):
        dec.tok.copy_(x[:, p:p + 1]); dec.step(); grp.append(dec.logits.clone())
    grp = torch.cat(grp, 1); torch.cuda.synchronize(); t_graph = time.time() - t0
    d = (eag - grp).abs()
    agree = (eag.argmax(-1) == grp.argmax(-1)).float().mean().item()
    # top-1 margin where they disagree (near-ties are expected at bf16)
    dis = (eag.argmax(-1) != grp.argmax(-1))
    top2 = eag.topk(2, -1).values; margin = (top2[..., 0] - top2[..., 1])[dis]
    print(f"teacher-forced {B}x{L}: max|dlogit| {d.max().item():.4f}  mean|dlogit| {d.mean().item():.6f}  "
          f"argmax agree {agree*100:.3f}%  disagreements {int(dis.sum())} (eager top-1 margin at those: "
          f"max {margin.max().item() if margin.numel() else 0:.4f})")
    print(f"  eager {t_eager:.1f}s ({t_eager/L*1000:.1f} ms/step)  graph {t_graph:.1f}s ({t_graph/L*1000:.2f} ms/step)  "
          f"speedup {t_eager/max(t_graph,1e-6):.1f}x")
    # per-position profile: structural bug (diff from pos 0) vs drift (grows with pos)
    dmax_pos = d.amax(dim=(0, 2))            # [L]
    agree_pos = (eag.argmax(-1) == grp.argmax(-1)).float().mean(0)   # [L]
    print("  per-position max|dlogit| (first 8):", [f"{v:.4f}" for v in dmax_pos[:8].tolist()])
    for a0, a1 in [(0, 8), (8, 64), (64, 256), (256, 512), (512, L)]:
        print(f"  pos {a0:4d}-{a1:4d}: max|dlogit| {dmax_pos[a0:a1].max().item():.4f}  mean|dlogit| {d[:, a0:a1].mean().item():.5f}  "
              f"argmax agree {agree_pos[a0:a1].mean().item()*100:.2f}%")
    # ---- free-run greedy continuations ----------------------------------------------------
    Lp = L // 2
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        caches = m.decode_init(); lg = None
        for p in range(Lp):
            lg = m.decode_step(x[:, p:p + 1], p, caches)
        outs = []
        for k in range(a.gen):
            nxt = lg[:, -1].argmax(-1, keepdim=True); outs.append(nxt)
            lg = m.decode_step(nxt, Lp + k, caches)
        eg = torch.cat(outs, 1).cpu()
    gg, _ = dec.generate(x[:, :Lp], a.gen)
    same = (eg == gg).all(1)
    first_div = [(int((eg[i] != gg[i]).nonzero()[0]) if not same[i] else -1) for i in range(B)]
    print(f"free-run {a.gen} tokens from {Lp}-token prefix: identical continuations {int(same.sum())}/{B}; "
          f"first divergence step per row {first_div}")
    ok = agree >= 0.999 and d.max().item() < 0.1 and int(same.sum()) >= max(1, B - 1)
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
