"""Validate the RAW-DIAGONAL (+ sliding-window) wave3d_x operator against the module's
exact sequential scan (models/models/dualmod/scan.py: raw self in every step => nilpotent => the
engine at K=C=64 must reproduce it EXACTLY up to bf16 noise).

  R1  raw_diag=True, full causal, K=64 vs model(exact scan): loss / logits / grads.
  R2  raw_diag=True, local_window=W,  K=64 vs model(exact scan, cfg.local_window=W).
  R3  info: legacy (refined diagonal) K=64 gap vs exact -- the bug this fixes.
  R4  info: K in {8,16,24} CE vs K=64 (raw_diag): the approximation ladder.

Usage (WORKER node):
  PYTHONPATH=. python lm/train_1p3b/validate_rawdiag.py --T 1024 --B 2 --L 2 --W 256
"""
import argparse, time
from collections import defaultdict
import torch, torch.nn.functional as F
from models.dualmod.model import DualModLM
from engines.wave3d.config_1p3b import build_config
from engines.wave3d.wave3d import make_engine

PASS, FAIL = "PASS", "FAIL"

FINE = False
def group_name(n):
    p = n.split(".")
    if p[0] != "blocks": return p[0]
    return ".".join(p[:1] + p[2:-1]) if (FINE and p[2] == "attn") else ".".join(p[:1] + p[2:3])

def grad_snapshot(model):
    g = {n: (None if p.grad is None else p.grad.detach().float().clone()) for n, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    return g

def grad_table(g_ref, g_new, tol, title):
    groups = defaultdict(lambda: [0.0, 0.0]); missing = []
    for n, gr in g_ref.items():
        gn = g_new.get(n)
        if gr is None and gn is None: continue
        if gr is None or gn is None: missing.append(n); continue
        groups[group_name(n)][0] += (gr - gn).pow(2).sum().item(); groups[group_name(n)][1] += gr.pow(2).sum().item()
    ok, worst = not missing, 0.0
    for g in sorted(groups):
        r = (groups[g][0] ** 0.5) / max(groups[g][1] ** 0.5, 1e-30); worst = max(worst, r); ok &= r < tol
        print(f"    {g:34} rel={r:.3e}{'' if r < tol else '   <-- FAIL'}")
    if missing: print(f"    MISSING grads: {missing[:5]}")
    print(f"  [{PASS if ok else FAIL}] {title}: worst rel={worst:.3e} (tol {tol:.0e})")
    return ok

ENG_KW = {}
ATTN = "cudnn_fe"; FP32 = False
def make(model, B, T, **kw):
    return make_engine(model, B, T, engine="x", attn_impl=ATTN, stash_attn=True, graphs=False, **ENG_KW, **kw)
def _ac():
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=not FP32)

def run(model, idx, tgt, eng, K):
    with _ac():
        lg = eng.forward_logits(idx, K)
        l = F.cross_entropy(lg.float().view(-1, lg.shape[-1]), tgt.reshape(-1))
    l.backward(); g = grad_snapshot(model)
    return l.item(), lg.detach().float(), g

def ref(model, idx, tgt):
    with _ac():
        lg, l = model(idx, tgt)
    l.backward(); g = grad_snapshot(model)
    return l.item(), lg.detach().float(), g

def _rels(g_ref, g_new):
    groups = defaultdict(lambda: [0.0, 0.0])
    for n, gr in g_ref.items():
        gn = g_new.get(n)
        if gr is None or gn is None: continue
        groups[group_name(n)][0] += (gr - gn).pow(2).sum().item(); groups[group_name(n)][1] += gr.pow(2).sum().item()
    return {g: (v[0] ** 0.5) / max(v[1] ** 0.5, 1e-30) for g, v in groups.items()}

def compare(tag, r, w, tol_loss=3e-3, tol_lg=0.25, tol_g=5e-2, floor=None):
    lr, lgr, gr = r; lw, lgw, gw = w
    d = abs(lr - lw); mad = (lgw - lgr).abs().max().item(); rel = ((lgw - lgr).norm() / lgr.norm()).item()
    ok = d < tol_loss and mad < tol_lg
    print(f"  [{tag}] loss ref={lr:.6f} eng={lw:.6f} |d|={d:.3e} (tol {tol_loss:.0e}); logits max|d|={mad:.3e} rel={rel:.3e} (tol {tol_lg})")
    if floor is not None:
        # bf16 noise floor: ref(bf16) vs ref(fp32) per group; engine passes if within tol OR < 2x floor
        fl = _rels(floor[2], gr); re = _rels(floor[2], gw); okg = True
        for g in sorted(re):
            p_ = re[g] < tol_g or re[g] < 2 * fl.get(g, 0.0); okg &= p_
            print(f"    {g:34} eng(bf16) vs ref(fp32) rel={re[g]:.3e}   floor ref(bf16) vs ref(fp32)={fl.get(g, float('nan')):.3e}{'' if p_ else '   <-- FAIL'}")
        print(f"  [{PASS if okg else FAIL}] {tag} grads (tol {tol_g:.0e} or < 2x bf16 noise floor)")
        ok &= okg
    else:
        ok &= grad_table(gr, gw, tol_g, f"{tag} grads")
    print(f"[{PASS if ok else FAIL}] {tag}")
    return ok

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=1024); ap.add_argument("--B", type=int, default=2)
    ap.add_argument("--L", type=int, default=2); ap.add_argument("--W", type=int, default=256)
    ap.add_argument("--gate_bias", type=float, default=0.0); ap.add_argument("--fine", action="store_true")
    ap.add_argument("--only", default=""); ap.add_argument("--eng_kw", default="{}")
    ap.add_argument("--floor", action="store_true", help="bf16 mode: also compute the fp32 reference as the bf16 noise floor")
    ap.add_argument("--fp32", action="store_true", help="fp32 model + dense mathk kernel (no autocast): exactness check without bf16 noise")
    a = ap.parse_args(); dev = "cuda"
    global FINE, ATTN, FP32; FINE = a.fine
    if a.fp32:
        ATTN, FP32 = "mathk", True; torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    import json; ENG_KW.update(json.loads(a.eng_kw)); print("engine kwargs:", ENG_KW)
    torch.manual_seed(0)
    cfg = build_config(seq_len=a.T); cfg.n_layers = a.L; cfg.gate_bias_init = a.gate_bias
    model = DualModLM(cfg).to(dev)
    torch.manual_seed(1)
    idx = torch.randint(0, cfg.vocab_size, (a.B, a.T), device=dev); tgt = torch.randint(0, cfg.vocab_size, (a.B, a.T), device=dev)
    res = {}
    # ---- R1 full causal, raw diagonal
    print("\n=========== R1 raw_diag full-causal K=64 vs exact scan ===========")
    cfg.local_window = 0
    r_full = ref(model, idx, tgt)
    fl_full = fl_win = None
    if not FP32 and a.floor:
        FP32 = True; fl_full = ref(model, idx, tgt); FP32 = False
    eng = make(model, a.B, a.T, raw_diag=True); w = run(model, idx, tgt, eng, 64)
    res["R1"] = compare("R1 raw_diag K64", r_full, w, floor=fl_full)
    if a.only == "R1":
        print(f"[{PASS if res['R1'] else FAIL}] validate_rawdiag (R1 only)"); return
    # ---- R4 ladder (info)
    print("\n=========== R4 raw_diag ladder CE vs exact (info) ===========")
    for K in (8, 16, 24, 32):
        lk = run(model, idx, tgt, eng, K)[0]; print(f"  K={K:2d} CE={lk:.6f}  gap vs exact={lk - r_full[0]:+.5f}")
    del eng; torch.cuda.empty_cache()
    # ---- R3 legacy refined diagonal (info)
    print("\n=========== R3 legacy (refined diagonal) K=64 vs exact (info: the bug) ===========")
    eng = make(model, a.B, a.T, raw_diag=False); w3 = run(model, idx, tgt, eng, 64)
    print(f"  legacy K64 CE={w3[0]:.6f} exact={r_full[0]:.6f} |d|={abs(w3[0]-r_full[0]):.3e} logits max|d|={(w3[1]-r_full[1]).abs().max().item():.3e}")
    del eng; torch.cuda.empty_cache()
    # ---- R2 windowed
    print(f"\n=========== R2 raw_diag + local_window={a.W} K=64 vs exact windowed scan ===========")
    cfg.local_window = a.W
    r_win = ref(model, idx, tgt)
    if not FP32 and a.floor:
        FP32 = True; fl_win = ref(model, idx, tgt); FP32 = False
    print(f"  (info) exact windowed CE={r_win[0]:.6f} vs exact full CE={r_full[0]:.6f} (must differ)")
    eng = make(model, a.B, a.T, raw_diag=True, local_window=a.W); w2 = run(model, idx, tgt, eng, 64)
    res["R2"] = compare(f"R2 raw_diag W={a.W} K64", r_win, w2, floor=fl_win)
    for K in (8, 16, 24):
        lk = run(model, idx, tgt, eng, K)[0]; print(f"  K={K:2d} windowed CE={lk:.6f} gap vs exact={lk - r_win[0]:+.5f}")
    print("\n==== SUMMARY ====")
    for k, v in res.items(): print(f"  {k}: {PASS if v else FAIL}")
    print(f"[{PASS if all(res.values()) else FAIL}] validate_rawdiag")

if __name__ == "__main__":
    main()
