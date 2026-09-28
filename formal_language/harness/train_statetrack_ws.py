"""Keyed-A5 trainer on the WaveScan fused engine (exact16 schedule).

WHY: the default statetrack path builds DualModLM with attn_mode="sequential"
— the naive O(B*H*T^2) eager reference scan. Keyed A5 at K16/K32 has ctx len
T ~ 289 / 577 (n_ops = K*D), and the eval at full T through that quadratic
path OOMs on 80GB. Language never hits this because it runs the *fused
WaveScanEngine* (chunked, memory-LINEAR in T). This script runs the SAME keyed
task through that engine with `uniform_chunks(T, 16, 16)` — the exact schedule
(K=c is exact by nilpotency; == the flagship's --schedule_mode exact16).

The engine is a next-token LM executor with its own captured backward, but
keyed uses AT-POSITION masked CE over a disjoint STATE output range. So we use
the engine purely as a memory-linear fwd/bwd executor:
  * keep_logits=True  -> engine.forward(x) fills engine.logits [B,T,V]
  * we compute the masked at-position CE loss and dL/dlogits ourselves
  * engine.backward(sid, d_logits=...) drives the memory-linear backward
  * engine.finish_grads(world_size=1) populates p.grad -> normal opt.step
Depth-curriculum: engine T is FIXED, so short-D batches are PAD-padded to the
full-D length T (trailing PAD is causally inert; loss/eval only on mask).
Warm-start (--init_from) loads a K8-solve state_dict (strict=False).
"""
import argparse
import json
import math
import os

import numpy as np
import torch

from formal_language.harness.statetrack_gen import (make_keyed_group_batch, cayley, step_rng, make_cvp_batch, cvp_vocab, cvp_certify, cvp_cell_ok, CVP_TOK_PER_GATE,
                                   BASE, PAD)
from engines.wavescan.engine.graphs import WaveScanEngine
from engines.wavescan.engine.schedule import uniform_chunks

SCALES = {"63m": dict(d=768, L=9, h=12), "w1024": dict(d=1024, L=9, h=16), "w1280": dict(d=1280, L=9, h=20),
          "w1536": dict(d=1536, L=9, h=24), "w2048": dict(d=2048, L=9, h=32)}
TID = 17  # TASK_ID["keyed"] — keep RNG streams identical to the eager harness


def build_dm(scale, vocab, max_seq_len, ctx_mod, n_layers=0, key_mod=True):
    """Same DualModLM config as formal_language/harness/train_statetrack.py build_model so a K8
    solver ckpt warm-loads. attn_mode is irrelevant on the engine path (the
    engine bypasses model.forward and runs its own scan over these weights)."""
    from models.dualmod.config import DualModConfig
    from models.dualmod.model import DualModLM
    s = SCALES[scale]
    cfg = DualModConfig(attn_mode="sequential", d_model=s["d"], n_layers=(n_layers or s["L"]),
                        n_heads=s["h"], max_seq_len=max_seq_len,
                        vocab_size=vocab, gate_bias_init=0.0,
                        checkpoint_chunk=0, ctx_mod=ctx_mod, enable_key_mod=key_mod)
    return DualModLM(cfg)


def keyed_batch(rng, B, mul, n, K, D, K_max, T_fix, shuffle=False):
    """One keyed-A5 batch, padded/truncated to fixed engine length T_fix.
    Returns cuda tensors (tok,tgt,msk) + cpu depth sidecar (np).
    shuffle: remap the key ids through a random permutation of the K_max slots (one perm per batch)."""
    tok, tgt, msk, depth = make_keyed_group_batch(rng, B, mul, n, K, D,
                                                   rho=0.0, K_max=K_max)
    if shuffle:
        perm = rng.permutation(K_max)
        iskey = (tok >= BASE) & (tok < BASE + K_max)
        tok = np.where(iskey, BASE + perm[np.clip(tok - BASE, 0, K_max - 1)], tok)
    T = tok.shape[1]
    if T < T_fix:                                    # pad trailing PAD (inert)
        pad = ((0, 0), (0, T_fix - T))
        tok = np.pad(tok, pad, constant_values=PAD)
        tgt = np.pad(tgt, pad, constant_values=0)
        msk = np.pad(msk, pad, constant_values=False)
        depth = np.pad(depth, pad, constant_values=0)
    elif T > T_fix:
        tok, tgt, msk, depth = (a[:, :T_fix] for a in (tok, tgt, msk, depth))
    return (torch.from_numpy(tok).long().cuda(),
            torch.from_numpy(tgt).long().cuda(),
            torch.from_numpy(msk).bool().cuda(), depth)


def cvp_batch(rng, B, args, n, T_fix, n_topo=1, m=None, leak=0):
    """One keyed-CVP batch (formal_language/harness/statetrack_gen.make_cvp_batch), padded to T_fix like keyed_batch.
    m < cvp_inputs (curriculum) keeps the token layout fixed via m_max=cvp_inputs."""
    tok, tgt, msk, depth = make_cvp_batch(rng, B, n, m=(m or args.cvp_inputs), fmt=args.cvp_format, pad=args.cvp_pad,
                                          chain=args.cvp_chain, n_queries=args.cvp_queries, n_max=args.n_max, n_topo=n_topo,
                                          window=args.cvp_window, m_max=args.cvp_inputs, leak_every=leak, addr=args.cvp_addr)
    T = tok.shape[1]
    if T < T_fix:
        pad = ((0, 0), (0, T_fix - T))
        tok = np.pad(tok, pad, constant_values=PAD); tgt = np.pad(tgt, pad, constant_values=0)
        msk = np.pad(msk, pad, constant_values=False); depth = np.pad(depth, pad, constant_values=0)
    elif T > T_fix:
        tok, tgt, msk, depth = (a[:, :T_fix] for a in (tok, tgt, msk, depth))
    return (torch.from_numpy(tok).long().cuda(), torch.from_numpy(tgt).long().cuda(),
            torch.from_numpy(msk).bool().cuda(), depth)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="keyed", choices=["keyed", "cvp"])
    ap.add_argument("--n_ops", type=int, default=64, help="cvp: computed gates n")
    ap.add_argument("--n_max", type=int, default=0, help="cvp: vocab layout size (default n_ops)")
    ap.add_argument("--cvp_inputs", type=int, default=32)
    ap.add_argument("--cvp_format", default="tagged", choices=["tagged", "silent"])
    ap.add_argument("--cvp_pad", type=int, default=0)
    ap.add_argument("--cvp_chain", type=float, default=1.0)
    ap.add_argument("--cvp_queries", type=int, default=4)
    ap.add_argument("--cvp_window", type=int, default=0)
    ap.add_argument("--cvp_addr", default="abs", choices=["abs", "rel"])
    ap.add_argument("--n_layers", type=int, default=0, help="override the scale's layer count (0 = scale default)")
    ap.add_argument("--no_kmod", action="store_true", help="ablation: disable key modulation (enable_key_mod=False)")
    ap.add_argument("--no_fused", action="store_true", help="use the verbatim (unfused) engine cell")
    ap.add_argument("--init_stack", action="store_true", help="with --init_from from a shallower model: extra blocks start as identity")
    ap.add_argument("--cvp_leak", type=int, default=0,
                    help="leaky curriculum: start revealing every gate value (g=1), double g each stage over the curriculum, "
                         "end fully silent; value = number of doubling stages (e.g. 4 => g=1,2,4,8 then silent). 0 = off")
    ap.add_argument("--scale", default="w1280")
    ap.add_argument("--ctx_mod", default="mult_res")
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--D", type=int, default=8)
    ap.add_argument("--K_max", type=int, default=64)
    ap.add_argument("--rho", type=float, default=0.0)
    ap.add_argument("--group", default="a5")
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--bsz", type=int, default=64)
    ap.add_argument("--eval_bsz", type=int, default=128)
    ap.add_argument("--curriculum", action="store_true")
    ap.add_argument("--curriculum_end", type=int, default=0)
    ap.add_argument("--loss_guard", action="store_true")
    ap.add_argument("--init_from", default=None)
    ap.add_argument("--key_shuffle", action="store_true", help="keyed: random permutation of the K_max key slots per batch (every slot trains)")
    ap.add_argument("--K_curriculum", type=int, default=0, help="keyed: ramp the number of registers from this value -> --K over the curriculum window (full D)")
    ap.add_argument("--freeze_body_steps", type=int, default=0, help="warm start: for the first N steps update ONLY tok_emb/lm_head (lets new key rows settle before the body moves)")
    ap.add_argument("--newkey_init", default="none", choices=["none", "mean"], help="keyed warm start: init key rows beyond the source ckpt's K from the mean of trained key rows (+noise)")
    ap.add_argument("--early_stop", type=float, default=0.0, help="stop + FINAL save once acc >= this at a 1000-step eval (0 = off)")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    global TID
    if args.task == "cvp":
        TID = 21                                  # TASK_ID["cvp"]
    mul_np, n = cayley(args.group)
    vocab = BASE + args.K_max + 2 * n
    # exact full-D length (rho=0 => deterministic: 1 BOS + 2 tok/op + 2 tok/query)
    T_raw = 1 + 2 * (args.K * args.D) + 2 * args.K
    if args.task == "cvp":
        args.n_max = args.n_max or args.n_ops
        vocab = cvp_vocab(args.n_max, args.cvp_inputs)
        T_raw = 1 + 2 * args.cvp_inputs + CVP_TOK_PER_GATE * args.n_ops \
            + (args.cvp_pad if args.cvp_format == "silent" else 0) + 2 * min(args.cvp_queries, args.n_ops) \
            + (args.n_ops if args.cvp_leak else 0)              # room for g=1 leaked values
        m_, s_ = cvp_certify(np.random.default_rng(9_000 + args.n_ops), args.n_ops, m=args.cvp_inputs,
                             chain=args.cvp_chain, B=2048, n_batches=4, window=args.cvp_window)
        print(f"[cvp] certify m{args.cvp_inputs} n{args.n_ops} chain{args.cvp_chain:g}: marginal {m_:.3f} shortcut {s_:.3f}", flush=True)
        if not cvp_cell_ok(m_, s_):
            raise SystemExit("[cvp] DEGENERATE cell (raise --cvp_inputs)")
    # round UP to a multiple of the chunk size c=16 so every uniform_chunks cell
    # is exactly 16 wide — the fused _attend_head_kernel tl.dot needs K>=16, so a
    # size-1 remainder chunk fails to compile. Extra tokens are trailing PAD.
    T_fix = ((T_raw + 15) // 16) * 16
    max_seq_len = max(512, T_fix + 8)

    model = build_dm(args.scale, vocab, max_seq_len, args.ctx_mod, args.n_layers, key_mod=not args.no_kmod).cuda()
    n_par = sum(p.numel() for p in model.parameters())
    warm = ""
    if args.init_from:
        ck = torch.load(args.init_from, map_location="cpu",
                        weights_only=False)
        src = ck.get("model", ck)
        own = model.state_dict()                 # cross-task warm start: skip shape-mismatched tensors (embedding / head)
        skipped = [k for k, v in src.items() if k in own and tuple(v.shape) != tuple(own[k].shape)]
        if args.task == "cvp":
            # CVP chaining to a LARGER n: the layout is [gate ids 0..m_max+n_max-1 | BIT0 BIT1 VAL0 VAL1]; copy the
            # gate-id prefix rows in place and move the 4 tail rows to the new tail (gate ids for new gates stay random).
            src = dict(src)
            for k in ("tok_emb.weight", "lm_head.weight"):
                if k in skipped and src[k].shape[1] == own[k].shape[1] and src[k].shape[0] < own[k].shape[0]:
                    new = own[k].clone(); ns = src[k].shape[0]
                    new[:ns - 4] = src[k][:ns - 4]; new[-4:] = src[k][-4:]
                    src[k] = new; skipped.remove(k)
                    print(f"init_from: {k} transferred with layout growth {ns} -> {own[k].shape[0]}", flush=True)
        elif args.task != "cvp" and ck.get("args", {}).get("task", "keyed") not in ("cvp",) and "K" in ck.get("args", {}):
            # KEYED chaining to a LARGER K_max (source must itself be a keyed run; cross-task sources keep embeddings shape-skipped): layout is [BASE | KEY 0..K_max-1 | ELEM n | STATE n]; keep the BASE+old-K_max
            # prefix rows in place and move the 2n group rows to the new tail (new key slots stay random).
            src = dict(src)
            for k in ("tok_emb.weight", "lm_head.weight"):
                if k in skipped and src[k].shape[1] == own[k].shape[1] and src[k].shape[0] < own[k].shape[0]:
                    new = own[k].clone(); ns = src[k].shape[0]
                    new[:ns - 2 * n] = src[k][:ns - 2 * n]; new[-2 * n:] = src[k][-2 * n:]
                    src[k] = new; skipped.remove(k)
                    print(f"init_from: {k} transferred with keyed layout growth {ns} -> {own[k].shape[0]}", flush=True)
        src = {k: v for k, v in src.items() if k not in skipped}
        loaded = model.load_state_dict(src, strict=False)
        print(f"init_from: shape-skipped={skipped}", flush=True)
        if args.newkey_init == "mean" and args.task != "cvp":
            K_src = int(ck.get("args", {}).get("K", 0) or 0)
            if 0 < K_src < args.K_max:
                with torch.no_grad():
                    for name in ("tok_emb.weight", "lm_head.weight"):
                        W = dict(model.named_parameters()).get(name)
                        if W is None: continue
                        old_rows = W[BASE:BASE + K_src]
                        mu, sd = old_rows.mean(0, keepdim=True), old_rows.std(0, keepdim=True)
                        W[BASE + K_src:BASE + args.K_max] = mu + 0.3 * sd * torch.randn_like(W[BASE + K_src:BASE + args.K_max])
                print(f"newkey_init=mean: key rows {K_src}..{args.K_max-1} <- mean(trained keys) + 0.3*sd noise", flush=True)
        if args.init_stack:
            # depth-stacking warm start: blocks absent from the ckpt start as IDENTITY (zero their output
            # projections wo / mlp.w2) so the deeper model computes exactly the shallow solver's function at init.
            have = {int(k.split(".")[1]) for k in src if k.startswith("blocks.")}
            with torch.no_grad():
                for i, blk in enumerate(model.blocks):
                    if i not in have:
                        blk.attn.wo.weight.zero_(); blk.mlp.w2.weight.zero_()
            print(f"init_stack: ckpt blocks {sorted(have)} -> blocks {[i for i in range(len(model.blocks)) if i not in have]} start as identity", flush=True)
        warm = f"_warm-{os.path.basename(args.init_from)[:12]}"
        print(f"init_from {args.init_from}: missing={len(loaded.missing_keys)} "
              f"unexpected={len(loaded.unexpected_keys)}", flush=True)

    tag = args.tag or (f"keyed_ws_{args.scale}_{args.ctx_mod}"
                       f"{'' if args.seed == 1337 else f'_s{args.seed}'}{warm}"
                       f"_K{args.K}D{args.D}")
    if args.task == "cvp":
        tag = args.tag or (f"cvp_ws_{args.scale}_{args.ctx_mod}{'_L%d' % args.n_layers if args.n_layers else ''}_{args.cvp_format}_m{args.cvp_inputs}n{args.n_ops}"
                           + (f"_nmax{args.n_max}" if args.n_max != args.n_ops else "")
                           + (f"_pad{args.cvp_pad}" if args.cvp_pad else "") + (f"_win{args.cvp_window}" if args.cvp_window else "") + (f"_chain{args.cvp_chain:g}" if args.cvp_chain != 1.0 else "")
                           + ("_cur" if args.curriculum else "") + ("_kshuf" if args.key_shuffle else "") + (f"_Kcur{args.K_curriculum}" if args.K_curriculum else "") + ("_nkmean" if args.newkey_init == "mean" else "") + (f"_fb{args.freeze_body_steps}" if args.freeze_body_steps else "") + (f"_leak{args.cvp_leak}" if args.cvp_leak else "") + ("_rel" if args.cvp_addr == "rel" else "") + ("_nokmod" if args.no_kmod else "") + (f"_st{args.steps // 1000}k" if args.steps != 12000 else "")
                           + ("" if args.seed == 1337 else f"_s{args.seed}") + warm + ("_stack" if args.init_stack else ""))

    B, Tf = args.bsz, T_fix
    schedules = [uniform_chunks(Tf, 16, 16)]         # exact16
    engine = WaveScanEngine(model, schedules, B, Tf, autocast_bf16=True,
                            use_graphs=False, fused_sweep=not (args.no_fused or args.no_kmod), keep_logits=True)   # nokmod: fused cell unsupported; verbatim cell parity-checked (evals/engine_parity_nokmod.py)
    print(f"{tag} dualmod: {n_par/1e6:.2f}M params, vocab {vocab}, "
          f"T_fix {Tf}, engine fused_sweep={engine.fused_sweep}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.01)

    # ---- frozen eval set at FULL D (never curriculum-shortened) ----
    e_rng = step_rng(TID, args.seed, 2, 0)
    n_ev = (args.eval_bsz // B) * B or B
    xe, tge, me, de = (cvp_batch(e_rng, n_ev, args, args.n_ops, Tf, n_topo=16) if args.task == "cvp" else
                       keyed_batch(e_rng, n_ev, mul_np, n, args.K, args.D, args.K_max, Tf))

    def acc_eval():
        hit = tot = 0
        kh, ph = {}, {}
        for r0 in range(0, n_ev, B):
            xb = xe[r0:r0 + B]
            mb = me[r0:r0 + B]
            with torch.no_grad():
                lg = engine.forward(xb, 0)           # [B,T,V] (keep_logits)
            pred = lg.argmax(-1)
            tp = mb.nonzero(as_tuple=True)
            ok = (pred[tp[0], tp[1]] == tge[r0:r0 + B][tp])
            hit += ok.sum().item()
            tot += mb.sum().item()
            ks = de[r0:r0 + B][mb.cpu().numpy()]
            for k, o in zip(ks.tolist(), ok.tolist()):
                kh.setdefault(k, [0, 0]); kh[k][0] += o; kh[k][1] += 1
            for p_, o in zip(tp[1].tolist(), ok.tolist()):
                b_ = min(7, p_ * 8 // Tf)
                ph.setdefault(b_, [0, 0]); ph[b_][0] += o; ph[b_][1] += 1
        out = {1: round(hit / max(tot, 1), 4)}
        out["x1_by_k"] = {k: round(a / b, 3) for k, (a, b) in sorted(kh.items())}
        out["x1_by_pos8"] = [round(a / b, 3) for _, (a, b) in sorted(ph.items())]
        return out

    hist, trail = [], []
    for step in range(args.steps):
        lr = args.lr * 0.5 * (1 + math.cos(math.pi * step / args.steps))
        for g in opt.param_groups:
            g["lr"] = lr
        D_t = args.D
        frac = 1.0
        if args.curriculum:
            end = args.curriculum_end or (args.steps // 2)
            frac = min(1.0, step / max(1, end))
            D_t = max(1, int(round(1 + frac * (args.D - 1))))
        rng = step_rng(TID, args.seed, 1, step)
        if args.task == "cvp":
            n_t, m_t = args.n_ops, args.cvp_inputs
            if args.curriculum:
                n_t = max(4, int(round(4 + frac * (args.n_ops - 4))))
                m_t = max(4, int(round(4 + frac * (args.cvp_inputs - 4))))
            leak = 0
            if args.cvp_leak:                                  # leak doubling: stage s in [0, cvp_leak) -> g = 2^s; last stage silent
                end = args.curriculum_end or (args.steps // 2)
                stage = min(args.cvp_leak, int(step * (args.cvp_leak + 1) / max(1, end)))
                leak = 0 if stage >= args.cvp_leak else 2 ** stage
            xb, tgt, mb, _ = cvp_batch(rng, B, args, n_t, Tf, m=m_t, leak=leak)
        else:
            K_t = args.K
            if args.K_curriculum:
                end_k = args.curriculum_end or (args.steps // 2)
                K_t = max(args.K_curriculum, min(args.K, int(round(args.K_curriculum + (args.K - args.K_curriculum) * min(1.0, step / max(1, end_k))))))
            xb, tgt, mb, _ = keyed_batch(rng, B, mul_np, n, K_t, D_t,
                                         args.K_max, Tf, shuffle=args.key_shuffle)

        logits = engine.forward(xb, 0)               # [B,T,V] bf16, no autograd
        tp = mb.nonzero(as_tuple=True)
        lg = logits[tp[0], tp[1]].float()            # [N, V] masked rows
        gold = tgt[tp]                               # [N]
        Nm = max(gold.numel(), 1)
        loss = torch.nn.functional.cross_entropy(lg, gold)
        # dL/dlogits at masked positions = (softmax - onehot)/N ; 0 elsewhere
        d_logits = torch.zeros_like(logits, dtype=torch.float32)
        soft = torch.softmax(lg, dim=-1)
        soft[torch.arange(Nm, device=lg.device), gold] -= 1.0
        d_logits[tp[0], tp[1]] = soft / Nm
        engine.backward(0, d_logits=d_logits.to(engine.d_logits.dtype))
        engine.finish_grads(world_size=1)

        lv = loss.item()
        skip = False
        if args.loss_guard:
            trail.append(lv)
            if len(trail) > 200:
                trail.pop(0)
            med = sorted(trail)[len(trail) // 2]
            if len(trail) >= 50 and lv > 4 * max(med, 1e-3):
                skip = True
                print(f"LOSS-GUARD skip step {step}: {lv:.3f} vs med {med:.3f}",
                      flush=True)
        if not skip and args.freeze_body_steps and step < args.freeze_body_steps:
            for pn, pp in model.named_parameters():
                if pp.grad is not None and not (pn.startswith("tok_emb") or pn.startswith("lm_head")):
                    pp.grad.zero_()
        if not skip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        opt.zero_grad(set_to_none=True)

        if step % 1000 == 0 or step == args.steps - 1:
            a = acc_eval()
            hist.append((step, round(lv, 4), a))
            print(f"step {step}: loss {lv:.4f} acc {a}", flush=True)
            if args.early_stop and step >= 1000 and a.get(1, 0) >= args.early_stop:
                print(f"EARLY-STOP at step {step}: acc {a.get(1):.4f} >= {args.early_stop}", flush=True)
                break
            if a[1] > 0.998:
                print("early stop: solved", flush=True)
                break

    final = acc_eval()
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.makedirs(os.path.join(repo, "analysis"), exist_ok=True)
    os.makedirs(os.path.join(repo, "out", "statetrack_ckpts"), exist_ok=True)
    json.dump({"task": tag, "arch": "dualmod", "params": n_par,
               "final": final, "hist": hist, "args": vars(args)},
              open(os.path.join(repo, "analysis",
                                f"statetrack_{tag}_dualmod.json"), "w"))
    torch.save({"model": model.state_dict(), "args": vars(args)},
               os.path.join(repo, "out", "statetrack_ckpts",
                            f"{tag}_dualmod.pt"))
    print(f"FINAL {tag} dualmod: {final}", flush=True)


if __name__ == "__main__":
    main()
