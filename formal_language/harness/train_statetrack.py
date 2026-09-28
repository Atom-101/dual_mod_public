"""State-tracking ladder trainer (plans/state_tracking_ladder.md).

train_mqar.py skeleton + statetrack_gen on-the-fly batches. One (task, arch)
config per invocation; length-generalization eval at {1x, 2x, 4x} the
training op count from frozen disjoint-seed eval sets; khop reports
accuracy stratified by hop count k (difficulty sidecar).

Loss convention (pinned, matches house MQAR): mask indexes the loss —
cross-entropy at masked target positions predicted from position-1 logits;
attention/scan sees the full sequence.

Leakage instruments (khop): --sorted_pairs (control arm: pairs in chain
order — shuffled vs sorted must match or something leaks), --ablate_pairs
(chance audit: informative context replaced by noise; acc must be ~1/n_sym).

  python formal_language/harness/train_statetrack.py --task group --group s5 --arch dualmod
  python formal_language/harness/train_statetrack.py --task khop --arch vanilla
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from formal_language.harness.statetrack_gen import (PAD, make_matinv_batch, make_dfa_batch, BASE, cayley, make_group_batch,
                                  make_mixed_tagged_batch, make_niah_batch,
                                  make_group_final_batch,
                                  make_group_tagged_batch, make_khop_batch,
                                  make_ca_batch,
                                  make_keyed_group_batch,
                                  make_match3_batch, make_parity_batch,
                                  step_rng)

TASK_ID = {"group": 11, "khop": 12, "parity": 13, "match3": 14, "niah": 15,
           "keyed": 17,
           "mix": 16, "ca": 18, "matinv": 19, "dfa": 20, "cvp": 21}


def T_cvp_full(args, n_ops):
    from formal_language.harness.statetrack_gen import CVP_TOK_PER_GATE
    return 1 + 2 * args.cvp_inputs + CVP_TOK_PER_GATE * n_ops \
        + (args.cvp_pad if args.cvp_format == "silent" else 0) + 2 * min(args.cvp_queries, n_ops) \
        + (n_ops // args.cvp_leak_every if getattr(args, "cvp_leak_every", 0) else 0)


def make_batch(args, mul, n_elem, rng, B, n_ops, eval_fixed_k=None,
               D_override=None):
    """Returns (tok, tgt, msk, diff). tgt None => next-token convention
    (loss at msk positions predicting the token there from position-1);
    tgt set (tagged) => at-position convention CE(logits[pos], tgt[pos]).
    D_override: train-time depth for keyed (depth curriculum); eval uses args.D."""
    diff = None
    if args.task == "keyed":
        km = args.K_max or args.K
        K_t = getattr(args, "_K_t", None) or args.K
        tok, tgt, msk, depth = make_keyed_group_batch(
            rng, B, mul, n_elem, K_t, D_override or args.D, args.rho, K_max=km)
        if getattr(args, "key_shuffle", False) and not getattr(args, "_eval_now", False):
            perm = rng.permutation(km); iskey = (tok >= BASE) & (tok < BASE + km)
            tok = np.where(iskey, BASE + perm[np.clip(tok - BASE, 0, km - 1)], tok)
        return (torch.from_numpy(tok), torch.from_numpy(tgt),
                torch.from_numpy(msk), depth)
    if args.task == "mix":
        tok, tgt, msk, _ = make_mixed_tagged_batch(
            rng, B, n_ops, mul[0], n_elem[0], mul[1], n_elem[1],
            force=eval_fixed_k)          # eval_fixed_k reused as force str
        return (torch.from_numpy(tok), torch.from_numpy(tgt),
                torch.from_numpy(msk), None)
    if args.task == "group" and args.format == "tagged":
        tok, tgt, msk = make_group_tagged_batch(rng, B, n_ops, mul, n_elem,
                                                op_pool=getattr(args, "_op_pool", None))
        return (torch.from_numpy(tok), torch.from_numpy(tgt),
                torch.from_numpy(msk), None)
    if args.task == "group" and args.format == "final":
        if eval_fixed_k is not None:
            # fixed-k eval: exact tiling, every episode length k
            k = eval_fixed_k
            T = 1 + max(1, (args.n_ops * 2) // (k + 2)) * (k + 2)
            tok, msk, diff = make_group_final_batch(rng, B, T, mul, n_elem,
                                                    k_max=k, k_min=k)
        else:
            T = 1 + 2 * n_ops
            tok, msk, diff = make_group_final_batch(rng, B, T, mul, n_elem,
                                                    k_max=args.n_ops)
    elif args.task == "group":
        tok, msk = make_group_batch(rng, B, n_ops, mul, n_elem)
    elif args.task == "khop":
        tok, msk, diff = make_khop_batch(
            rng, B, n_ops, args.queries, args.k_max,
            sorted_pairs=args.sorted_pairs, ablate_pairs=args.ablate_pairs)
    elif args.task == "ca":
        # n_ops carries the CA depth t (scaled by gen_factors for depth-extrapolation).
        # ca_think -> pad=t THINK tokens (compute-starvation control); ca_gap for leaky.
        pad = n_ops if args.ca_think else 0
        tok, tgt, msk = make_ca_batch(rng, B, args.ca_w, n_ops, args.ca_rule,
                                      args.ca_format, pad=pad, gap=args.ca_gap)
        return (torch.from_numpy(tok), torch.from_numpy(tgt),
                torch.from_numpy(msk), None)
    elif args.task == "dfa":
        trans, n_st, n_sym, _ = args._dfa
        tok, tgt, msk = make_dfa_batch(rng, B, n_ops, trans, n_st, n_sym, sep=args.dfa_sep)
        return (torch.from_numpy(tok), torch.from_numpy(tgt),
                torch.from_numpy(msk), None)
    elif args.task == "matinv":
        # n_ops carries the matrix dimension n (scaled by gen_factors for size-extrapolation).
        # mi_think -> pad=n*n THINK tokens (compute-starvation control, cf. ca_think).
        pad = n_ops * n_ops if args.mi_think else 0
        tok, tgt, msk = make_matinv_batch(rng, B, n_ops, args.mi_p, args.mi_out, pad=pad)
        return (torch.from_numpy(tok), torch.from_numpy(tgt),
                torch.from_numpy(msk), None)
    elif args.task == "cvp":
        # n_ops carries the number of computed gates n (scaled by gen_factors for size-extrapolation).
        from formal_language.harness.statetrack_gen import make_cvp_batch
        # curriculum (train only): D_override carries the ramp fraction in (0,1]; m ramps 4->cvp_inputs, n 4->n_ops
        frac = D_override if (D_override is not None and not getattr(args, "_cvp_eval", False)) else 1.0
        m_t = max(4, int(round(4 + frac * (args.cvp_inputs - 4)))) if frac < 1.0 else args.cvp_inputs
        n_t = max(4, int(round(4 + frac * (n_ops - 4)))) if frac < 1.0 else n_ops
        tok, tgt, msk, depth = make_cvp_batch(rng, B, n_t, m=m_t, fmt=args.cvp_format,
                                              pad=args.cvp_pad, chain=args.cvp_chain,
                                              n_queries=args.cvp_queries, n_max=args.max_ops, m_max=args.cvp_inputs,
                                              n_topo=(16 if getattr(args, "_cvp_eval", False) else 1), window=args.cvp_window,
                                              leak_every=args.cvp_leak_every, addr=args.cvp_addr)
        if tok.shape[1] < T_cvp_full(args, n_ops):        # pad curriculum batches to the full length (trailing PAD, masked)
            padw = ((0, 0), (0, T_cvp_full(args, n_ops) - tok.shape[1]))
            tok = np.pad(tok, padw, constant_values=PAD); tgt = np.pad(tgt, padw); msk = np.pad(msk, padw); depth = np.pad(depth, padw)
        return (torch.from_numpy(tok), torch.from_numpy(tgt),
                torch.from_numpy(msk), depth)
    elif args.task == "parity":
        tok, msk = make_parity_batch(rng, B, n_ops)
    elif args.task == "niah":
        tok, msk, diff = make_niah_batch(
            rng, B, n_ops, args.v_len, args.queries,
            pairs_min=(2 if eval_fixed_k is None else None))
    else:
        tok, msk = make_match3_batch(rng, B, n_ops, M=args.m3_mod, W=args.m3_win)
    return torch.from_numpy(tok), None, torch.from_numpy(msk), diff


def vocab_of(args, n_elem):
    if args.task == "keyed":
        return BASE + (args.K_max or args.K) + 2 * n_elem
    if args.task == "group":
        return BASE + 2 * n_elem
    if args.task == "khop":
        return BASE + args.k_max + args.max_ops          # k toks + symbols at max eval size
    if args.task == "parity":
        return 12
    if args.task == "cvp":
        from formal_language.harness.statetrack_gen import cvp_vocab
        return cvp_vocab(args.max_ops, args.cvp_inputs)
    if args.task == "ca":
        # cells 8,9 · rule 10,11 · time tokens BASE+4 .. BASE+4+t_max
        return BASE + 4 + args.max_ops + 1
    if args.task == "dfa":
        return BASE + args._dfa[2] + args._dfa[1]
    if args.task == "matinv":
        from formal_language.harness.statetrack_gen import mi_vocab
        return mi_vocab(args.mi_p, args.max_ops)
    if args.task == "niah":
        return BASE + 512 + 64
    if args.task == "mix":
        return BASE + 2 * 10 + 2 * 60
    return BASE + args.m3_mod + 2


SCALES = {"5m": dict(d=256, L=4, h=4), "63m": dict(d=768, L=9, h=12),
          # width ladder at fixed L9 (head_dim=64), for the A5 formation-vs-width
          # sweep: recurrence gets composition depth from the scan, so width (SNR
          # at formation) is the lottery lever, not depth.
          "w1024": dict(d=1024, L=9, h=16),
          "w1280": dict(d=1280, L=9, h=20),
          "w1536": dict(d=1536, L=9, h=24),
          "w2048": dict(d=2048, L=9, h=32),
          # big LSTM scales to param-match / over-match DM (294M) on the keyed
          # capacity test — LSTM's ceiling scales with hidden width, so it must NOT
          # be param-hobbled there (unlike single-chain A5 where it saturates).
          "w3072": dict(d=3072, L=9, h=48),
          "w4096": dict(d=4096, L=9, h=64)}


from models.baselines.lstm_lm import LSTMLM   # nonlinear-recurrence calibration arm


def build_model(arch, vocab, max_seq_len, scale="5m", deq_sweeps=4,
                ctx_mod="linear", n_layers=0, nope=False, loop_layers=2, loop_k=64, key_mod=True, logn_ref=0, fbt_sched="", loop_emb_scale=False,
                loop_huginn=False, loop_bp=0):
    s = dict(SCALES[scale])
    if n_layers:                      # depth-sweep override (plans/depth_sweep.md)
        s["L"] = n_layers
    if arch == "lstm":
        return LSTMLM(vocab, s["d"], n_layers or max(2, s["L"] // 2)), "dualmod"
    if arch == "looped":
        from models.dualmod.config import DualModConfig
        from models.looped.looped_lm import LoopedLM
        cfg = DualModConfig(attn_mode="vanilla", d_model=s["d"], n_layers=1, n_heads=s["h"],
                            max_seq_len=max_seq_len, vocab_size=vocab, gate_bias_init=0.0,
                            checkpoint_chunk=0, use_rope=not nope)
        if loop_huginn:               # recurrent-depth recipe (models/recurrent_depth)
            from models.recurrent_depth.recurrent_depth_lm import RecurrentDepthLM
            return RecurrentDepthLM(cfg, n_unique=loop_layers, k_max=loop_k, bp_iters=loop_bp, emb_scale=loop_emb_scale), "looped"
        return LoopedLM(cfg, n_unique=loop_layers, k_max=loop_k), "looped"
    if arch == "fbt":                 # full-bandwidth transformer, exact sequential (models/fbt/fbt_lm.py)
        from models.dualmod.config import DualModConfig
        from models.fbt.fbt_lm import FBTLM
        cfg = DualModConfig(attn_mode="vanilla", d_model=s["d"], n_layers=s["L"], n_heads=s["h"],
                            max_seq_len=max_seq_len, vocab_size=vocab, gate_bias_init=0.0,
                            checkpoint_chunk=0, use_rope=not nope)
        return FBTLM(cfg), ("fbtp" if fbt_sched else "dualmod")
    if arch in ("dualmod", "vanilla"):
        from models.dualmod.config import DualModConfig
        from models.dualmod.model import DualModLM
        cfg = DualModConfig(
            attn_mode="vanilla" if arch == "vanilla" else "sequential",
            d_model=s["d"], n_layers=s["L"], n_heads=s["h"],
            max_seq_len=max_seq_len,
            vocab_size=vocab, gate_bias_init=0.0, checkpoint_chunk=0,
            deq_sweeps=deq_sweeps, ctx_mod=ctx_mod, use_rope=not nope, enable_key_mod=key_mod, attn_logn_ref=logn_ref)
        return DualModLM(cfg), "dualmod"
    from models.baselines.gdn import build_gdn    # Gated DeltaNet via flash-linear-attention
    return build_gdn(vocab, s["d"], s["L"], s["h"]), "fla"


_LOOP_K = [None]     # looped arch: K used by logits_of (train: sampled per step; eval: swept)


def logits_of(kind, model, x):
    if kind in ("looped", "fbtp"):
        return model(x, K=_LOOP_K[0])[0]
    if kind == "fla":
        return model(input_ids=x).logits
    return model(x)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASK_ID), required=True)
    ap.add_argument("--arch", choices=["dualmod", "vanilla", "gdn", "lstm", "looped", "fbt"],
                    required=True)
    ap.add_argument("--loop_layers", type=int, default=2, help="looped: unique blocks per iteration")
    ap.add_argument("--loop_k", type=int, default=64, help="looped: max iterations K (train upper bound)")
    ap.add_argument("--loop_emb_scale", action="store_true", help="looped --loop_huginn: scale the token embedding by sqrt(d) before the prelude (Geiping et al. Sec. 3.2 gamma). Off = the original runs")
    ap.add_argument("--fbt_sched", default="", help="fbt: train with the paper's parallel multi-pass approximation; weighted pass counts, e.g. '75:1,22:2,3:3' (Wang et al. Sec. 3.3). Empty = exact sequential. Eval sweeps --eval_loop_ks (0 = exact sequential)")
    ap.add_argument("--loop_k_min", type=int, default=0, help="looped: train with K ~ U[k_min, loop_k] (0 = fixed loop_k)")
    ap.add_argument("--eval_loop_ks", default="", help="looped: comma list of K to evaluate at (default: loop_k)")
    ap.add_argument("--loop_huginn", action="store_true", help="looped: Huginn recipe (random init state, no step emb, K ~ 1+Poisson(lognormal mean --loop_k_mean), truncated bp --loop_bp)")
    ap.add_argument("--loop_k_mean", type=float, default=32.0)
    ap.add_argument("--loop_bp", type=int, default=8)
    ap.add_argument("--loop_k_per_ops", type=int, default=0,
                    help="looped, Fan-style: tie K to the instance length — train K = ceil(n_ops_t / S), eval K = ceil(n_ops*f / S) "
                         "per length factor f (step embedding clamps at loop_k; set loop_k >= the largest eval K). 0 = off")
    ap.add_argument("--ctx_mod", default="linear",
                    choices=["linear", "vmlp", "fusion", "kmlp", "puremlp",
                             "puremlp_hw", "mult", "mult_res", "glu",
                             "linpoint"],
                    help="DM ctx-path transition arms (LSTM-gap)")
    ap.add_argument("--deq_sweeps", type=int, default=4,
                    help="DM per-step recursion depth (transition-"
                         "strength dial; LSTM-gap experiment)")
    ap.add_argument("--eval_dump", action="store_true",
                    help="save final ckpt + per-position eval artifact"
                         " (gold, gold_rank, top10, argmax) for the"
                         " composition analyses")
    ap.add_argument("--group", default="s5",
                    help="z10|s3|s4|s5|a5 (a5 = smallest non-solvable —"
                         " the headline; s3/s4 solvable, debugging only)")
    ap.add_argument("--format", choices=["interleave", "final", "tagged"],
                    default="interleave",
                    help="group only. interleave leaks prefix states (grid-1"
                         " confound); final = ops->final state (sparse, grid-2"
                         " null); tagged = ops-only input, per-position state"
                         " targets (Merrill et al) — dense AND leak-free: v3")
    ap.add_argument("--scale", choices=list(SCALES), default="5m")
    ap.add_argument("--gen_ball", action="store_true",
                    help="generator word-length curriculum: train ops"
                         " from a Cayley-ball of radius ramping 1->max"
                         " over the first half (A5 has no quotient"
                         " ladder, but DATA can grade by word length)")
    ap.add_argument("--tag_suffix", default="",
                    help="appended to the run tag (distinguish otherwise-"
                         "identical runs, e.g. _cold vs _curr)")
    ap.add_argument("--init_from", default=None,
                    help="warm-start: load state_dict (strict=False)"
                         " from a saved statetrack ckpt — the freeze-"
                         "then-unfreeze occupation test (init a nonlin"
                         " arm from the FORMED linear baseline)")
    ap.add_argument("--loss_guard", action="store_true",
                    help="skip optimizer step when loss > 4x trailing"
                         " median (toy storm guard: unguarded storm"
                         " destroyed a solved z10 at step 27k)")
    ap.add_argument("--probe_frac", type=float, default=0.0,
                    help="fraction of train steps at FULL n_ops during"
                         " the curriculum ramp (flagship-probe analogue;"
                         " tests whether minority deep pressure prevents"
                         " shallow entrenchment)")
    ap.add_argument("--curriculum_start", type=int, default=0,
                    help="matinv: ramp n from this value (default 2) -> n_ops; use with --init_from a solved smaller n")
    ap.add_argument("--curriculum_end", type=int, default=0,
                    help="step at which the n_ops ramp completes"
                         " (0 = steps//2, the original fraction rule —"
                         " HARMFUL at long budgets: 40k arms 3x below"
                         " 12k at matched full-N exposure)")
    ap.add_argument("--curriculum", action="store_true",
                    help="ramp train n_ops 16 -> n_ops over first half of"
                         " steps (optimization hygiene; eval always full N)")
    ap.add_argument("--n_ops", type=int, default=64,
                    help="ops per train sequence (khop: #symbols; group"
                         " final: max episode k)")
    ap.add_argument("--gen_factors", default="1,2,4",
                    help="eval length multipliers (for ca: multiplies the CA depth t)")
    # --- P-rung: cellular-automata prediction (plans/p_rung_spec.md) ---
    ap.add_argument("--ca_rule", type=int, default=110, choices=[90, 110],
                    help="110 = P-complete (nonlinear fold); 90 = linear GF(2) twin")
    ap.add_argument("--ca_w", type=int, default=16, help="CA row width (state-size axis)")
    ap.add_argument("--ca_format", default="silent", choices=["silent", "trace", "leaky"],
                    help="silent = P-complete (t steps internal); trace = depth-1 sanity; "
                         "leaky = fill-in curriculum (format B, gap-doubling)")
    ap.add_argument("--ca_think", action="store_true",
                    help="silent: insert t THINK tokens between query and answer (pad=t) — "
                         "the compute-starvation control; gives token-recurrences t genuine steps")
    ap.add_argument("--ca_gap", type=int, default=1,
                    help="leaky fill-in gap (rows at multiples of gap given; gap=t -> only 0,t)")
    ap.add_argument("--dfa", default="z2",
                    help="dfa task: language name (z2,z5,z10,a5,s3,flipflop,contains_a,contains_ab,count_a_ge3)")
    ap.add_argument("--dfa_sep", action="store_true",
                    help="dfa: predict states only at separator tokens between symbols (C-RASP paper protocol)")
    ap.add_argument("--len_min", type=int, default=0,
                    help="variable-length training: sample T ~ U[len_min, n_ops] per batch (C-RASP protocol)")
    ap.add_argument("--cvp_inputs", type=int, default=32, help="cvp: number of input bits m (entropy; >=32 keeps marginals ~0.5)")
    ap.add_argument("--cvp_format", default="tagged", choices=["tagged", "silent"],
                    help="cvp: tagged = value target at every gate (depth == position, streaming P); "
                         "silent = queries only after [THINK x cvp_pad]")
    ap.add_argument("--cvp_pad", type=int, default=0, help="cvp silent: THINK tokens between circuit and queries")
    ap.add_argument("--cvp_chain", type=float, default=1.0, help="cvp: P(carry operand = previous gate); 1 => depth == gate index")
    ap.add_argument("--cvp_queries", type=int, default=4, help="cvp: queried gates at the end (always includes the last gate)")
    ap.add_argument("--cvp_addr", default="abs", choices=["abs", "rel"], help="cvp operand addressing: absolute gate ids or relative offsets")
    ap.add_argument("--no_kmod", action="store_true", help="ablation: disable key modulation (enable_key_mod=False)")
    ap.add_argument("--attn_logn_ref", type=int, default=0, help="length-adaptive attention logit scale ln(n)/ln(ref) (Chiang & Cholak); 0 = off; applies to dualmod AND vanilla")
    ap.add_argument("--cvp_gate", default="xormaj", choices=["xormaj", "copy", "xor2"], help="cvp gate function (control ablation)")
    ap.add_argument("--cvp_leak_every", type=int, default=0,
                    help="cvp tagged: reveal the true value of every g-th gate as an INPUT token (g=1: every gate => pure "
                         "depth-1 lookup+gate, the pointer-format learnability control); 0 = silent values")
    ap.add_argument("--cvp_window", type=int, default=0, help="cvp: operands drawn from the last W gates only (0 = any earlier gate); locality makes retrieval positional, depth unchanged")
    ap.add_argument("--mi_p", type=int, default=2, help="matinv: field GF(p)")
    ap.add_argument("--mi_out", default="inv", choices=["inv", "solve", "mul"],
                    help="matinv: inv=A^-1 (NC2 rung), solve=x:Ax=b (NC2), mul=A.B (TC0 twin)")
    ap.add_argument("--mi_think", action="store_true",
                    help="matinv: n*n THINK tokens before the answer region")
    ap.add_argument("--queries", type=int, default=8)      # khop/niah
    ap.add_argument("--v_len", type=int, default=8)        # niah value len
    ap.add_argument("--k_max", type=int, default=8)        # khop
    ap.add_argument("--sorted_pairs", action="store_true")
    ap.add_argument("--ablate_pairs", action="store_true")
    ap.add_argument("--m3_mod", type=int, default=151)
    ap.add_argument("--m3_win", type=int, default=16)
    ap.add_argument("--K", type=int, default=8, help="keyed: #registers (capacity axis)")
    ap.add_argument("--D", type=int, default=8, help="keyed: ops/register (depth axis)")
    ap.add_argument("--rho", type=float, default=0.0, help="keyed: cross-register op fraction")
    ap.add_argument("--K_max", type=int, default=0, help="keyed: vocab key slots (0=use K)")
    ap.add_argument("--key_shuffle", action="store_true", help="keyed: random key-slot permutation per train batch")
    ap.add_argument("--K_curriculum", type=int, default=0, help="keyed: ramp live registers from this value -> --K over the curriculum window")
    ap.add_argument("--newkey_init", default="none", choices=["none", "mean"], help="keyed warm start: init untrained key rows from the mean of trained key rows")
    ap.add_argument("--nope", action="store_true",
                    help="no positional encoding (identity RoPE) for dualmod/vanilla — C-RASP length-gen protocol")
    ap.add_argument("--n_layers", type=int, default=0,
                    help="override the scale's layer count (depth sweep); 0 = scale default. "
                         "LSTM: exact LSTM depth (default is scale L//2).")
    ap.add_argument("--const_lr", action="store_true",
                    help="hold LR constant at --lr (no cosine decay) — for the "
                         "escape-vs-schedule ablations (more time in the escape "
                         "LR window since the data stream is infinite)")
    ap.add_argument("--early_stop", type=float, default=0.0,
                    help="stop (and run the FINAL eval/save) once in-dist accuracy >= this at a 1000-step eval (0 = off)")
    ap.add_argument("--snap_interval", type=int, default=0,
                    help="save weight snapshots every N steps (formation"
                         " forensics: watch the escape happen in weight space)")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--bsz", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    if args.task == "cvp" and args.cvp_gate != "xormaj":
        import formal_language.harness.statetrack_gen as _sg; _sg.CVP_GATE[0] = args.cvp_gate
    factors = [int(f) for f in args.gen_factors.split(",")]
    if args.task == "dfa":
        from formal_language.harness.statetrack_gen import dfa_lib
        args._dfa = dfa_lib(args.dfa)
        print(f"[dfa] {args.dfa}: states {args._dfa[1]} syms {args._dfa[2]} in_crasp={args._dfa[3]}", flush=True)
    if args.task == "keyed":
        factors = [1]              # (K,D) surface is an external grid of runs
    args.max_ops = args.n_ops * max(factors)

    mul = n_elem = None
    if args.task in ("group", "keyed"):
        mul, n_elem = cayley(args.group)
    elif args.task == "mix":
        ma, na = cayley("z10")
        mb, nb = cayley("a5")
        mul, n_elem = (ma, mb), (na, nb)
    vocab = vocab_of(args, n_elem if isinstance(n_elem, int) else 0)
    max_T = 1 + (4 if args.task == "khop" else 2) * args.max_ops \
        + (4 * args.queries if args.task == "khop" else 0)
    if args.task == "keyed":
        _nop = args.K * args.D
        _nx = int(round(args.rho * _nop))
        max_T = 1 + (_nop - _nx) * 2 + _nx * 3 + 2 * args.K
    if args.task == "niah":
        max_T = 1 + args.max_ops * (args.v_len + 3) \
            + args.queries * (args.v_len + 2)
    if args.task == "ca":
        # silent: seq = 3 + w (t-INDEPENDENT — depth is internal, the P-rung point);
        # trace: seq = 2 + t_max * w. Time token = BASE+4+t requires vocab >= 12+t_max.
        if args.ca_format == "silent":
            # 3 + row0(w) + THINK(pad=t if think) + ANS(w); pad at max eval t = max_ops
            max_T = 3 + 2 * args.ca_w + (args.max_ops if args.ca_think else 0)
        elif args.ca_format == "leaky":
            max_T = 2 + (args.max_ops + 1) * args.ca_w
        else:  # trace
            max_T = 2 + args.max_ops * args.ca_w
        # PER-CELL NON-DEGENERACY GATE (every trained+eval'd (rule,w,t) certifies itself):
        from formal_language.harness.statetrack_gen import ca_certify, ca_cell_ok
        ca_ts = sorted(set([args.n_ops] + [args.n_ops * f for f in factors]))
        bad = []
        print(f"[ca] certify rule{args.ca_rule} w{args.ca_w}: " + " ".join(
            f"t{t}=%.2f/%.2f" % ca_certify(np.random.default_rng(9_000 + t), args.ca_w, t, args.ca_rule)
            for t in ca_ts) + "  (marginal/short-cycle)")
        for t in ca_ts:
            m, s = ca_certify(np.random.default_rng(9_000 + t), args.ca_w, t, args.ca_rule)
            if not ca_cell_ok(m, s):
                bad.append(f"t={t}(marg {m:.2f}, cyc {s:.2f})")
        if bad:
            raise SystemExit(f"[ca] DEGENERATE cell(s) for rule{args.ca_rule} w{args.ca_w}: "
                             f"{', '.join(bad)}. Rule 90 period ~ w (use t<w/2, or larger w / "
                             f"smaller gen_factors); rule 110 is clean. See plans/p_rung_spec.md.")
    if args.task == "cvp":
        from formal_language.harness.statetrack_gen import cvp_certify, cvp_cell_ok, CVP_TOK_PER_GATE
        max_T = 1 + 2 * args.cvp_inputs + CVP_TOK_PER_GATE * args.max_ops \
            + (args.cvp_pad if args.cvp_format == "silent" else 0) + 2 * min(args.cvp_queries, args.n_ops) \
            + (args.max_ops // args.cvp_leak_every if args.cvp_leak_every else 0)
        ns = sorted(set([args.n_ops] + [args.n_ops * f for f in factors]))
        bad = []
        for n_ in ns:
            m_, s_ = cvp_certify(np.random.default_rng(9_000 + n_), n_, m=args.cvp_inputs, chain=args.cvp_chain, B=2048, n_batches=4, window=args.cvp_window)
            print(f"[cvp] certify m{args.cvp_inputs} n{n_} chain{args.cvp_chain:g}: marginal {m_:.3f} shortcut {s_:.3f}", flush=True)
            if not cvp_cell_ok(m_, s_):
                bad.append(f"n={n_}(marg {m_:.2f}, short {s_:.2f})")
        if bad and args.cvp_gate == "xormaj":     # the copy/xor2 gate ablations are single-operand BY DESIGN (control experiments)
            raise SystemExit(f"[cvp] DEGENERATE cell(s): {', '.join(bad)} (raise --cvp_inputs).")
        elif bad:
            print(f"[cvp] gate={args.cvp_gate}: shortcut certification waived (control ablation): {', '.join(bad)}", flush=True)
    if args.task == "dfa":
        max_T = (2 + 2 * args.max_ops) if args.dfa_sep else (1 + args.max_ops)
    if args.task == "matinv":
        from formal_language.harness.statetrack_gen import mi_certify
        nn_max = args.max_ops * args.max_ops
        max_T = 3 + nn_max + (nn_max if args.mi_out == "mul" else (args.max_ops if args.mi_out == "solve" else 0)) \
            + (nn_max if args.mi_think else 0) + (args.max_ops if args.mi_out == "solve" else nn_max)
        ns = sorted(set([args.n_ops] + [args.n_ops * f for f in factors]))
        print("[matinv] certify GF(%d) %s: " % (args.mi_p, args.mi_out) + " ".join(
            "n%d=%.2f/%.2f" % ((n,) + mi_certify(np.random.default_rng(9_000 + n), n, args.mi_p, args.mi_out))
            for n in ns) + "  (answer marginal max / trivial-guess agreement; chance %.2f)" % (1 / args.mi_p), flush=True)
    torch.manual_seed(args.seed)
    model, kind = build_model(args.arch, vocab, max(512, max_T + 8),
                              scale=args.scale, deq_sweeps=args.deq_sweeps,
                              ctx_mod=args.ctx_mod, n_layers=args.n_layers, nope=args.nope, loop_layers=args.loop_layers, loop_k=args.loop_k, key_mod=not args.no_kmod, logn_ref=args.attn_logn_ref, fbt_sched=args.fbt_sched, loop_emb_scale=args.loop_emb_scale,
                              loop_huginn=args.loop_huginn, loop_bp=args.loop_bp)
    _LOOP_K[0] = args.loop_k
    model = model.cuda()
    if args.init_from:
        ck = torch.load(args.init_from, map_location="cpu",
                        weights_only=False)
        own = model.state_dict()
        src = {k: v for k, v in ck["model"].items()
               if k in own and own[k].shape == v.shape}
        if args.task == "cvp":
            # chaining to a LARGER circuit: vocab layout = [ids 0..m_max+n_max-1 | BIT0 BIT1 VAL0 VAL1]; grow every
            # vocab-sized tensor (embedding / head / head bias, any arch) by copying the id prefix and moving the 4 tail rows.
            from formal_language.harness.statetrack_gen import cvp_vocab
            _ca = ck.get("args", {})
            sv = cvp_vocab(_ca.get("max_ops") or _ca["n_ops"], _ca["cvp_inputs"]) if "cvp_inputs" in _ca else \
                next(v.shape[0] for k, v in ck["model"].items() if "emb" in k and v.dim() == 2)   # source vocab size (arch-agnostic)
            for k, v in ck["model"].items():
                if k in own and k not in src and v.shape[0] == sv and own[k].shape[0] > sv and own[k].shape[1:] == v.shape[1:]:
                    new = own[k].clone(); new[:sv - 4] = v[:sv - 4]; new[-4:] = v[-4:]; src[k] = new
                    print(f"init_from: {k} transferred with layout growth {sv} -> {own[k].shape[0]}", flush=True)
        elif args.task == "keyed" and "K" in ck.get("args", {}) and ck["args"].get("task", "keyed") == "keyed":
            # KEYED chaining to a LARGER K_max (source must be a keyed run): layout is [BASE | KEY 0..K_max-1 | ELEM n | STATE n];
            # keep the BASE+old-K_max prefix rows in place and move the 2n group rows to the new tail (new key slots stay random,
            # or mean-initialised below). Ported from train_statetrack_ws.py.
            sv = next(v.shape[0] for k, v in ck["model"].items() if "emb" in k and v.dim() == 2)
            for k, v in ck["model"].items():
                if k in own and k not in src and v.shape[0] == sv and own[k].shape[0] > sv and own[k].shape[1:] == v.shape[1:]:
                    new = own[k].clone(); new[:sv - 2 * n_elem] = v[:sv - 2 * n_elem]; new[-2 * n_elem:] = v[-2 * n_elem:]; src[k] = new
                    print(f"init_from: {k} transferred with keyed layout growth {sv} -> {own[k].shape[0]}", flush=True)
        skipped = [k for k in ck["model"] if k not in src]
        missing, unexpected = model.load_state_dict(src, strict=False)
        if args.task == "keyed" and args.newkey_init == "mean":
            K_src = int(ck.get("args", {}).get("K", 0) or 0); km = args.K_max or args.K
            if 0 < K_src < km:
                with torch.no_grad():
                    for pn, W in model.named_parameters():
                        if W.dim() == 2 and W.shape[0] == vocab and ("emb" in pn or "head" in pn):
                            o = W[BASE:BASE + K_src]; W[BASE + K_src:BASE + km] = o.mean(0, keepdim=True) + 0.3 * o.std(0, keepdim=True) * torch.randn_like(W[BASE + K_src:BASE + km])
                            print(f"newkey_init=mean on {pn}: rows {K_src}..{km-1}", flush=True)
        print(f"init_from {args.init_from}: loaded={len(src)} "
              f"shape-skipped={len(skipped)} missing={len(missing)}",
              flush=True)
    n_par = sum(p.numel() for p in model.parameters())
    fmt_sfx = {"interleave": "", "final": "_final", "tagged": "_tag"}
    tag = f"{args.task}{'_' + args.group if args.task == 'group' else ''}" \
        + (f"_loop{args.loop_layers}x{args.loop_k}{'r'+str(args.loop_k_min) if args.loop_k_min else ''}{'p'+str(args.loop_k_per_ops) if args.loop_k_per_ops else ''}{'hug'+str(args.loop_bp) if args.loop_huginn else ''}{'gs' if args.loop_emb_scale else ''}" if args.arch == "looped" else "") \
        + (f"_fbtp{args.fbt_sched.replace(':', '').replace(',', '-')}" if args.arch == "fbt" and args.fbt_sched else "") \
        + (fmt_sfx[args.format] if args.task == "group" else "") \
        + ("" if args.scale == "5m" else f"_{args.scale}") \
        + ("" if args.steps == 12000 else f"_st{args.steps // 1000}k") \
        + ("" if args.seed == 1337 else f"_s{args.seed}") \
        + ((lambda b: f"_warm-{b[:12]}")(os.path.basename(
            args.init_from).replace("group_", "").replace(
            "_dualmod.pt", "")) if args.init_from else "") \
        + ("" if args.deq_sweeps == 4 else f"_dq{args.deq_sweeps}") \
        + ("" if args.ctx_mod == "linear" else f"_{args.ctx_mod}") \
        + ("" if args.lr == 1e-3 else f"_lr{args.lr:g}") \
        + ("_const" if args.const_lr else "") \
        + (f"_L{args.n_layers}" if args.n_layers else "") \
        + ("_nope" if args.nope else "") \
        + (f"_K{args.K}D{args.D}" + ("_kshuf" if args.key_shuffle else "") + (f"_Kcur{args.K_curriculum}" if args.K_curriculum else "") + ("_nkmean" if args.newkey_init == "mean" else "") if args.task == "keyed" else "") \
        + (f"_{args.dfa}" + ("_sep" if args.dfa_sep else "") + (f"_len{args.len_min}-{args.n_ops}" if args.len_min else "") if args.task == "dfa" else "") \
        + (f"_{args.mi_out}p{args.mi_p}n{args.n_ops}" + ("_think" if args.mi_think else "")
           + (("_cur" + (f"{args.curriculum_start}" if args.curriculum_start else "")) if args.curriculum else "") if args.task == "matinv" else "") \
        + (f"_{args.cvp_format}_m{args.cvp_inputs}n{args.n_ops}" + (f"_pad{args.cvp_pad}" if args.cvp_pad else "") + (f"_g{args.cvp_gate}" if args.cvp_gate != "xormaj" else "") + (f"_win{args.cvp_window}" if args.cvp_window else "") + (f"_leak{args.cvp_leak_every}" if args.cvp_leak_every else "") + ("_rel" if args.cvp_addr == "rel" else "") + ("_nokmod" if args.no_kmod else "")
           + (f"_chain{args.cvp_chain:g}" if args.cvp_chain != 1.0 else "") if args.task == "cvp" else "") \
        + (f"_logn{args.attn_logn_ref}" if args.attn_logn_ref else "") \
        + args.tag_suffix
    print(f"{tag} {args.arch}: {n_par/1e6:.2f}M params, vocab {vocab}, "
          f"train ops {args.n_ops}, eval x{factors}, "
          f"curriculum={args.curriculum}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.01)

    tid = TASK_ID[args.task]
    evals = {}                                # factor -> (x, tgt, m, diff)
    args._cvp_eval = True                     # cvp: eval sets span 16 circuits (train batches: 1)
    for f in factors:
        rng = step_rng(tid, args.seed, 2, f)      # split 2 = eval, frozen
        if args.task == "mix":
            for sub in ("a", "b"):
                x, tg, m, d = make_batch(args, mul, n_elem, rng, 256,
                                         args.n_ops * f, eval_fixed_k=sub)
                evals[f"{'z10' if sub == 'a' else 'a5'}_x{f}"] = (
                    x.cuda(), tg.cuda(), m.cuda(), d)
            continue
        fixed_k = args.n_ops * f if (args.task == "group"
                                     and args.format == "final") else None
        x, tg, m, d = make_batch(args, mul, n_elem, rng, 256, args.n_ops * f,
                                 eval_fixed_k=fixed_k)
        evals[f] = (x.cuda(), tg.cuda() if tg is not None else None,
                    m.cuda(), d)
    args._cvp_eval = False

    def acc_eval():
        args._eval_now = True; args._K_t = None
        try:
            return _acc_eval_inner()
        finally:
            args._eval_now = False

    def _acc_eval_inner():
        if kind == "looped" and args.loop_k_per_ops:
            merged = {}
            for f in list(evals):
                n_f = args.n_ops * f if isinstance(f, (int, float)) else args.n_ops
                _LOOP_K[0] = max(1, -(-int(n_f) // args.loop_k_per_ops))
                for f_, v_ in _acc_eval_once(only=f).items():
                    merged[f_] = v_
                merged[f"K_at_x{f}"] = _LOOP_K[0]
            _LOOP_K[0] = args.loop_k
            return merged
        if kind == "fbtp":
            ks = [int(k) for k in args.eval_loop_ks.split(",") if k.strip()] or [0, 1, 2, 3]
            merged = {}
            for K in ks:
                _LOOP_K[0] = K or None
                for f_, v_ in _acc_eval_once().items():
                    merged[f"P{K}_{f_}"] = v_
            _LOOP_K[0] = None
            merged[1] = merged.get("P0_1", merged[f"P{ks[-1]}_1"])       # headline = exact sequential recurrence
            return merged
        if kind == "looped":
            ks = [int(k) for k in args.eval_loop_ks.split(",") if k.strip()] or [args.loop_k]
            merged = {}
            for K in ks:
                _LOOP_K[0] = K
                for f_, v_ in _acc_eval_once().items():
                    merged[f"K{K}_{f_}"] = v_
            _LOOP_K[0] = args.loop_k
            merged[1] = merged.get(f"K{args.loop_k}_1", merged[f"K{ks[-1]}_1"])   # headline = train K
            return merged
        return _acc_eval_once()

    def _acc_eval_once(only=None):
        model.eval()
        out = {}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for f, (xe, tge, me, de) in evals.items():
                if only is not None and f != only:
                    continue
                hit = tot = 0
                kh = {}
                pos_hit = {}                       # position-octile strata
                T_e = xe.shape[1]
                for r0 in range(0, xe.shape[0], 32):
                    xb, mb = xe[r0:r0 + 32], me[r0:r0 + 32]
                    pred = logits_of(kind, model, xb).argmax(-1)
                    tp = mb.nonzero(as_tuple=True)
                    if tge is not None:            # tagged: at-position
                        ok = (pred[tp[0], tp[1]] == tge[r0:r0 + 32][tp])
                        for p_, o in zip(tp[1].tolist(), ok.tolist()):
                            b_ = min(7, p_ * 8 // T_e)
                            pos_hit.setdefault(b_, [0, 0])
                            pos_hit[b_][0] += o
                            pos_hit[b_][1] += 1
                    else:                          # next-token convention
                        ok = (pred[tp[0], tp[1] - 1] == xb[tp])
                    hit += ok.sum().item()
                    tot += mb.sum().item()
                    if de is not None:
                        if de.shape[1] == xe.shape[1]:         # [B,T] sidecar
                            ks = de[r0:r0 + 32][mb.cpu().numpy()]
                        else:                                  # [B,n_q]
                            ks = de[r0:r0 + 32].reshape(-1)
                        for k, o in zip(ks.tolist(), ok.tolist()):
                            kh.setdefault(k, [0, 0])
                            kh[k][0] += o
                            kh[k][1] += 1
                out[f] = round(hit / tot, 4)
                if kh:
                    out[f"x{f}_by_k"] = {k: round(a / b, 3)
                                         for k, (a, b) in sorted(kh.items())}
                if pos_hit:
                    out[f"x{f}_by_pos8"] = [round(a / b, 3) for _, (a, b)
                                            in sorted(pos_hit.items())]
        model.train()
        return out

    model.train()
    hist = []
    trail = []
    _snapdir = None
    if args.snap_interval:
        _snapdir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "out", "statetrack_ckpts", "snaps")
        os.makedirs(_snapdir, exist_ok=True)
    wl = None
    if args.gen_ball and args.task == "group":
        from formal_language.harness.statetrack_composition_analysis import word_lengths
        wl = word_lengths(args.group, mul, n_elem)
        wl_max = int(wl.max())
    for step in range(args.steps):
        lr = args.lr if args.const_lr else \
            args.lr * 0.5 * (1 + math.cos(math.pi * step / args.steps))
        for g in opt.param_groups:
            g["lr"] = lr
        rng = step_rng(tid, args.seed, 1, step)   # split 1 = train
        if wl is not None:
            frac_b = min(1.0, step / max(1, args.steps // 2))
            rad = max(1, int(round(frac_b * wl_max)))
            args._op_pool = np.where(wl <= rad)[0]
        n_ops_t = args.n_ops
        if args.len_min:
            n_ops_t = int(np.random.default_rng(10_000_000 + args.seed * 100_003 + step).integers(args.len_min, args.n_ops + 1))
        D_t = None
        if args.curriculum:
            end = args.curriculum_end or (args.steps // 2)
            frac = min(1.0, step / max(1, end))
            if args.task == "keyed":
                # depth curriculum: ramp ops/register 1 -> args.D (eval still full D)
                D_t = max(1, int(round(1 + frac * (args.D - 1))))
            elif args.task == "cvp":
                D_t = frac                    # make_batch ramps (m, n) 4 -> full from this fraction
                n_ops_t = args.n_ops
            elif args.task == "matinv":
                n0 = args.curriculum_start or 2
                n_ops_t = max(2, int(round(n0 + frac * (args.n_ops - n0))))
            else:
                n_ops_t = max(16, int(16 + frac * (args.n_ops - 16)))
                if args.probe_frac > 0 and \
                        (step * args.probe_frac) % 1 > (1 - args.probe_frac):
                    n_ops_t = args.n_ops      # probe step: full length
        args._K_t = None
        if args.task == "keyed" and args.K_curriculum:
            end_k = args.curriculum_end or (args.steps // 2)
            args._K_t = max(args.K_curriculum, min(args.K, int(round(args.K_curriculum + (args.K - args.K_curriculum) * min(1.0, step / max(1, end_k))))))
        x, tg, m, _ = make_batch(args, mul, n_elem, rng, args.bsz, n_ops_t,
                                 D_override=D_t)
        x, m = x.cuda(), m.cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if kind == "looped" and args.loop_k_per_ops:
                _LOOP_K[0] = max(1, -(-int(n_ops_t) // args.loop_k_per_ops))     # K tied to the instance length
            elif kind == "looped" and args.loop_huginn:
                tau = math.exp(torch.randn(1).item() * 0.5 + math.log(args.loop_k_mean) - 0.125)   # log-normal, mean = loop_k_mean
                _LOOP_K[0] = int(min(args.loop_k, 1 + torch.poisson(torch.tensor([tau])).item()))
            elif kind == "looped" and args.loop_k_min:
                _LOOP_K[0] = int(torch.randint(args.loop_k_min, args.loop_k + 1, (1,)))
            elif kind == "fbtp":
                _ws = [(float(a), int(b)) for a, b in (t.split(":") for t in args.fbt_sched.split(","))]
                _LOOP_K[0] = _ws[int(torch.multinomial(torch.tensor([w for w, _ in _ws]), 1))][1]
            logits = logits_of(kind, model, x)
        tp = m.nonzero(as_tuple=True)
        if tg is not None:                        # tagged: at-position CE
            loss = F.cross_entropy(logits[tp[0], tp[1]].float(),
                                   tg.cuda()[tp])
        else:
            loss = F.cross_entropy(logits[tp[0], tp[1] - 1].float(), x[tp])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if args.loss_guard:
            trail.append(loss.item())
            if len(trail) > 200:
                trail.pop(0)
            med = sorted(trail)[len(trail) // 2]
            if len(trail) >= 50 and loss.item() > 4 * max(med, 1e-3):
                opt.zero_grad(set_to_none=True)
                print(f"LOSS-GUARD skip step {step}: {loss.item():.3f} vs "
                      f"med {med:.3f}", flush=True)
                continue
        opt.step()
        opt.zero_grad(set_to_none=True)
        if _snapdir is not None and (step % args.snap_interval == 0
                                     or step == args.steps - 1):
            torch.save({"model": model.state_dict(), "step": step,
                        "loss": round(loss.item(), 4)},
                       os.path.join(_snapdir,
                                    f"{tag}_{args.arch}_snap{step:06d}.pt"))
        if step % 1000 == 0 or step == args.steps - 1:
            a = acc_eval()
            bn = [float(w.norm()) for k, w in model.state_dict().items()
                  if any(t in k for t in ("v_pre", "v_map", "v_gate",
                                          "k_pre", "v_comb"))]
            if bn:
                a["branch_wnorm"] = round(sum(bn) / len(bn), 2)
            hist.append((step, round(loss.item(), 4), a))
            print(f"step {step}: loss {loss.item():.4f} acc {a}", flush=True)
            if a.get(1, 0) > 0.998 and a.get(max(factors), 0) > 0.998:
                print("early stop: solved incl. max length", flush=True)
                break
            _main = a.get(1, None)
            if _main is None:      # looped: K-keyed metrics
                _vals = [v for k, v in a.items() if isinstance(k, str) and k.endswith("_1") and isinstance(v, float)]
                _main = max(_vals) if _vals else 0.0
            if args.early_stop and step >= 1000 and _main >= args.early_stop:
                print(f"EARLY-STOP at step {step}: in-dist acc {_main:.4f} >= {args.early_stop} (user 2026-09-23: end at 1.0, chain early)", flush=True)
                break

    final = acc_eval()
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    suffix = ("_sorted" if args.sorted_pairs else "") + \
             ("_ablate" if args.ablate_pairs else "")
    if args.eval_dump:
        ckdir = os.path.join(repo, "out", "statetrack_ckpts")
        os.makedirs(ckdir, exist_ok=True)
        torch.save({"model": model.state_dict(), "args": vars(args)},
                   os.path.join(ckdir, f"{tag}_{args.arch}.pt"))
        model.eval()
        dump = {}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for f, (xe, tge, me, de) in evals.items():
                gold, rank, top10, pos, row = [], [], [], [], []
                for r0 in range(0, xe.shape[0], 32):
                    xb, mb = xe[r0:r0 + 32], me[r0:r0 + 32]
                    lg = logits_of(kind, model, xb).float()
                    tp = mb.nonzero(as_tuple=True)
                    if tge is not None:
                        lgm = lg[tp[0], tp[1]]
                        g = tge[r0:r0 + 32][tp]
                    else:
                        lgm = lg[tp[0], tp[1] - 1]
                        g = xb[tp]
                    gl = lgm.gather(1, g[:, None]).squeeze(1)
                    rank.append((lgm > gl[:, None]).sum(1).cpu())
                    top10.append(lgm.topk(10, dim=1).indices.cpu())
                    gold.append(g.cpu())
                    pos.append(tp[1].cpu())
                    row.append((tp[0] + r0).cpu())
                os.makedirs(os.path.join(repo, "analysis"), exist_ok=True)
                np.savez(os.path.join(repo, "analysis",
                                      f"stdump_{tag}_{args.arch}_x{f}.npz"),
                         gold=torch.cat(gold).numpy(),
                         gold_rank=torch.cat(rank).numpy(),
                         top10=torch.cat(top10).numpy(),
                         pos=torch.cat(pos).numpy(),
                         row=torch.cat(row).numpy(), T=xe.shape[1])
        print("eval dump + ckpt saved", flush=True)
    out = os.path.join(repo, "analysis",
                       f"statetrack_{tag}_{args.arch}{suffix}.json")
    json.dump({"task": tag, "arch": args.arch, "params": n_par,
               "n_ops": args.n_ops, "factors": factors,
               "sorted_pairs": args.sorted_pairs,
               "ablate_pairs": args.ablate_pairs,
               "final": final, "hist": hist}, open(out, "w"), indent=1)
    print(f"FINAL {tag} {args.arch}{suffix}: {final} -> {out}", flush=True)


if __name__ == "__main__":
    main()
