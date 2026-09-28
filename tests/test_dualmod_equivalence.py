"""Correctness tests T1-T5 from plan.md §9 (+ chunked-checkpoint grad match).

All tests run in fp32 (tolerance 1e-5 per the spec); parametrized over CPU and
CUDA when available. Mode switches flip cfg.attn_mode on a single model — the
config object is shared by all blocks, and vanilla/sequential/deq are just
different forward paths over the same parameters.
"""

import pytest
import torch

from models.dualmod.config import DualModConfig
from models.dualmod.model import DualModLM

torch.set_float32_matmul_precision("highest")

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
TOL = 1e-5


def small_cfg(**kw):
    base = dict(d_model=64, n_layers=2, n_heads=4, vocab_size=97,
                max_seq_len=32, attn_mode="sequential", checkpoint_chunk=0,
                dtype="fp32")
    base.update(kw)
    return DualModConfig(**base)


def make_model(cfg, device, seed=1234):
    torch.manual_seed(seed)
    return DualModLM(cfg).to(device).eval()


def perturb(model, seed=7, gate_bias=0.0, ctx_std=0.05):
    """Move weights away from the near-vanilla init: gates ~0.5, ctx branch live."""
    torch.manual_seed(seed)
    with torch.no_grad():
        for blk in model.blocks:
            a = blk.attn
            if hasattr(a, "w_kctx"):
                a.w_gk.bias.fill_(gate_bias)
                a.w_gk.weight.normal_(0, 0.05)
                a.w_kctx.weight.normal_(0, ctx_std)
            if hasattr(a, "w_vctx"):
                a.w_gv.bias.fill_(gate_bias)
                a.w_gv.weight.normal_(0, 0.05)
                a.w_vctx.weight.normal_(0, ctx_std)


def rand_idx(cfg, B, T, device, seed=99):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, cfg.vocab_size, (B, T), generator=g).to(device)


# ---- T1: vanilla equivalence of the scan -----------------------------------

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("B,T,chunk", [(2, 16, 0), (3, 32, 0), (2, 31, 5), (1, 8, 3)])
def test_t1_scan_matches_vanilla(device, B, T, chunk):
    cfg = small_cfg(enable_key_mod=False, enable_value_mod=False,
                    checkpoint_chunk=chunk)
    model = make_model(cfg, device)
    idx = rand_idx(cfg, B, T, device)
    with torch.no_grad():
        cfg.attn_mode = "sequential"
        logits_seq, _ = model(idx)
        cfg.attn_mode = "vanilla"
        logits_van, _ = model(idx)
    assert (logits_seq - logits_van).abs().max().item() < TOL


# ---- T2: gate-closed equivalence --------------------------------------------

@pytest.mark.parametrize("device", DEVICES)
def test_t2_gates_closed_matches_vanilla(device):
    cfg = small_cfg(enable_key_mod=True, enable_value_mod=True)
    model = make_model(cfg, device)
    with torch.no_grad():
        for blk in model.blocks:
            blk.attn.w_gk.bias.fill_(20.0)
            blk.attn.w_gv.bias.fill_(20.0)
            blk.attn.w_gk.weight.zero_()
            blk.attn.w_gv.weight.zero_()
            # make the ctx branch non-trivially large: sigma(-20) must kill it
            blk.attn.w_kctx.weight.normal_(0, 0.5)
            blk.attn.w_vctx.weight.normal_(0, 0.5)
    idx = rand_idx(cfg, 2, 24, device)
    with torch.no_grad():
        cfg.attn_mode = "sequential"
        logits_seq, _ = model(idx)
        cfg.attn_mode = "vanilla"
        logits_van, _ = model(idx)
    assert (logits_seq - logits_van).abs().max().item() < TOL


# ---- T3: decode ≡ teacher-forced train --------------------------------------

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("key_mod,value_mod", [(True, True), (False, True)])
def test_t3_decode_matches_teacher_forced(device, key_mod, value_mod):
    cfg = small_cfg(enable_key_mod=key_mod, enable_value_mod=value_mod)
    model = make_model(cfg, device)
    perturb(model)  # gates ~0.5, ctx branch live: genuinely non-vanilla
    B, T = 2, 24
    idx = rand_idx(cfg, B, T, device)
    with torch.no_grad():
        logits_tf, _ = model(idx)
        caches = model.decode_init()
        steps = [model.decode_step(idx[:, t:t + 1], t, caches) for t in range(T)]
        logits_dec = torch.cat(steps, dim=1)
    diff = (logits_tf - logits_dec).abs().max().item()
    assert diff < TOL, f"train/decode gap {diff}"


# ---- T4: DEQ frontier ---------------------------------------------------------

@pytest.mark.parametrize("device", DEVICES)
def test_t4_deq_exact_at_k_equals_t(device):
    T = 8
    cfg = small_cfg(max_seq_len=T, deq_sweeps=T)
    model = make_model(cfg, device)
    perturb(model)
    idx = rand_idx(cfg, 2, T, device)
    with torch.no_grad():
        cfg.attn_mode = "sequential"
        logits_seq, _ = model(idx)
        cfg.attn_mode = "deq"
        logits_deq, _ = model(idx)
    assert (logits_seq - logits_deq).abs().max().item() < TOL


@pytest.mark.parametrize("device", DEVICES)
def test_t4_deq_frontier_positions(device):
    """After m sweeps, positions j <= m-1 carry exactly their sequential values."""
    T = 8
    cfg = small_cfg(max_seq_len=T, deq_sweeps=T)
    model = make_model(cfg, device)
    perturb(model)
    idx = rand_idx(cfg, 2, T, device)
    with torch.no_grad():
        cfg.attn_mode = "sequential"
        logits_seq, _ = model(idx)
        cfg.attn_mode = "deq"
        for m in range(1, T + 1):
            cfg.deq_sweeps = m
            logits_m, _ = model(idx)
            exact = (logits_seq[:, :m] - logits_m[:, :m]).abs().max().item()
            assert exact < TOL, f"sweeps={m}: frontier positions differ by {exact}"


# ---- T5: causality --------------------------------------------------------------

@pytest.mark.parametrize("device", DEVICES)
def test_t5_causality(device):
    cfg = small_cfg()
    model = make_model(cfg, device)
    perturb(model)
    B, T, t = 2, 16, 7
    idx = rand_idx(cfg, B, T, device)
    emb = model.tok_emb(idx).detach().requires_grad_(True)
    logits, _ = model(inputs_embeds=emb)
    grad, = torch.autograd.grad(logits[:, t].sum(), emb)
    assert grad[:, t + 1:].abs().max().item() == 0.0
    assert grad[:, :t + 1].abs().max().item() > 0.0


# ---- chunked checkpointing: gradients must match the no-checkpoint run --------

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("chunk", [5, 8])
def test_checkpoint_grads_match(device, chunk):
    cfg = small_cfg()
    B, T = 2, 24
    idx = rand_idx(cfg, B, T, device)
    tgt = rand_idx(cfg, B, T, device, seed=100)

    grads = {}
    for c in (0, chunk):
        model = make_model(cfg, device)
        perturb(model)
        model.train()
        cfg.checkpoint_chunk = c
        _, loss = model(idx, targets=tgt)
        loss.backward()
        grads[c] = {n: p.grad.clone() for n, p in model.named_parameters()
                    if p.grad is not None}
    assert grads[0].keys() == grads[chunk].keys()
    for n in grads[0]:
        d = (grads[0][n] - grads[chunk][n]).abs().max().item()
        assert d < 1e-6, f"grad mismatch {n}: {d}"
