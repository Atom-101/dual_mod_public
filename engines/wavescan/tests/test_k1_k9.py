"""WaveScan test suite T-K1..T-K9 (megakernel.md §12).

fp32 build tolerances; bf16 checked statistically where noted. The eager repo
scan is the oracle everywhere. Never optimize past a red test.
"""

import numpy as np
import pytest
import torch

from models.dualmod.config import DualModConfig
from models.dualmod.model import DualModLM
from models.dualmod.rope import apply_rope, build_rope_cache

from engines.wavescan.cells.reference import eager_cell, eager_layer
from engines.wavescan.cells.scan_cell_fwd import chunk_cell_forward
from engines.wavescan.engine.forward import MaskCache, layer_forward, model_forward
from engines.wavescan.engine.schedule import (WaveScanSchedule, exact_sequential,
                                      uniform_chunks)

DEV = "cuda"
FP32_TOL = 1e-5


def small_cfg(**kw):
    d = dict(d_model=128, n_layers=2, n_heads=4, vocab_size=512,
             max_seq_len=64, gate_bias_init=0.0, checkpoint_chunk=0,
             ctx_proj_init_std=0.02)
    d.update(kw)
    return DualModConfig(**d)


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(0)
    cfg = small_cfg()
    model = DualModLM(cfg).to(DEV).float().eval()
    return cfg, model


def rand_inputs(cfg, B, T, seed=1):
    g = torch.Generator(device="cpu").manual_seed(seed)
    xhat = torch.randn(B, T, cfg.d_model, generator=g).to(DEV)
    cos, sin = build_rope_cache(cfg.max_seq_len, cfg.head_dim, cfg.rope_theta)
    return xhat, cos.to(DEV), sin.to(DEV)


def relerr(a, b):
    return ((a - b).norm() / (b.norm() + 1e-12)).item()


# ---- T-K1: single cell ≡ eager per-position steps ---------------------------

@pytest.mark.parametrize("c,a", [(1, 0), (1, 17), (8, 0), (8, 24), (16, 8)])
def test_k1_cell_exact(setup, c, a):
    cfg, model = setup
    attn = model.blocks[0].attn
    B = 3
    xhat, cos, sin = rand_inputs(cfg, B, a + c)
    with torch.no_grad():
        # committed prefix built by the oracle so both paths read identical rows
        if a > 0:
            _, K_comm, V_comm = eager_cell(
                attn, xhat[:, :a], cos[:a], sin[:a],
                xhat.new_zeros(B, cfg.n_heads, 0, cfg.head_dim),
                xhat.new_zeros(B, cfg.n_heads, 0, cfg.head_dim))
        else:
            K_comm = xhat.new_zeros(B, cfg.n_heads, 0, cfg.head_dim)
            V_comm = K_comm.clone()
        o_ref, K_ref, V_ref = eager_cell(attn, xhat[:, a:], cos[a:a + c],
                                         sin[a:a + c], K_comm, V_comm)
        o, K_rows, V_rows = chunk_cell_forward(
            attn, xhat[:, a:], cos[a:a + c], sin[a:a + c], K_comm, V_comm,
            n_sweeps=c)  # K=c sweeps = exact
    assert relerr(attn._split(o), o_ref) < FP32_TOL
    assert relerr(K_rows, K_ref) < FP32_TOL
    assert relerr(V_rows, V_ref) < FP32_TOL


# ---- T-K2: full-layer chunked ≡ eager scan ----------------------------------

def offset_schedule(T):
    """random-offset style: a 3-wide first cell, then 8-chunks, all K=c exact."""
    return [(0, 3, 3)] + [(p + 3, c, c) for p, c, _ in uniform_chunks(T - 3, 8)]


@pytest.mark.parametrize("sched_fn", [
    lambda T: exact_sequential(T),
    lambda T: uniform_chunks(T, 4),          # K=c exact
    lambda T: uniform_chunks(T, 8),
    offset_schedule,
])
def test_k2_layer_chunked_exact(setup, sched_fn):
    cfg, model = setup
    attn = model.blocks[0].attn
    B, T = 2, 48
    xhat, cos, sin = rand_inputs(cfg, B, T)
    sched = sched_fn(T)
    assert sum(c for _, c, _ in sched) == T
    masks = MaskCache(DEV)
    K_cache = torch.zeros(B, cfg.n_heads, cfg.max_seq_len, cfg.head_dim, device=DEV)
    V_cache = torch.zeros_like(K_cache)
    with torch.no_grad():
        o = layer_forward(attn, xhat, cos[:T], sin[:T], sched, K_cache,
                          V_cache, masks)
        o_ref, K_ref, V_ref = eager_layer(attn, xhat, cos[:T], sin[:T])
    assert relerr(o, attn._merge(o_ref)) < FP32_TOL
    assert relerr(K_cache[:, :, :T], K_ref) < FP32_TOL
    assert relerr(V_cache[:, :, :T], V_ref) < FP32_TOL


# ---- T-K3: jacobi(K=c) ≡ exact; per-sweep frontier --------------------------

@pytest.mark.parametrize("c", [4, 8, 16])
def test_k3_nilpotency_and_frontier(setup, c):
    cfg, model = setup
    attn = model.blocks[0].attn
    B, a = 2, 12
    xhat, cos, sin = rand_inputs(cfg, B, a + c, seed=3)
    with torch.no_grad():
        _, K_comm, V_comm = eager_cell(
            attn, xhat[:, :a], cos[:a], sin[:a],
            xhat.new_zeros(B, cfg.n_heads, 0, cfg.head_dim),
            xhat.new_zeros(B, cfg.n_heads, 0, cfg.head_dim))
        o_ex, K_ex, V_ex = eager_cell(attn, xhat[:, a:], cos[a:a + c],
                                      sin[a:a + c], K_comm, V_comm)
        for s in range(1, c + 1):
            o_s, K_s, V_s = chunk_cell_forward(
                attn, xhat[:, a:], cos[a:a + c], sin[a:a + c],
                K_comm, V_comm, n_sweeps=s)
            # frontier: o rows < s exact after s sweeps; cache rows < s exact
            assert relerr(attn._split(o_s)[:, :, :s], o_ex[:, :, :s]) < FP32_TOL
            assert relerr(K_s[:, :, :s], K_ex[:, :, :s]) < FP32_TOL
        # nilpotency at s = c: everything exact
        assert relerr(attn._split(o_s), o_ex) < FP32_TOL
        assert relerr(K_s, K_ex) < FP32_TOL
        assert relerr(V_s, V_ex) < FP32_TOL


# ---- T-K9: RoPE pairing, bounds, append-only --------------------------------

def test_k9_rope_pairing(setup):
    cfg, _ = setup
    dh = cfg.head_dim
    cos, sin = build_rope_cache(8, dh, cfg.rope_theta)
    for i in [0, 1, dh // 2 - 1]:
        e = torch.zeros(1, 1, 1, dh)
        e[..., i] = 1.0
        r = apply_rope(e, cos[3:4], sin[3:4])
        moved = (r[0, 0, 0] != 0).nonzero().flatten().tolist()
        assert set(moved) <= {i, i + dh // 2}, \
            f"rope pair convention violated: basis {i} moved dims {moved}"


def test_k9_append_only_and_bounds(setup):
    cfg, model = setup
    attn = model.blocks[0].attn
    B, T = 2, 40
    T_max = cfg.max_seq_len
    xhat, cos, sin = rand_inputs(cfg, B, T, seed=5)
    masks = MaskCache(DEV)
    K_cache = torch.full((B, cfg.n_heads, T_max, cfg.head_dim), 7.0, device=DEV)
    V_cache = torch.full_like(K_cache, 7.0)
    sched = uniform_chunks(T, 8)
    with torch.no_grad():
        checksums = {}   # pos0 -> bitwise checksum of rows [0, pos0) at write time
        for pos0, c, n_sweeps in sched:
            checksums[pos0] = K_cache[:, :, :pos0].sum().item()
            o_cat, K_rows, V_rows = chunk_cell_forward(
                attn, xhat[:, pos0:pos0 + c], cos[pos0:pos0 + c],
                sin[pos0:pos0 + c], K_cache[:, :, :pos0],
                V_cache[:, :, :pos0], n_sweeps, intra_mask=masks.get(c))
            K_cache[:, :, pos0:pos0 + c] = K_rows
            V_cache[:, :, pos0:pos0 + c] = V_rows
        # after everything ran, every prefix checksum must still hold
        for pos0, s in checksums.items():
            assert K_cache[:, :, :pos0].sum().item() == s, \
                "append-only violated: earlier rows changed"
    # bounds: rows beyond T untouched (still the fill value)
    assert torch.all(K_cache[:, :, T:] == 7.0)
    assert torch.all(V_cache[:, :, T:] == 7.0)


# ---- T-K6: backward vs eager autograd ---------------------------------------

def alloc_run_state(cfg, model, B, dtype=torch.float32):
    L = cfg.n_layers
    h, dh, T_max, d = cfg.n_heads, cfg.head_dim, cfg.max_seq_len, cfg.d_model
    K_caches = [torch.zeros(B, h, T_max, dh, device=DEV, dtype=dtype) for _ in range(L)]
    V_caches = [torch.zeros(B, h, T_max, dh, device=DEV, dtype=dtype) for _ in range(L)]
    xl = [torch.zeros(B, T_max, d, device=DEV, dtype=dtype) for _ in range(L + 1)]
    dK = torch.zeros(B, h, T_max, dh, device=DEV, dtype=torch.float32)
    dV = torch.zeros_like(dK)
    return K_caches, V_caches, xl, dK, dV


def wavescan_fwd_bwd(model, cfg, idx, schedule, d_logits, deterministic=False):
    from engines.wavescan.engine.backward import GradBuffers, model_backward
    B, T = idx.shape
    masks = MaskCache(DEV)
    K_caches, V_caches, xl, dK, dV = alloc_run_state(cfg, model, B)
    xl = [b[:, :T].clone() for b in xl]
    with torch.no_grad():
        logits = model_forward(model, idx, schedule, K_caches, V_caches,
                               masks, xl_saved=xl)
    grads = GradBuffers(list(model.parameters()))
    model_backward(model, idx, schedule, K_caches, V_caches, xl, d_logits,
                   masks, grads, dK, dV, deterministic=deterministic)
    return logits, grads


@pytest.mark.parametrize("sched_fn", [
    lambda T: exact_sequential(T),
    lambda T: uniform_chunks(T, 8),  # jacobi K=c exact
])
def test_k6_grads_vs_eager(setup, sched_fn):
    cfg, model = setup
    B, T = 2, 40
    g = torch.Generator().manual_seed(7)
    idx = torch.randint(0, cfg.vocab_size, (B, T), generator=g).to(DEV)
    d_logits = torch.randn(B, T, cfg.vocab_size, generator=g).to(DEV) * 1e-2

    # eager oracle grads
    model.zero_grad(set_to_none=True)
    logits_e, _ = model(idx)
    (logits_e * d_logits).sum().backward()
    eager_grads = {id(p): p.grad.clone() for p in model.parameters()}

    logits_w, grads = wavescan_fwd_bwd(model, cfg, idx, sched_fn(T), d_logits)
    assert relerr(logits_w, logits_e) < 1e-4
    for name, p in model.named_parameters():
        ge, gw = eager_grads[id(p)], grads.get(p)
        cos_sim = torch.nn.functional.cosine_similarity(
            ge.flatten().float(), gw.flatten(), dim=0).item()
        assert relerr(gw, ge.float()) < 1e-4, f"{name}: rel {relerr(gw, ge.float())}"
        assert cos_sim > 0.9999, f"{name}: cos {cos_sim}"
    model.zero_grad(set_to_none=True)


def test_k6_approx_schedule_grads_functional():
    """For K<c (approximate) schedules the eager scan is NOT the oracle — the
    correct reference is autograd through the SAME approximate forward, built
    functionally (cat-based cache, no in-place) at tiny scale."""
    torch.manual_seed(11)
    cfg = small_cfg(d_model=64, n_layers=2, n_heads=2, vocab_size=97,
                    max_seq_len=32)
    model = DualModLM(cfg).to(DEV).float().eval()
    B, T = 2, 24
    sched = uniform_chunks(T, 8, n_sweeps=2)  # K=2 < c=8: approximate
    idx = torch.randint(0, cfg.vocab_size, (B, T), device=DEV)
    d_logits = torch.randn(B, T, cfg.vocab_size, device=DEV) * 1e-2

    # functional forward with autograd
    masks = MaskCache(DEV)
    x = model.tok_emb(idx)
    cos, sin = model.rope_cos[:T], model.rope_sin[:T]
    for blk in model.blocks:
        xhat = blk.attn_norm(x)
        K_comm = x.new_zeros(B, cfg.n_heads, 0, cfg.head_dim)
        V_comm = K_comm.clone()
        outs = []
        for pos0, c, K in sched:
            o_cat, K_rows, V_rows = chunk_cell_forward(
                blk.attn, xhat[:, pos0:pos0 + c], cos[pos0:pos0 + c],
                sin[pos0:pos0 + c], K_comm, V_comm, K, masks.get(c))
            K_comm = torch.cat([K_comm, K_rows], dim=2)
            V_comm = torch.cat([V_comm, V_rows], dim=2)
            outs.append(o_cat)
        x = x + blk.attn.wo(torch.cat(outs, dim=1))
        x = x + blk.mlp(blk.mlp_norm(x))
    logits_f = model.lm_head(model.norm_f(x))
    model.zero_grad(set_to_none=True)
    (logits_f * d_logits).sum().backward()
    ref_grads = {id(p): p.grad.clone() for p in model.parameters()}

    logits_w, grads = wavescan_fwd_bwd(model, cfg, idx, sched, d_logits)
    assert relerr(logits_w, logits_f) < 1e-4
    for name, p in model.named_parameters():
        ge, gw = ref_grads[id(p)], grads.get(p)
        assert relerr(gw, ge.float()) < 1e-4, f"{name}: rel {relerr(gw, ge.float())}"
    model.zero_grad(set_to_none=True)


# ---- T-K4: Phase A graphs ≡ eager logits ------------------------------------

@pytest.mark.parametrize("sched_fn", [
    lambda T: exact_sequential(T),
    lambda T: uniform_chunks(T, 8),  # K=c exact
])
def test_k4_phase_a_graphs_vs_eager(setup, sched_fn):
    from engines.wavescan.engine.graphs import WaveScanEngine
    cfg, model = setup
    B, T = 2, 48
    g = torch.Generator().manual_seed(13)
    idx1 = torch.randint(0, cfg.vocab_size, (B, T), generator=g).to(DEV)
    idx2 = torch.randint(0, cfg.vocab_size, (B, T), generator=g).to(DEV)
    tgt = torch.randint(0, cfg.vocab_size, (B, T), generator=g).to(DEV)

    eng = WaveScanEngine(model, [sched_fn(T)], B, T, keep_logits=True,
                         use_graphs=True)
    with torch.no_grad():
        le1, _ = model(idx1)
        le2, _ = model(idx2)
    lw1 = eng.forward(idx1, 0, targets=tgt).clone()
    lw2 = eng.forward(idx2, 0, targets=tgt).clone()   # replay #2, new input
    assert relerr(lw1, le1) < 1e-4
    assert relerr(lw2, le2) < 1e-4
    # replay must equal the uncaptured engine bit-for-bit (same kernels)
    eng2 = WaveScanEngine(model, [sched_fn(T)], B, T, keep_logits=True,
                          use_graphs=False)
    lu = eng2.forward(idx2, 0, targets=tgt)
    assert torch.equal(lw2, lu)


def test_k4_phase_a_graph_backward(setup):
    """Graphed backward ≡ uncaptured backward ≡ eager grads (exact schedule)."""
    from engines.wavescan.engine.graphs import WaveScanEngine
    cfg, model = setup
    B, T = 2, 40
    g = torch.Generator().manual_seed(17)
    idx = torch.randint(0, cfg.vocab_size, (B, T), generator=g).to(DEV)
    tgt = torch.randint(0, cfg.vocab_size, (B, T), generator=g).to(DEV)

    # capture FIRST: a kept-alive eager backward graph pins AccumulateGrad
    # nodes to the legacy stream and poisons capture (engine contract).
    eng = WaveScanEngine(model, [uniform_chunks(T, 8)], B, T,
                         keep_logits=False, use_graphs=True)

    model.zero_grad(set_to_none=True)
    logits_e, _ = model(idx)
    loss_e = torch.nn.functional.cross_entropy(
        logits_e.float().reshape(-1, cfg.vocab_size), tgt.reshape(-1))
    loss_e.backward()
    eager_grads = {id(p): p.grad.clone() for p in model.parameters()}

    loss_w = eng.train_step(idx, tgt, 0).clone()
    assert abs(loss_w.item() - loss_e.item()) < 1e-4
    for name, p in model.named_parameters():
        ge, gw = eager_grads[id(p)], eng.grads.get(p)
        assert relerr(gw, ge.float()) < 1e-4, f"{name}: {relerr(gw, ge.float())}"
    model.zero_grad(set_to_none=True)


# ---- T-K8: determinism ------------------------------------------------------

def test_k8_determinism_50_steps():
    from engines.wavescan.engine.graphs import WaveScanEngine
    torch.use_deterministic_algorithms(True)
    try:
        losses = []
        for rep in range(2):
            torch.manual_seed(123)
            cfg = small_cfg(d_model=64, n_layers=2, n_heads=2, vocab_size=97,
                            max_seq_len=32)
            model = DualModLM(cfg).to(DEV).float()
            B, T = 2, 32
            sched = WaveScanSchedule(pool_size=2, seed=5).draw_pool(T)
            eng = WaveScanEngine(model, sched, B, T, use_graphs=True)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
            g = torch.Generator().manual_seed(9)
            ls = []
            for step in range(50):
                idx = torch.randint(0, cfg.vocab_size, (B, T), generator=g).to(DEV)
                tgt = torch.randint(0, cfg.vocab_size, (B, T), generator=g).to(DEV)
                loss = eng.train_step(idx, tgt, sid=step % 2)
                eng.finish_grads()
                opt.step()
                opt.zero_grad(set_to_none=False)
                ls.append(loss.item())
            losses.append(ls)
        assert losses[0] == losses[1], "determinism violated across reruns"
    finally:
        torch.use_deterministic_algorithms(False)


# ---- T-K5: Phase B wavefront ≡ Phase A --------------------------------------

def test_k5_wave_vs_phase_a(setup):
    from engines.wavescan.engine.graphs import WaveScanEngine
    cfg, model = setup
    B, T = 2, 48
    sched = WaveScanSchedule(pool_size=2, seed=3).draw_pool(T)
    g = torch.Generator().manual_seed(23)
    idx = torch.randint(0, cfg.vocab_size, (B, T), generator=g).to(DEV)
    tgt = torch.randint(0, cfg.vocab_size, (B, T), generator=g).to(DEV)

    eng_a = WaveScanEngine(model, sched, B, T, keep_logits=True, use_graphs=True)
    eng_b = WaveScanEngine(model, sched, B, T, keep_logits=True, use_graphs=True,
                           wave=True)
    for sid in range(2):
        la = eng_a.forward(idx, sid, targets=tgt).clone()
        lb = eng_b.forward(idx, sid, targets=tgt).clone()
        assert relerr(lb, la) < 1e-4, f"sched {sid}: {relerr(lb, la)}"
        # loss scalar agreement too
        assert abs(eng_a.loss.item() - eng_b.loss.item()) < 1e-4


def test_k5_wave_train_step_grads(setup):
    """Wave forward + (layer-major) backward: grads match Phase A train_step."""
    from engines.wavescan.engine.graphs import WaveScanEngine
    cfg, model = setup
    B, T = 2, 40
    sched = [uniform_chunks(T, 8, 2)]  # approximate schedule: same for both
    g = torch.Generator().manual_seed(29)
    idx = torch.randint(0, cfg.vocab_size, (B, T), generator=g).to(DEV)
    tgt = torch.randint(0, cfg.vocab_size, (B, T), generator=g).to(DEV)

    eng_a = WaveScanEngine(model, sched, B, T, use_graphs=True)
    eng_b = WaveScanEngine(model, sched, B, T, use_graphs=True, wave=True)
    la = eng_a.train_step(idx, tgt, 0).clone()
    lb = eng_b.train_step(idx, tgt, 0).clone()
    assert abs(la.item() - lb.item()) < 1e-4
    for name, p in model.named_parameters():
        ga, gb = eng_a.grads.get(p), eng_b.grads.get(p)
        assert relerr(gb, ga) < 1e-4, f"{name}: {relerr(gb, ga)}"


# ---- Phase C: backward wavefront ≡ Phase A backward (§8 acceptance) ----------

def test_phase_c_wave_backward_grads(setup):
    from engines.wavescan.engine.graphs import WaveScanEngine
    cfg, model = setup
    B, T = 2, 40
    sched = [uniform_chunks(T, 8, 2)]
    g = torch.Generator().manual_seed(31)
    idx = torch.randint(0, cfg.vocab_size, (B, T), generator=g).to(DEV)
    tgt = torch.randint(0, cfg.vocab_size, (B, T), generator=g).to(DEV)

    eng_a = WaveScanEngine(model, sched, B, T, use_graphs=True)
    eng_c = WaveScanEngine(model, sched, B, T, use_graphs=True,
                           wave=True, wave_bwd=True)
    la = eng_a.train_step(idx, tgt, 0).clone()
    lc = eng_c.train_step(idx, tgt, 0).clone()
    assert abs(la.item() - lc.item()) < 1e-4
    for name, p in model.named_parameters():
        ga, gc = eng_a.grads.get(p), eng_c.grads.get(p)
        assert relerr(gc, ga) < 1e-4, f"{name}: {relerr(gc, ga)}"


# ---- compiled cells: same grads as eager (engine compile_cells=True) --------

def test_compiled_cells_grads_vs_eager(setup):
    from engines.wavescan.engine.graphs import WaveScanEngine
    cfg, model = setup
    B, T = 2, 40
    g = torch.Generator().manual_seed(37)
    idx = torch.randint(0, cfg.vocab_size, (B, T), generator=g).to(DEV)
    tgt = torch.randint(0, cfg.vocab_size, (B, T), generator=g).to(DEV)

    eng = WaveScanEngine(model, [uniform_chunks(T, 8)], B, T,
                         use_graphs=True, compile_cells=True)
    loss_w = eng.train_step(idx, tgt, 0).clone()

    model.zero_grad(set_to_none=True)
    logits_e, _ = model(idx)
    loss_e = torch.nn.functional.cross_entropy(
        logits_e.float().reshape(-1, cfg.vocab_size), tgt.reshape(-1))
    loss_e.backward()
    assert abs(loss_w.item() - loss_e.item()) < 1e-4
    for name, p in model.named_parameters():
        gw = eng.grads.get(p)
        assert relerr(gw, p.grad.float()) < 2e-4, f"{name}: {relerr(gw, p.grad.float())}"
    model.zero_grad(set_to_none=True)


# ---- fused-sweep path (Triton K1/K2 + hand backward) ≡ eager ----------------

def test_fused_sweep_engine_grads(setup):
    from engines.wavescan.engine.graphs import WaveScanEngine
    cfg, model = setup
    B, T = 2, 64
    g = torch.Generator().manual_seed(43)
    idx = torch.randint(0, cfg.vocab_size, (B, T), generator=g).to(DEV)
    tgt = torch.randint(0, cfg.vocab_size, (B, T), generator=g).to(DEV)

    # exact schedule (K=c) vs eager autograd
    eng = WaveScanEngine(model, [uniform_chunks(T, 16)], B, T,
                         use_graphs=True, fused_sweep=True)
    loss_w = eng.train_step(idx, tgt, 0).clone()
    model.zero_grad(set_to_none=True)
    logits, _ = model(idx)
    loss_e = torch.nn.functional.cross_entropy(
        logits.float().reshape(-1, cfg.vocab_size), tgt.reshape(-1))
    loss_e.backward()
    assert abs(loss_w.item() - loss_e.item()) < 1e-4
    for name, p in model.named_parameters():
        e = relerr(eng.grads.get(p), p.grad.float())
        assert e < 1e-4, f"{name}: {e}"
    model.zero_grad(set_to_none=True)
    del logits, loss_e   # capture contract: no live eager autograd graphs

    # approximate schedule vs the reference engine (same schedule)
    eng_a = WaveScanEngine(model, [uniform_chunks(T, 16, 4)], B, T,
                           use_graphs=True, fused_sweep=True)
    eng_r = WaveScanEngine(model, [uniform_chunks(T, 16, 4)], B, T,
                           use_graphs=False)
    la = eng_a.train_step(idx, tgt, 0).clone()
    lr = eng_r.train_step(idx, tgt, 0).clone()
    assert abs(la.item() - lr.item()) < 1e-4
    for name, p in model.named_parameters():
        e = relerr(eng_a.grads.get(p), eng_r.grads.get(p))
        assert e < 1e-4, f"{name}: {e}"
