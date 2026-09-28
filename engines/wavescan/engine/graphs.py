"""Phase A/B/C execution engine: CUDA-graph capture/replay (megakernel.md §6).

Why per-CELL graphs, not one whole-step graph: allocations freed during a
stream capture cannot be reclaimed by the native caching allocator (event
queries are illegal inside capture), so a monolithic capture's private pool
grows with the TOTAL transient volume of the step — ~250 GB at dev63m B=64 —
and OOMs at capture_end. Capturing each (layer, cell) as its own graph bounds
each pool by one cell's live set; pools are PER LAYER so (a) blocks recycle
across that layer's cell captures, (b) Phase B/C can replay different layers
concurrently on different streams without pool aliasing races. All pool
memory is intra-graph transient — every cross-graph tensor (caches, xl,
dx_layers, grad-cache, grad buffers, loss) lives in static engine buffers —
so within one stream the replay order of graphs sharing a pool is
unconstrained.

Replay cost: ~2×(cells) graph launches per step (≈1 ms CPU at dev63m c32),
vs the eager path's ~10^5 python-dispatched kernels.

Modes: fused-loss (default; loss computed from x^L in T-chunks, no [B,T,V]
tensor) or keep_logits (tests/WaveScanFn). wave=True replays forward cells on
per-layer streams with events (Phase B); wave_bwd=True does the same for the
reverse scan (Phase C). DDP: no comm inside graphs — finish_grads() does one
manual all-reduce over the fp32 grad buffers (§6).
"""

import gc
import time

import torch

from .backward import GradBuffers, _grad, fused_loss_forward
from .forward import MaskCache, layer_forward
from ..cells.scan_cell_fwd import (chunk_cell_forward, chunk_cell_forward_diff, chunk_cell_forward_inplace)


class WaveScanEngine:
    def __init__(self, model, schedules, B, T, device="cuda",
                 autocast_bf16=False, keep_logits=False, loss_chunk=64,
                 use_graphs=True, wave=False, wave_bwd=False, use_fused=True,
                 compile_cells=False, fused_sweep=False, cap_priority=0,
                 share_layer_pools=False, sat_tau=0.0, grad_bf16_ids=None):
        cfg = model.cfg
        self._grad_bf16_ids = grad_bf16_ids
        self.model, self.cfg = model, cfg
        self.schedules = schedules
        self.B, self.T = B, T
        self.device = device
        self.autocast_bf16 = autocast_bf16
        self.keep_logits = keep_logits
        self.loss_chunk = loss_chunk
        self.use_graphs = use_graphs
        self.wave = wave
        self.wave_bwd = wave_bwd
        # share_layer_pools: capture the per-layer proj/projb/blockmlp/close
        # graphs into pool slot 0 instead of per-layer pools. Layer-major
        # replay is strictly sequential, so sharing is safe (same argument as
        # the cell graphs below) and collapses pool retention from
        # L x per-layer transients to ~one layer's worth — each captured
        # graph permanently reserves its transient working set in its pool,
        # which is what "peak allocated" does NOT show. Required on 80GB
        # parts (H100) at flagship scale; default off preserves the exact
        # B300-validated capture layout.
        assert not (share_layer_pools and (wave or wave_bwd)), \
            "share_layer_pools requires layer-major replay (no wave)"
        self.share_layer_pools = share_layer_pools
        # saturating backward (stability_dynamics.md §8): per-row norm cap on
        # the inter-cell gradient carry (dK/dV grad-cache accumulation + dx
        # slice), direction preserved; per-row zero on non-finite. 0 = off.
        # Engine-level (both fused and eager cell paths); plain layer-major
        # replay only — the per-layer reference capture is loop-ordered.
        assert sat_tau == 0 or not (use_graphs or wave_bwd), \
            "sat_tau requires no-graphs layer-major backward"
        self.sat_tau = float(sat_tau)
        self.sat_events = torch.zeros((), device=device)
        self.sat_rows_zeroed = torch.zeros((), device=device)
        self.chain_rho_max = torch.zeros((), device=device)
        self._sat_ref = None
        L, h, dh, d = cfg.n_layers, cfg.n_heads, cfg.head_dim, cfg.d_model
        self.L = L
        act = torch.bfloat16 if autocast_bf16 else torch.float32

        self.idx = torch.zeros(B, T, dtype=torch.long, device=device)
        self.targets = torch.zeros(B, T, dtype=torch.long, device=device)
        self.K_caches = [torch.zeros(B, h, T, dh, device=device, dtype=act)
                         for _ in range(L)]
        self.V_caches = [torch.zeros(B, h, T, dh, device=device, dtype=act)
                         for _ in range(L)]
        self.xl = [torch.zeros(B, T, d, device=device, dtype=act)
                   for _ in range(L + 1)]
        # per-layer grad-cache + per-layer dx: needed concurrently by Phase C,
        # and harmless (few GB fp32) in layer-major mode — one code path.
        self.dK = [torch.zeros(B, h, T, dh, device=device, dtype=torch.float32)
                   for _ in range(L)]
        self.dV = [torch.zeros_like(self.dK[0]) for _ in range(L)]
        self.dx = [torch.zeros(B, T, d, device=device, dtype=torch.float32)
                   for _ in range(L + 1)]
        self.grads = GradBuffers(list(model.parameters()), bf16_ids=self._grad_bf16_ids)
        self.masks = MaskCache(device)
        self.loss = torch.zeros((), device=device, dtype=torch.float32)
        if keep_logits:
            self.logits = torch.zeros(B, T, cfg.vocab_size, device=device,
                                      dtype=act)
            self.d_logits = torch.zeros_like(self.logits)

        self.fused_sweep = fused_sweep
        if fused_sweep:
            assert not wave and not wave_bwd, \
                "fused_sweep is layer-major (shared per-layer proj buffers)"
            from ..cells.fused_sweep import (FusedSweepCell, LayerBufs,
                                             _CellBwdScratch)
            self.fs = FusedSweepCell(model)
            self._fs_mod = getattr(cfg, "ctx_mod", "linear")
            c_max = max(cc for sched in schedules for _, cc, _ in sched)
            cs = sorted({c2 for sched in schedules for _, c2, _ in sched})
            # per-c sweep buffers: a [:, :c] slice of a c_max buffer is
            # non-contiguous, and the sweep kernels index it as contiguous
            # [B,c,d] (correct only when c == c_max)
            self.fs_u = {cc: torch.zeros(B, cc, d, device=device, dtype=act)
                         for cc in cs}
            self.fs_o = {cc: torch.zeros(B, cc, d, device=device,
                                         dtype=torch.float32) for cc in cs}
            self.fs_P = LayerBufs(B, h, T, dh, d, act, device,
                                  ctx_mod=self._fs_mod)
            self.fs_ocat = [torch.zeros(B, T, d, device=device, dtype=act)
                            for _ in range(L)]
            # One scratch per DISTINCT chunk size: _CellBwdScratch.views()
            # does [:, :c].reshape(...) which is a true view only when
            # c == c_max — at c < c_max the non-contiguous slice reshape
            # silently COPIES, severing the kernels' pointer-accumulation
            # (root cause of the c-ladder NaN backward). Per-c instances are
            # exact and cost only MBs.
            self.fs_scratch = {cc: _CellBwdScratch(B, h, cc, dh, d, device)
                               for cc in cs}
            # variants: rows 4d:5d carry the folded u-driven pre-GEMM grad,
            # routed to v_map / v_pre-u-half in the bwd epilogue
            pk = 5 * d if self._fs_mod != "linear" else 4 * d
            self.dW_pack = [torch.zeros(pk, d, device=device,
                                        dtype=torch.float32) for _ in range(L)]
            self.dnormw = [torch.zeros(d, device=device, dtype=torch.float32)
                           for _ in range(L)]
            # w_vctx grad accumulator (fp32, applied in the bwd epilogue):
            # re-routed for variants because the packed dW_pack[d:2d] block is
            # forced to 0 (v_ctx = w_vctx(v_in), not w_vctx(u), for non-linear
            # ctx_mod). v_gate / v_pre-xhat-half grads flow through the
            # deferred hoist chain in layer_close instead.
            if self._fs_mod != "linear":
                self.dW_vctx = [torch.zeros(d, d, device=device,
                                            dtype=torch.float32)
                                for _ in range(L)]
        self.fused = None
        if use_fused:
            try:
                from ..cells.triton_kernels import FusedRefine
                self.fused = FusedRefine(model)
            except (ImportError, AssertionError):
                pass          # variant configs / no triton: reference path

        # torch.compile the cell bodies: inductor fuses the per-sweep
        # elementwise chains (fwd AND the AOT-generated backward) far past
        # what the hand kernels reach — the step is tiny-op bound, so this is
        # the main lever. dynamic=True keeps one artifact across cell shapes.
        self.compile_cells = compile_cells
        self._fwd_cell_fn = chunk_cell_forward_inplace
        # The backward recompute must differentiate the SAME function the
        # forward ran (CMK incident, Jul 14: the packed diff cell silently
        # reimplemented linear-only semantics — garbage grads for every
        # ctx_mod/kmod config). The packed cell now reproduces
        # PACKED_CTX_MODS x kmod (gate-logit pin); anything else falls back
        # to the verbatim refine_kv cell. Gate ANY refine_kv change with
        # analysis/kmod_parity_check.py --prod.
        from ..cells.scan_cell_fwd import PACKED_CTX_MODS
        _needs_verbatim = (
            getattr(cfg, "ctx_mod", "linear") not in PACKED_CTX_MODS
            or getattr(cfg, "kmod_vraw_bound", "none") != "none"
            # independent-v changes the value blend formula (g_raw*v_raw +
            # g_ctx*v_ctx); the packed diff cell + FusedRefine kernels bake
            # the convex g/(1-g) blend, so route to the autograd verbatim
            # cell (refine_kv is the single source of truth). v1: no kernel
            # surgery — dev-scale runs are fine on compiled cells.
            or getattr(cfg, "gate_type_v", "convex") == "independent")
        if _needs_verbatim:
            def _verbatim_diff(attn, xhat_c, cos_c, sin_c, K_comm, V_comm,
                               n_sweeps, intra_mask, use_triton_refine=False):
                return chunk_cell_forward(attn, xhat_c, cos_c, sin_c,
                                          K_comm, V_comm, n_sweeps,
                                          intra_mask=intra_mask)
            self._diff_cell_fn = _verbatim_diff
        else:
            self._diff_cell_fn = chunk_cell_forward_diff
        if compile_cells:
            import torch._dynamo as _dyn
            _dyn.config.recompile_limit = max(_dyn.config.recompile_limit, 64)
            _dyn.config.cache_size_limit = max(_dyn.config.cache_size_limit, 64)
            self._fwd_cell_fn = torch.compile(chunk_cell_forward_inplace,
                                              dynamic=True)
            self._diff_cell_fn = torch.compile(self._diff_cell_fn,
                                               dynamic=True)

        self.streams = [torch.cuda.Stream() for _ in range(L)]
        # Dedicated CAPTURE stream per layer: cuBLAS workspaces are keyed by
        # (device, stream); if two layers' cell graphs are captured on the
        # same (pool-recycled) stream they bake the SAME workspace pointers
        # and concurrent wave replay races on that scratch (observed as
        # nondeterministic Phase C grads; bwd wgrad GEMMs use workspace).
        # One capture stream per layer ⇒ workspaces disjoint across layers,
        # and replay within a layer is sequential ⇒ safe.
        # cap_priority: concurrent multi-engine replay (microbatch streams)
        # needs DISJOINT capture streams across engines or their baked cuBLAS
        # workspaces collide — the stream pool has 32 per priority, so a
        # second engine draws from another priority pool.
        self.cap_streams = [torch.cuda.Stream(priority=cap_priority)
                            for _ in range(L)]
        self._misc_cap_stream = torch.cuda.Stream(priority=cap_priority)
        self._pools = [None] * L          # per-layer graph memory pools
        self._misc_pool = None            # embed/head/loss segments
        # graphs: keyed [sid][li][k]
        self._gf_cells, self._gb_cells = {}, {}
        self._g_embed = self._g_loss = self._g_headbwd = self._g_embbwd = None
        self._g_prologue = self._g_bwdepi = None
        if use_graphs:
            self._capture_all()

    # ---- segment bodies (capturable: static in/out buffers only) -----------

    def _autocast(self):
        if self.autocast_bf16:
            return torch.autocast("cuda", dtype=torch.bfloat16,
                                  cache_enabled=False)
        return torch.autocast("cuda", enabled=False)

    @torch.no_grad()
    def _embed_body(self):
        with self._autocast():
            self.xl[0].copy_(self.model.tok_emb(self.idx))
            if self.fused_sweep:
                self.fs.pack()
            if self.fused is not None:
                self.fused.pack()     # refresh packed ctx/gate weights per step

    @torch.no_grad()
    def _cell_fwd_body(self, li, sid, k):
        pos0, c, n_sweeps = self.schedules[sid][k]
        blk = self.model.blocks[li]
        cos = self.model.rope_cos[pos0:pos0 + c]
        sin = self.model.rope_sin[pos0:pos0 + c]
        with self._autocast():
            x_c = self.xl[li][:, pos0:pos0 + c]
            if self.fused_sweep:
                o_cat = self.fs.cell_forward(
                    li, self.fs_P, cos, sin, self.K_caches[li],
                    self.V_caches[li], pos0, n_sweeps,
                    self.fs_u[c], self.fs_o[c],
                    ocat_out=self.fs_ocat[li][:, pos0:pos0 + c], x_c=x_c)
                x_mid = x_c + blk.attn.wo(o_cat.to(x_c.dtype))
                self.xl[li + 1][:, pos0:pos0 + c] = \
                    x_mid + blk.mlp(blk.mlp_norm(x_mid))
                return
            xhat_c = blk.attn_norm(x_c)
            o_cat = self._fwd_cell_fn(
                blk.attn, xhat_c, cos, sin, self.K_caches[li],
                self.V_caches[li], pos0, n_sweeps, self.masks.get(c),
                fused=None if self.compile_cells else self.fused)
            x_mid = x_c + blk.attn.wo(o_cat)
            self.xl[li + 1][:, pos0:pos0 + c] = \
                x_mid + blk.mlp(blk.mlp_norm(x_mid))

    @torch.no_grad()
    def _loss_body(self):
        with self._autocast():
            if self.keep_logits:
                self.logits.copy_(
                    self.model.lm_head(self.model.norm_f(self.xl[self.L])))
            self.loss.copy_(fused_loss_forward(
                self.model, self.xl[self.L], self.targets, self.loss_chunk))

    @torch.no_grad()
    def _prologue_body(self):
        self.grads.zero_()
        for li in range(self.L):
            self.dK[li].zero_()
            self.dV[li].zero_()
        # extended hygiene (PDN-DM-s42 storm #2 replayed clean => a SECOND
        # persistent-state class beyond fs_scratch carries blown-backward
        # residue): the dx ladder and the per-layer backward accumulators.
        # In the healthy path these are fully rewritten each step, so
        # zeroing is numerically neutral (certified by the grads-vs-eager
        # suite); ~2ms/step.
        for t in self.dx:
            t.zero_()
        if self.fused_sweep:
            for name in ("dq", "dvraw", "dkself", "dss"):
                getattr(self.fs_P, name).zero_()
        if self.fused_sweep:
            for li in range(self.L):
                self.dW_pack[li].zero_()
                self.dnormw[li].zero_()
                if self._fs_mod != "linear":
                    self.dW_vctx[li].zero_()
            # scratch hygiene: an inf/NaN lodged in the cell-bwd accumulators
            # by one blown backward PERSISTS and poisons every later step
            # (observed live on PDN-DM-s42: one bomb at step 2550, then every
            # backward 1e3-1e15 while forward stayed clean). ~50MB of zero_
            # per step is noise next to the step itself.
            for scr in self.fs_scratch.values():
                for v in vars(scr).values():
                    if torch.is_tensor(v):
                        v.zero_()
                    elif isinstance(v, (list, tuple)):
                        for x in v:
                            if torch.is_tensor(x):
                                x.zero_()

    @torch.no_grad()
    def _head_bwd_body(self):
        model = self.model
        head_params = [model.norm_f.weight, model.lm_head.weight]
        with self._autocast():
            if self.keep_logits:
                x_fin = self.xl[self.L].detach().requires_grad_(True)
                with torch.enable_grad():
                    logits = model.lm_head(model.norm_f(x_fin))
                gs = _grad([logits], [x_fin] + head_params, [self.d_logits])
                self.dx[self.L].copy_(gs[0])
                for p, g in zip(head_params, gs[1:]):
                    self.grads.add(p, g)
            else:
                inv_n = 1.0 / (self.B * self.T)
                for t0 in range(0, self.T, self.loss_chunk):
                    t1 = min(t0 + self.loss_chunk, self.T)
                    x_fin = self.xl[self.L][:, t0:t1].detach().requires_grad_(True)
                    with torch.enable_grad():
                        logits = model.lm_head(model.norm_f(x_fin))
                        loss = torch.nn.functional.cross_entropy(
                            logits.float().reshape(-1, logits.shape[-1]),
                            self.targets[:, t0:t1].reshape(-1),
                            reduction="sum") * inv_n
                    gs = _grad([loss], [x_fin] + head_params,
                               [torch.ones_like(loss)])
                    self.dx[self.L][:, t0:t1] = gs[0]
                    for p, g in zip(head_params, gs[1:]):
                        self.grads.add(p, g)

    @torch.no_grad()
    @torch.no_grad()
    def _saturate_carry(self, li, pos0, c):
        """Gradient ADC (stability_dynamics.md §8): after one cell's
        backward, cap the per-row norm of the carry this layer still
        propagates — the dK/dV grad-cache accumulation [:, :, :pos0] plus
        this cell's dx slice — at sat_tau x the row's layer-entry gradient
        norm. Direction preserved (row rescale); non-finite rows zeroed.
        Forward dynamics untouched; healthy rows multiply by 1.0."""
        B = self.B
        parts = [self.dx[li][:, pos0:pos0 + c]]
        if pos0 > 0:
            parts += [self.dK[li][:, :, :pos0], self.dV[li][:, :, :pos0]]
        sq = torch.zeros(B, device=self.dx[li].device, dtype=torch.float32)
        for t in parts:
            sq += t.float().pow(2).flatten(1).sum(1)
        n = sq.sqrt()
        bad = ~torch.isfinite(n)
        cap = self.sat_tau * self._sat_ref
        scale = torch.where(bad, torch.zeros_like(n),
                            (cap / n.clamp_min(1e-30)).clamp(max=1.0))
        self.chain_rho_max = torch.maximum(
            self.chain_rho_max,
            torch.where(bad, torch.full_like(n, float("inf")),
                        n / self._sat_ref).max())
        self.sat_events += (scale < 1.0).sum()
        self.sat_rows_zeroed += bad.sum()
        for t in parts:
            t.mul_(scale.view(B, *([1] * (t.dim() - 1))).to(t.dtype))

    def _cell_bwd_body(self, li, sid, k):
        pos0, c, n_sweeps = self.schedules[sid][k]
        if self.fused_sweep:
            cos = self.model.rope_cos[pos0:pos0 + c]
            sin = self.model.rope_sin[pos0:pos0 + c]
            with self._autocast():
                self.fs.cell_backward(
                    li, self.fs_P, self.xl[li][:, pos0:pos0 + c], cos, sin,
                    self.K_caches[li], self.V_caches[li], pos0, n_sweeps,
                    self.dK[li][:, :, pos0:pos0 + c],
                    self.dV[li][:, :, pos0:pos0 + c], self.grads,
                    self.dK[li], self.dV[li], self.fs_scratch[c],
                    self.dW_pack[li], self.dnormw[li],
                    self.dW_vctx[li] if self._fs_mod != "linear" else None)
            if self.sat_tau > 0:
                self._saturate_carry(li, pos0, c)
            return
        blk = self.model.blocks[li]
        params = list(blk.parameters())
        cos = self.model.rope_cos[pos0:pos0 + c]
        sin = self.model.rope_sin[pos0:pos0 + c]
        dK, dV = self.dK[li], self.dV[li]
        with self._autocast():
            x_c = self.xl[li][:, pos0:pos0 + c].detach().requires_grad_(True)
            K_comm = self.K_caches[li][:, :, :pos0].detach().requires_grad_(True)
            V_comm = self.V_caches[li][:, :, :pos0].detach().requires_grad_(True)
            with torch.enable_grad():
                xhat_c = blk.attn_norm(x_c)
                if self.compile_cells:
                    o_cat, K_rows, V_rows = self._diff_cell_fn(
                        blk.attn, xhat_c, cos, sin, K_comm, V_comm, n_sweeps,
                        intra_mask=self.masks.get(c), use_triton_refine=False)
                elif self.fused is not None:
                    o_cat, K_rows, V_rows = self._diff_cell_fn(
                        blk.attn, xhat_c, cos, sin, K_comm, V_comm, n_sweeps,
                        intra_mask=self.masks.get(c))
                else:
                    o_cat, K_rows, V_rows = chunk_cell_forward(
                        blk.attn, xhat_c, cos, sin, K_comm, V_comm, n_sweeps,
                        intra_mask=self.masks.get(c))
                x_mid = x_c + blk.attn.wo(o_cat)
                x_next = x_mid + blk.mlp(blk.mlp_norm(x_mid))
            gs = _grad([x_next, K_rows, V_rows],
                       [x_c, K_comm, V_comm] + params,
                       [self.dx[li + 1][:, pos0:pos0 + c].to(x_next.dtype),
                        dK[:, :, pos0:pos0 + c].to(K_rows.dtype),
                        dV[:, :, pos0:pos0 + c].to(V_rows.dtype)])
            self.dx[li][:, pos0:pos0 + c] = gs[0].float()
            if pos0 > 0:
                dK[:, :, :pos0] += gs[1].float()
                dV[:, :, :pos0] += gs[2].float()
            for p, g in zip(params, gs[3:]):
                self.grads.add(p, g)
        if self.sat_tau > 0:
            self._saturate_carry(li, pos0, c)

    @torch.no_grad()
    def _proj_body(self, li, zero_grads=False):
        with self._autocast():
            self.fs.layer_proj(li, self.xl[li], self.fs_P,
                               zero_grads=zero_grads)

    @torch.no_grad()
    def _blockmlp_bwd_body(self, li):
        with self._autocast():
            self.fs.layer_blockmlp_bwd(li, self.xl[li], self.fs_ocat[li],
                                       self.dx[li + 1], self.dx[li],
                                       self.fs_P, self.grads)

    @torch.no_grad()
    def _close_body(self, li):
        with self._autocast():
            self.fs.layer_close(li, self.xl[li], self.fs_P, self.dx[li],
                                self.grads)

    def _accum_ctx_dW(self, mat, dW):
        """Accumulate a full [d_out, d_in] dW into a ctx operator's grad
        buffer. ctx_rank>0: mat is LoRA-factored (W = up@down) — project the
        composed dW onto the factors (d_down = up^T @ dW, d_up = dW @ down^T)
        and accumulate into the FACTOR grad buffers, exactly mirroring the
        LoRA-gate chain in _bwd_epilogue_body / layer_close. ctx_rank==0: the
        matrix is a dense nn.Linear — straight into its weight buffer
        (byte-identical to the pre-ctx_rank path)."""
        if getattr(self.cfg, "ctx_rank", 0) > 0:
            up = mat.up.weight.float()
            down = mat.down.weight.float()
            dW = dW.float()
            self.grads.buf[id(mat.down.weight)] += up.t() @ dW
            self.grads.buf[id(mat.up.weight)] += dW @ down.t()
        else:
            self.grads.buf[id(mat.weight)] += dW

    @torch.no_grad()
    def _bwd_epilogue_body(self):
        d = self.cfg.d_model
        for li, blk in enumerate(self.model.blocks):
            a = blk.attn
            self._accum_ctx_dW(a.w_kctx, self.dW_pack[li][:d])
            self._accum_ctx_dW(a.w_vctx, self.dW_pack[li][d:2 * d])
            # gate u-half: dW_pack rows 2d:4d are grads wrt the u-columns of
            # the EFFECTIVE dense gate weight. For a factored (LoRA) gate,
            # chain them into the b/a factors; the x-half + bias grads land
            # in layer_close. Dense gate: commit straight into columns d:.
            lora = a.w_gk.weight.shape[1] != 2 * d
            if lora:
                for g, gu in ((a.w_gk, self.dW_pack[li][2 * d:3 * d]),
                              (a.w_gv, self.dW_pack[li][3 * d:4 * d])):
                    bw = g.b.weight.float()          # [d, r]
                    au = g.a.weight[:, d:].float()   # [r, d] (u-columns)
                    self.grads.buf[id(g.b.weight)] += gu.float() @ au.t()
                    self.grads.buf[id(g.a.weight)][:, d:] += bw.t() @ gu.float()
            else:
                self.grads.buf[id(a.w_gk.weight)][:, d:] += \
                    self.dW_pack[li][2 * d:3 * d]
                self.grads.buf[id(a.w_gv.weight)][:, d:] += \
                    self.dW_pack[li][3 * d:4 * d]
            self.grads.buf[id(a.norm_ctx.weight)] += self.dnormw[li]
            # ctx_mod variant branch grads (dW_pack[d:2d] is 0 for variants,
            # so w_vctx is fully supplied by dW_vctx; rows 4d:5d are the
            # folded u-driven pre-GEMM; v_gate / v_pre-xhat-half grads land
            # in layer_close via the deferred hoist chain)
            if self._fs_mod != "linear":
                self._accum_ctx_dW(a.w_vctx, self.dW_vctx[li])
                if self._fs_mod in ("mult", "mult_res"):
                    self._accum_ctx_dW(a.v_map, self.dW_pack[li][4 * d:])
                else:
                    self.grads.buf[id(a.v_pre.weight)][:, d:] += \
                        self.dW_pack[li][4 * d:]

    @torch.no_grad()
    def _emb_bwd_body(self):
        emb_w = self.model.tok_emb.weight
        with torch.enable_grad():
            emb_leaf = emb_w.detach().requires_grad_(True)
            out = torch.nn.functional.embedding(self.idx, emb_leaf)
        (g,) = _grad([out], [emb_leaf], [self.dx[0].to(out.dtype)])
        self.grads.add(emb_w, g)

    # ---- capture -------------------------------------------------------------

    def _capture(self, body, pool_slot):
        """Warmup once on a side stream, then capture into the given pool
        slot ('misc' or layer index) on that slot's dedicated capture stream
        (per-layer cuBLAS workspace isolation — see __init__ comment)."""
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            body()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        if pool_slot == "misc":
            pool, cap = self._misc_pool, self._misc_cap_stream
        else:
            pool, cap = self._pools[pool_slot], self.cap_streams[pool_slot]
        with torch.cuda.graph(g, pool=pool, stream=cap):
            body()
        if pool is None:
            if pool_slot == "misc":
                self._misc_pool = g.pool()
            else:
                self._pools[pool_slot] = g.pool()
        return g

    def _capture_all(self):
        gc.collect()   # stale eager autograd graphs poison capture (AccumulateGrad)
        t0 = time.perf_counter()
        self._g_embed = self._capture(self._embed_body, "misc")
        self._g_loss = self._capture(self._loss_body, "misc")
        self._g_prologue = self._capture(self._prologue_body, "misc")
        self._g_headbwd = self._capture(self._head_bwd_body, "misc")
        self._g_embbwd = self._capture(self._emb_bwd_body, "misc")
        if self.fused_sweep:
            self._g_bwdepi = self._capture(self._bwd_epilogue_body, "misc")
            slot = (lambda li: 0) if self.share_layer_pools else (lambda li: li)
            self._g_proj = [self._capture(
                lambda: self._proj_body(li, False), slot(li)) for li in range(self.L)]
            self._g_projb = [self._capture(
                lambda: self._proj_body(li, True), slot(li)) for li in range(self.L)]
            self._g_blockmlp = [self._capture(
                lambda: self._blockmlp_bwd_body(li), slot(li)) for li in range(self.L)]
            self._g_close = [self._capture(
                lambda: self._close_body(li), slot(li)) for li in range(self.L)]
        for sid in range(len(self.schedules)):
            n = len(self.schedules[sid])
            self._gf_cells[sid] = [[None] * n for _ in range(self.L)]
            self._gb_cells[sid] = [[None] * n for _ in range(self.L)]
            for li in range(self.L):
                # Concurrent wave replay needs pool + cuBLAS-workspace
                # isolation per layer; layer-major replay is sequential, so
                # all layers can share slot 0 — at flagship scale per-layer
                # pools are the difference between ~10 GB and ~225 GB (blocks
                # freed inside one capture are not reclaimed, so each pool
                # holds a full cell's sweep transients).
                slot_f = li if self.wave else 0
                slot_b = li if self.wave_bwd else 0
                for k in range(n):
                    self._gf_cells[sid][li][k] = self._capture(
                        lambda: self._cell_fwd_body(li, sid, k), slot_f)
                for k in range(n - 1, -1, -1):   # reverse, matching replay
                    self._gb_cells[sid][li][k] = self._capture(
                        lambda: self._cell_bwd_body(li, sid, k), slot_b)
        torch.cuda.synchronize()
        self.capture_seconds = time.perf_counter() - t0

    # ---- replay --------------------------------------------------------------

    def _run_fwd(self, sid):
        n = len(self.schedules[sid])
        if self.use_graphs:
            self._g_embed.replay()
        else:
            self._embed_body()
        if self.wave:
            origin = torch.cuda.current_stream()
            events = [[None] * n for _ in range(self.L)]
            for s in self.streams:
                s.wait_stream(origin)
            for li in range(self.L):
                st = self.streams[li]
                with torch.cuda.stream(st):
                    for k in range(n):
                        if li > 0:
                            st.wait_event(events[li - 1][k])
                        if self.use_graphs:
                            self._gf_cells[sid][li][k].replay()
                        else:
                            self._cell_fwd_body(li, sid, k)
                        ev = torch.cuda.Event()
                        ev.record(st)
                        events[li][k] = ev
            for s in self.streams:
                origin.wait_stream(s)
        else:
            for li in range(self.L):
                if self.fused_sweep:
                    if self.use_graphs:
                        self._g_proj[li].replay()
                    else:
                        self._proj_body(li, False)
                for k in range(n):
                    if self.use_graphs:
                        self._gf_cells[sid][li][k].replay()
                    else:
                        self._cell_fwd_body(li, sid, k)
        if self.use_graphs:
            self._g_loss.replay()
        else:
            self._loss_body()

    def _run_bwd(self, sid):
        n = len(self.schedules[sid])
        if self.use_graphs:
            self._g_prologue.replay()
            self._g_headbwd.replay()
        else:
            self._prologue_body()
            self._head_bwd_body()
        if self.sat_tau > 0:
            self.sat_events.zero_()
            self.sat_rows_zeroed.zero_()
            self.chain_rho_max.zero_()
        if self.wave_bwd:
            origin = torch.cuda.current_stream()
            events = [[None] * n for _ in range(self.L)]
            for s in self.streams:
                s.wait_stream(origin)
            for li in range(self.L - 1, -1, -1):
                st = self.streams[li]
                with torch.cuda.stream(st):
                    for k in range(n - 1, -1, -1):
                        if li < self.L - 1:
                            st.wait_event(events[li + 1][k])
                        if self.use_graphs:
                            self._gb_cells[sid][li][k].replay()
                        else:
                            self._cell_bwd_body(li, sid, k)
                        ev = torch.cuda.Event()
                        ev.record(st)
                        events[li][k] = ev
            for s in self.streams:
                origin.wait_stream(s)
        else:
            for li in range(self.L - 1, -1, -1):
                if self.fused_sweep:
                    if self.use_graphs:
                        self._g_projb[li].replay()
                        self._g_blockmlp[li].replay()
                    else:
                        self._proj_body(li, True)
                        self._blockmlp_bwd_body(li)
                if self.sat_tau > 0:
                    # per-row layer-entry gradient norm: the saturation
                    # reference for every cell in this layer
                    self._sat_ref = (self.dx[li + 1].float().pow(2)
                                     .flatten(1).sum(1).sqrt()
                                     .clamp_min(1e-12))
                for k in range(n - 1, -1, -1):
                    if self.use_graphs:
                        self._gb_cells[sid][li][k].replay()
                    else:
                        self._cell_bwd_body(li, sid, k)
                if self.fused_sweep:
                    if self.use_graphs:
                        self._g_close[li].replay()
                    else:
                        self._close_body(li)
        if self.use_graphs:
            self._g_embbwd.replay()
        else:
            self._emb_bwd_body()
        if self.fused_sweep:
            if self.use_graphs:
                self._g_bwdepi.replay()
            else:
                self._bwd_epilogue_body()

    # ---- public API ----------------------------------------------------------

    def _reset_c_scratch(self, sid):
        """Multi-sid engines with DIFFERENT chunk sizes: scratch written at a
        larger c leaves stale tails beyond a smaller schedule's [:c] slices,
        which the small-c backward consumes (observed: c64 -> c32 switch
        gives gn 1e6 then NaN; constant-c multi-sid — the FS1 k-ladder — is
        unaffected and this never triggers there)."""
        c_now = self.schedules[sid][0][1]
        if getattr(self, "_last_c", None) in (None, c_now):
            self._last_c = c_now
            return
        self._last_c = c_now

    def forward(self, idx, sid=0, targets=None):
        self._reset_c_scratch(sid)
        self.idx.copy_(idx)
        if targets is not None:
            self.targets.copy_(targets)
        self._run_fwd(sid)
        return self.logits if self.keep_logits else self.loss

    def backward(self, sid=0, d_logits=None):
        if d_logits is not None:
            self.d_logits.copy_(d_logits)
        self._run_bwd(sid)

    def train_step(self, idx, targets, sid=0):
        """fwd + bwd; returns the static loss scalar (clone to keep). Grads
        land in self.grads fp32 buffers; call finish_grads() before opt.step."""
        self.forward(idx, sid, targets=targets)
        self.backward(sid)
        return self.loss

    @torch.no_grad()
    def debug_scan(self):
        """Name every persistent engine tensor holding non-finite values.
        Called by the trainer when a wild-gn skip fires — identifies which
        buffer class carries blown-backward residue (three storm classes
        found by replay so far; the first two fixed by prologue hygiene)."""
        bad = []
        def chk(name, t):
            if torch.is_tensor(t) and t.is_floating_point()                     and not torch.isfinite(t).all():
                bad.append(name)
        for i, t in enumerate(self.xl):
            chk(f"xl[{i}]", t)
        for i, t in enumerate(self.dx):
            chk(f"dx[{i}]", t)
        for i in range(self.L):
            chk(f"K_caches[{i}]", self.K_caches[i])
            chk(f"V_caches[{i}]", self.V_caches[i])
            chk(f"dK[{i}]", self.dK[i])
            chk(f"dV[{i}]", self.dV[i])
        chk("grads.flat", self.grads.flat)
        chk("loss", self.loss)
        if self.fused_sweep:
            for i in range(self.L):
                chk(f"fs_ocat[{i}]", self.fs_ocat[i])
                chk(f"dW_pack[{i}]", self.dW_pack[i])
                chk(f"dnormw[{i}]", self.dnormw[i])
            for cc, scr in self.fs_scratch.items():
                for n, v in vars(scr).items():
                    if torch.is_tensor(v):
                        chk(f"fs_scratch[{cc}].{n}", v)
                    elif isinstance(v, (list, tuple)):
                        for j, x in enumerate(v):
                            chk(f"fs_scratch[{cc}].{n}[{j}]", x)
            for cc, t in self.fs_u.items():
                chk(f"fs_u[{cc}]", t)
            for cc, t in self.fs_o.items():
                chk(f"fs_o[{cc}]", t)
            for n, v in vars(self.fs_P).items():
                chk(f"fs_P.{n}", v)
        for n, p in self.model.named_parameters():
            if not torch.isfinite(p).all():
                bad.append(f"PARAM {n}")
        return bad

    def finish_grads(self, world_size=1, process_group=None,
                     bitwise_sync=True):
        """Manual DDP flat-bucket all-reduce of the grad buffers (§6), then
        populate param.grad for a normal optimizer step. bitwise_sync uses
        reduce+broadcast so every rank holds the IDENTICAL bytes — plain
        all-reduce was observed to differ across ranks at ~1e-7 on 8-GPU
        NVLink topologies, which compounds into weight desync over a long
        run (ranks never re-sync in this manual-DDP scheme)."""
        # pre-reduce local grad norm: identifies which rank originated a
        # blown backward (post-reduce gn is global and rank-identical)
        # f.norm() is a fused reduction (no full fp32 materialization); .double()
        # the scalar for the cross-bucket sum. (f.float().norm() would upcast the
        # whole bf16 trunk bucket -> a 35GB temporary at 8B.)
        self.local_gn = float(sum(f.norm().double() ** 2 for f in self.grads.flats) ** 0.5)
        if world_size > 1:
            dist = torch.distributed
            for flat in self.grads.flats:   # one flat NCCL bucket per dtype group
                if bitwise_sync:
                    dist.reduce(flat, dst=0, group=process_group)
                    flat.div_(world_size)
                    dist.broadcast(flat, src=0, group=process_group)
                else:
                    dist.all_reduce(flat, group=process_group)
                    flat.div_(world_size)
        self.grads.to_param_grads()
