"""Looped / Universal-Transformer baseline (Bansal et al. 2022 "end-to-end algorithm synthesis" recipe):
a small stack of `n_unique` vanilla-attention Blocks applied K times, weight-tied, with INPUT INJECTION
(the token embedding is re-added at every iteration) and a learned STEP EMBEDDING, then norm + head.
K is a runtime argument: train with stochastic K in [k_min, k_max] (progressive-loss style) and evaluate
at any K — the "iteration dial" a recurrence does not have. Uses models.dualmod.model.Block with
attn_mode='vanilla' so the per-block math matches our vanilla340 / TPP baselines exactly."""
import torch, torch.nn as nn, torch.nn.functional as F
from models.dualmod.config import DualModConfig
from models.dualmod.model import Block, RMSNorm


class LoopedLM(nn.Module):
    def __init__(self, cfg: DualModConfig, n_unique: int = 2, k_max: int = 64, inject: bool = True,
                 huginn: bool = False, bp_iters: int = 0, emb_scale: bool = False):
        """huginn: Geiping et al. 2025 recurrent-depth recipe — prelude embeds the tokens once, the loop state starts
        from Gaussian noise, the embedding is injected every iteration, NO step embedding, and backprop is truncated to
        the last `bp_iters` iterations (they use 8). K is sampled per step by the harness (1 + Poisson(log-normal))."""
        super().__init__()
        assert cfg.attn_mode == "vanilla"
        self.cfg, self.n_unique, self.k_max, self.inject = cfg, n_unique, k_max, inject
        self.huginn, self.bp_iters = huginn, bp_iters
        self.emb_scale = float(cfg.d_model) ** 0.5 if (huginn and emb_scale) else 1.0   # gamma = sqrt(h): their embedding scale (Sec. 3.2); 1.0 = the pre-2026-09-26 runs
        if huginn:   # Geiping et al. 2025 §3.2: adapter A: R^{2h}->R^h on [s, e]; sandwich RMSNorm in the recurrent block; coda block
            d = cfg.d_model
            self.adapter = nn.Linear(2 * d, d, bias=False)
            self.post_norms = nn.ModuleList(nn.ModuleList([RMSNorm(d), RMSNorm(d)]) for _ in range(n_unique))
            self.coda = Block(cfg)
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pre = Block(cfg)                                  # one non-looped encoder block (Bansal: "projection")
        self.loop = nn.ModuleList(Block(cfg) for _ in range(n_unique))
        self.step_emb = nn.Embedding(k_max + 1, cfg.d_model)
        nn.init.zeros_(self.step_emb.weight)
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        from models.dualmod.rope import build_rope_cache as rope_cache
        cos, sin = rope_cache(cfg.max_seq_len, cfg.head_dim, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self._init_weights()

    def _init_weights(self):
        """Same scheme as DualModLM._init_weights (std 0.02; residual projections scaled by the
        UNIQUE depth, not K — weight tying means K iterations share the same matrices)."""
        import math
        std = 0.02
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=std)
        resid_std = std / math.sqrt(2 * (1 + self.n_unique))
        for blk in [self.pre] + list(self.loop):
            for name in ("wo",):
                if hasattr(blk.attn, name):
                    nn.init.normal_(getattr(blk.attn, name).weight, mean=0.0, std=resid_std)
            if hasattr(blk.mlp, "w2"):
                nn.init.normal_(blk.mlp.w2.weight, mean=0.0, std=resid_std)
        nn.init.zeros_(self.step_emb.weight)

    def forward(self, idx, targets=None, K=None, return_all=False):
        K = self.k_max if K is None else K
        T = idx.shape[1]
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        e = self.tok_emb(idx)
        if self.huginn:
            e = self.pre(e * self.emb_scale, cos, sin)     # gamma*E(x) into the prelude; prelude output is the injected embedding
            x = torch.empty_like(e)                        # s_0 ~ truncated normal, variance 2/5 (their §4.1)
            nn.init.trunc_normal_(x, std=0.4 ** 0.5, a=-2 * 0.4 ** 0.5, b=2 * 0.4 ** 0.5)
        else:
            x = self.pre(e, cos, sin)
        outs = []
        for k in range(K):
            if self.huginn and self.bp_iters and self.training and k < K - self.bp_iters:
                x = x.detach()                              # truncated backprop through the last bp_iters iterations
            if self.huginn:
                x = self.adapter(torch.cat([x, e], dim=-1))          # R(e, s): adapter on the concatenation
                for blk, (n2, n4) in zip(self.loop, self.post_norms):  # sandwich norm: n2(x + Attn(n1(x))), n4(x + MLP(n3(x)))
                    x = n2(x + blk.attn(blk.attn_norm(x), cos, sin))
                    x = n4(x + blk.mlp(blk.mlp_norm(x)))
            else:
                if self.inject:
                    x = x + e
                x = x + self.step_emb.weight[min(k, self.k_max)]
                for blk in self.loop:
                    x = blk(x, cos, sin)
            if return_all:
                outs.append(x)
        if self.huginn:
            x = self.coda(x, cos, sin)
        logits = self.lm_head(self.norm_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), targets.reshape(-1))
        if return_all:
            return logits, loss, outs
        return logits, loss
