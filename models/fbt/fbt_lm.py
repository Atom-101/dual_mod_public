"""Full-Bandwidth Transformer (Wang et al. 2026, arXiv:2608.08888) — EXACT sequential arm.

The previous position's top-layer hidden state is fused with the current token embedding and fed
back as the layer-0 input (their Eq. 4):

    u_1 = e_1,          u_t = RMSNorm( W^U h_{t-1}  ⊙  σ(W^G RMSNorm(e_t)) )   for t >= 2      (their Eq. 4 + Fig. 9)

The hidden state occupies the value pathway and the token embedding enters only as a gate (the
paper's deliberate asymmetry; additive fusion would leave a shortcut that suppresses the state
pathway).  The paper trains with a scheduled multi-pass parallel approximation; here we train the
recurrence EXACTLY: one teacher-forced sequential pass with a KV cache, full backprop through every
position (no detach, no truncation).  Jitter ε ~ Uniform[-0.02, 0.02]^D is added to the fused input u_t in
training (their Eq. 13).  h_{t-1} is the top-layer vector the LM head reads (post final RMSNorm; tied embeddings,
as theirs).  W^U, W^G square, no bias, N(0, 0.02) init (the paper specifies none).  Stack = the harness's vanilla
Transformer++ block.  v1 (identity-init W^U, gate on raw e, jitter on h) is archived in analysis/fbt/v1_identity_init/.
"""
import torch
import torch.nn as nn

from models.dualmod.config import DualModConfig
from models.dualmod.model import DualModLM, RMSNorm


class FBTLM(nn.Module):
    def __init__(self, cfg: DualModConfig, jitter: float = 0.02):
        super().__init__()
        assert cfg.attn_mode == "vanilla"
        self.core = DualModLM(cfg)
        d = cfg.d_model
        self.w_u = nn.Linear(d, d, bias=False)      # state (value) pathway
        self.w_g = nn.Linear(d, d, bias=False)      # token gate
        self.e_norm = RMSNorm(d)                    # input_rmsnorm on e before the gate (their Fig. 9)
        self.in_norm = RMSNorm(d)                   # RMSNorm on the fused input before layer 0
        self.jitter = jitter
        nn.init.normal_(self.w_u.weight, std=0.02); nn.init.normal_(self.w_g.weight, std=0.02)

    def forward(self, idx, K=None):
        """K=None/0: exact sequential recurrence (teacher-forced, KV cache). K=P>=1: the paper's parallel multi-pass
        approximation (their Eqs. 9-11): pass 1 = plain stack on e; pass k re-runs the full stack in parallel on
        u_t = RMSNorm(W^U h^{(k-1)}_{t-1} * sigmoid(W^G RMSNorm(e_t))), u_1 = e_1, gradients through all passes."""
        core = self.core
        B, T = idx.shape
        assert T <= core.cfg.max_seq_len
        e = core.tok_emb(idx)                       # [B, T, d]
        if K:
            cos, sin = core.rope_cos[:T], core.rope_sin[:T]
            h = None
            for p in range(int(K)):
                if h is None:
                    u = e
                else:
                    hp = h[:, :-1]
                    if self.training and self.jitter > 0:
                        hp = hp + (torch.rand_like(hp) * 2 - 1) * self.jitter      # jitter on the carried state (their Eq. 13)
                    uf = self.in_norm(self.w_u(hp) * torch.sigmoid(self.w_g(self.e_norm(e[:, 1:]))))
                    u = torch.cat([e[:, :1], uf], dim=1)
                x = u
                for blk in core.blocks:
                    x = blk(x, cos, sin)
                h = core.norm_f(x)
            return (core.lm_head(h),)
        caches = core.decode_init()
        h_prev, outs = None, []
        for t in range(T):
            e_t = e[:, t:t + 1]
            if h_prev is None:
                u = e_t                                                   # u_1 = e_1 (their Eq. 8)
            else:
                u = self.w_u(h_prev) * torch.sigmoid(self.w_g(self.e_norm(e_t)))
                if self.training and self.jitter > 0:
                    u = u + (torch.rand_like(u) * 2 - 1) * self.jitter   # ε on the fused input (their Eq. 13)
                u = self.in_norm(u)
            x = u
            for blk, cache in zip(core.blocks, caches):
                x = blk.decode_step(x, t, cache, core.rope_cos, core.rope_sin)
            h = core.norm_f(x)                       # top-layer state = the vector the head reads
            outs.append(core.lm_head(h))
            h_prev = h
        return (torch.cat(outs, dim=1),)
