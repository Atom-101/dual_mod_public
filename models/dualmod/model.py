"""LLaMA-style decoder-only LM with DualModAttention (plan.md §2, §8).

RMSNorm pre-norm, SwiGLU MLP, RoPE, no biases except the gate biases, tied
embedding/unembedding. Only the attention differs from vanilla; the refined
k'', v'' go only to the cache, never into the residual stream directly.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import DualModAttention, RMSNorm
from .config import DualModConfig
from .rope import build_rope_cache


# ---- structured gate-bias inits (plans/asym_gate.md, Design A / UGI) ----------

def _logit(u):
    return torch.log(u / (1.0 - u))


def _quasi_uniform(n, lo, hi, gen):
    """Deterministic coverage of [lo,hi] in GATE space, random slot assignment.
    linspace (not sampling) => exact mean 0.5, exact range, seed-stable."""
    u = torch.linspace(lo, hi, n)
    return u[torch.randperm(n, generator=gen)]


def _key_bias_ugi(n_heads, head_dim, gen, eps=0.04):
    """One bias per RoPE pair, copied to BOTH members (rotate_half convention:
    pair = (i, i+head_dim//2)); rotation mixes the pair so they must share a bias."""
    half = head_dim // 2
    b = torch.empty(n_heads, head_dim)
    for h in range(n_heads):
        bp = _logit(_quasi_uniform(half, eps, 1 - eps, gen))
        b[h, :half] = bp
        b[h, half:] = bp
    return b.reshape(-1)                     # [d], exact mean gate 0.5


def _value_bias_ugi(n_heads, head_dim, gen, eps=0.04):
    """Per-dim UGI; values have no RoPE constraint."""
    b = torch.empty(n_heads, head_dim)
    for h in range(n_heads):
        b[h] = _logit(_quasi_uniform(head_dim, eps, 1 - eps, gen))
    return b.reshape(-1)                     # [d], exact mean gate 0.5


def _value_bias_band(n_heads, head_dim, gen, lo=0.2, hi=0.8):
    """G-B values: UGI over a narrower band U[lo,hi] (all dims mobile, mean 0.5)."""
    b = torch.empty(n_heads, head_dim)
    for h in range(n_heads):
        b[h] = _logit(_quasi_uniform(head_dim, lo, hi, gen))
    return b.reshape(-1)


def _key_bias_committed(n_heads, head_dim, gen, b_commit=3.0, n_frontier=6, frontier_hi=1.5):
    """G-B keys: per head, RoPE pairs split into raw-committed (+b), ctx-committed
    (−b), and a mobile frontier. Returns (bias[d], cls[d]) with cls in {0 raw,
    1 ctx, 2 frontier}. Exact mean gate 0.5 by symmetry."""
    half = head_dim // 2                          # n_pairs
    n_commit = (half - n_frontier) // 2
    assert 2 * n_commit + n_frontier == half, "frontier count must keep pairs even"
    b = torch.empty(n_heads, head_dim)
    cls = torch.empty(n_heads, head_dim, dtype=torch.long)
    for h in range(n_heads):
        bp = torch.cat([torch.full((n_commit,), b_commit),
                        torch.full((n_commit,), -b_commit),
                        torch.linspace(-frontier_hi, frontier_hi, n_frontier)])
        cl = torch.cat([torch.zeros(n_commit), torch.ones(n_commit),
                        torch.full((n_frontier,), 2.0)]).long()
        perm = torch.randperm(half, generator=gen)
        bp, cl = bp[perm], cl[perm]
        b[h, :half], b[h, half:] = bp, bp              # RoPE pair shares bias + class
        cls[h, :half], cls[h, half:] = cl, cl
    return b.reshape(-1), cls.reshape(-1)


def _init_lora_ctx(mod, target_std):
    """Init a LoRALinear (ctx_rank>0 ctx operator) so the MATERIALIZED weight
    W = up@down has marginal variance ~ target_std^2, comparable to the dense
    N(0,target_std) init it replaces: down,up ~ N(0, sqrt(target_std)/r**0.25)
    gives Var(W_ij) = r * s^4 = target_std^2. up.bias (v_gate) zeroed."""
    r = mod.down.weight.shape[0]
    s = (target_std ** 0.5) / (r ** 0.25)
    nn.init.normal_(mod.down.weight, mean=0.0, std=s)
    nn.init.normal_(mod.up.weight, mean=0.0, std=s)
    if mod.up.bias is not None:
        nn.init.zeros_(mod.up.bias)


def _init_lora_zero(mod):
    """Materialized W = up@down = 0 via up=0 (v_map's mult_res residual-identity
    init); down keeps its generic init so gradient can lift the branch."""
    nn.init.zeros_(mod.up.weight)
    if mod.up.bias is not None:
        nn.init.zeros_(mod.up.bias)


class SwiGLU(nn.Module):
    def __init__(self, cfg: DualModConfig):
        super().__init__()
        d, hidden = cfg.d_model, cfg.mlp_hidden
        self.w1 = nn.Linear(d, hidden, bias=False)   # gate
        self.w3 = nn.Linear(d, hidden, bias=False)   # up
        self.w2 = nn.Linear(hidden, d, bias=False)   # down

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: DualModConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = DualModAttention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin, collect=None):
        x = x + self.attn(self.attn_norm(x), cos, sin, collect=collect)
        x = x + self.mlp(self.mlp_norm(x))
        return x

    def decode_step(self, x_j, pos, cache, cos, sin):
        x_j = x_j + self.attn.decode_step(self.attn_norm(x_j), pos, cache, cos, sin)
        x_j = x_j + self.mlp(self.mlp_norm(x_j))
        return x_j


class DualModLM(nn.Module):
    def __init__(self, cfg: DualModConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        cos, sin = build_rope_cache(cfg.max_seq_len, cfg.head_dim, cfg.rope_theta)
        if not getattr(cfg, "use_rope", True):          # NoPE: cos=1, sin=0 -> apply_rope is identity
            cos, sin = torch.ones_like(cos), torch.zeros_like(sin)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self._init_weights()

    def _init_weights(self):
        cfg = self.cfg
        std = 0.02
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=std)
        resid_std = std / math.sqrt(2 * cfg.n_layers)
        for li, blk in enumerate(self.blocks):
            nn.init.normal_(blk.attn.wo.weight, mean=0.0, std=resid_std)
            nn.init.normal_(blk.mlp.w2.weight, mean=0.0, std=resid_std)
            a = blk.attn
            if hasattr(a, "w_kctx"):
                if cfg.ctx_rank > 0:
                    _init_lora_ctx(a.w_kctx, cfg.ctx_proj_init_std)
                else:
                    nn.init.normal_(a.w_kctx.weight, mean=0.0, std=cfg.ctx_proj_init_std)
                if cfg.gate_rank > 0:
                    nn.init.normal_(a.w_gk.a.weight, mean=0.0, std=0.02)
                    nn.init.normal_(a.w_gk.b.weight, mean=0.0, std=0.02)
                else:
                    nn.init.normal_(a.w_gk.weight, mean=0.0, std=std)
                # bias +4 -> sigma(4)~0.982 near-vanilla; "ugi"/"committed" spread at mean 0.5
                if cfg.gate_bias_init_k == "committed":
                    self._init_key_committed(a, li)
                else:
                    self._init_gate_bias(a.w_gk.bias, cfg.gate_bias_init_k, path=0, li=li)
            if hasattr(a, "w_vctx"):
                if cfg.ctx_rank > 0:
                    _init_lora_ctx(a.w_vctx, cfg.ctx_proj_init_std)
                else:
                    nn.init.normal_(a.w_vctx.weight, mean=0.0, std=cfg.ctx_proj_init_std)
                if cfg.gate_rank > 0:
                    nn.init.normal_(a.w_gv.a.weight, mean=0.0, std=0.02)
                    nn.init.normal_(a.w_gv.b.weight, mean=0.0, std=0.02)
                else:
                    nn.init.normal_(a.w_gv.weight, mean=0.0, std=std)
                if getattr(cfg, "gate_type_v", "convex") == "independent":
                    # both bias halves 0 -> g_raw = g_ctx = 0.5 (convex
                    # midpoint scale, sum=1); the structured convex inits
                    # (scalar +4 / ugi / band) assume a d-wide convex gate
                    nn.init.zeros_(a.w_gv.bias)
                else:
                    self._init_gate_bias(a.w_gv.bias, cfg.gate_bias_init_v, path=1, li=li)
            # ctx_mod residual pre layers: zero (exact identity at init) —
            # must run AFTER the generic init pass or it gets overwritten.
            # puremlp/mult keep HEALTHY pre inits (the small-init lesson
            # applies to the OUTPUT layer w_vctx only, which stays 0.006)
            if cfg.ctx_mod in ("vmlp", "fusion", "kmlp"):
                for pre in ("v_pre", "k_pre"):
                    if hasattr(a, pre):
                        nn.init.zeros_(getattr(a, pre).weight)
            elif cfg.ctx_mod in ("mult", "mult_res"):
                if cfg.ctx_rank > 0:
                    # factored v_gate: materialized W ~ std (matches the dense
                    # generic init + zero bias). v_map: mult_res => W=0 (tanh(0)
                    # identity carry), mult => ~std (matches dense generic).
                    _init_lora_ctx(a.v_gate, std)
                    if cfg.ctx_mod == "mult_res":
                        _init_lora_zero(a.v_map)
                    else:
                        _init_lora_ctx(a.v_map, std)
                elif cfg.ctx_mod == "mult_res":
                    nn.init.zeros_(a.v_map.weight)   # tanh(0)=0 -> identity
            elif cfg.ctx_mod == "linpoint":
                d_ = cfg.d_model
                eye = torch.eye(d_)
                with torch.no_grad():
                    a.v_pre.weight.copy_(torch.cat([eye, -eye], dim=0))
                    a.v_comb.weight.copy_(torch.cat([eye, -eye], dim=1))
            elif cfg.ctx_mod.endswith("_hw"):
                # healthy-W2 variants: the 0.006 output guard predates the
                # nonlin era and is the common factor in sandwich failures —
                # re-init w_vctx at standard scale
                nn.init.normal_(a.w_vctx.weight, mean=0.0, std=std)

    def _gate_gen(self, path, li):
        return torch.Generator().manual_seed(self.cfg.gate_init_seed * 1000 + 10 * li + path)

    def _init_gate_bias(self, bias, mode, path, li):
        cfg = self.cfg
        if mode == "scalar":
            nn.init.constant_(bias, cfg.gate_bias_init)
            return
        assert bias.numel() == cfg.d_model, "structured init assumes convex_sigmoid gate"
        gen = self._gate_gen(path, li)
        if mode == "ugi":
            fn = _key_bias_ugi if path == 0 else _value_bias_ugi
            b = fn(cfg.n_heads, cfg.head_dim, gen, cfg.ugi_eps)
        elif mode == "band":
            b = _value_bias_band(cfg.n_heads, cfg.head_dim, gen, cfg.value_band_lo, cfg.value_band_hi)
        else:
            raise NotImplementedError(mode)
        with torch.no_grad():
            bias.copy_(b.to(bias.dtype))

    def _init_key_committed(self, a, li):
        """G-B keys: committed ±3 / frontier bias + reserved-address W_Kctx init (§3.1):
        ctx-committed (reserved) dims are born ~empty, so give their W_Kctx rows real
        content (std 0.02) instead of the tiny 0.006, so the addresses aren't blank."""
        cfg = self.cfg
        gen = self._gate_gen(0, li)
        b, cls = _key_bias_committed(cfg.n_heads, cfg.head_dim, gen,
                                     cfg.key_commit_bias, cfg.key_frontier_pairs, cfg.key_frontier_hi)
        with torch.no_grad():
            a.w_gk.bias.copy_(b.to(a.w_gk.bias.dtype))
            reserved = (cls == 1).nonzero(as_tuple=True)[0]
            a.w_kctx.weight[reserved, :] = torch.randn(len(reserved), cfg.d_model,
                                                       generator=gen) * cfg.ctx_reserved_init_std

    # ---- teacher-forced forward ------------------------------------------

    def forward(self, idx=None, targets=None, inputs_embeds=None, collect=None):
        """collect: optional list of per-layer dicts for §10 telemetry."""
        x = self.tok_emb(idx) if inputs_embeds is None else inputs_embeds
        T = x.shape[1]
        assert T <= self.cfg.max_seq_len
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        for i, blk in enumerate(self.blocks):
            x = blk(x, cos, sin, collect=collect[i] if collect is not None else None)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]),
                                   targets.reshape(-1), ignore_index=-1)
        return logits, loss

    # ---- incremental decode (shares the per-position step with the scan) --

    def decode_init(self):
        return [{"K": None, "V": None} for _ in self.blocks]

    @torch.no_grad()
    def decode_step(self, idx_j, pos, caches):
        """idx_j: [B, 1] token ids at absolute position pos. Returns logits [B, 1, V]."""
        x = self.tok_emb(idx_j)
        for blk, cache in zip(self.blocks, caches):
            x = blk.decode_step(x, pos, cache, self.rope_cos, self.rope_sin)
        return self.lm_head(self.norm_f(x))

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        caches = self.decode_init()
        logits = None
        for t in range(idx.shape[1]):
            logits = self.decode_step(idx[:, t:t + 1], t, caches)
        out = idx
        for _ in range(max_new_tokens):
            pos = out.shape[1] - 1
            if pos + 1 >= self.cfg.max_seq_len:
                break
            lg = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                kth = torch.topk(lg, top_k, dim=-1).values[:, -1:]
                lg = lg.masked_fill(lg < kth, float("-inf"))
            nxt = torch.multinomial(torch.softmax(lg.float(), dim=-1), 1)
            out = torch.cat([out, nxt], dim=1)
            logits = self.decode_step(nxt, pos + 1, caches)
        return out

    # ---- accounting (§8) ---------------------------------------------------

    def num_params(self, non_embedding=True):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding and not self.cfg.tie_embeddings:
            n -= self.lm_head.weight.numel()
        if non_embedding:
            n -= self.tok_emb.weight.numel()
        return n

    def param_breakdown(self):
        base, dual = 0, 0
        dual_names = ("w_kctx", "w_vctx", "w_gk", "w_gv", "norm_ctx")
        for name, p in self.named_parameters():
            if any(f".{d}." in name for d in dual_names):
                dual += p.numel()
            else:
                base += p.numel()
        return {"base": base, "dualmod_extra": dual, "total": base + dual}
