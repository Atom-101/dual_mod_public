"""CUDA-graph greedy decode for DualModLM: the EXACT sequential scan (same math as
model.decode_step) with a static preallocated per-layer cache and a device-side
position counter, so one captured graph = one full decode step (18 layers) and the
CPU issues one graph launch per token instead of ~400 kernel launches.

Semantics (identical to attention.decode_step): slot `pos` of the K/V buffers holds
the RAW rotated self key / raw value during the read (masked softmax over slots
<= pos, fp32), then is overwritten with the REFINED rotated key / refined value.
Validate against eager with lm/eval/validate_graph_decode.py before scoring anything.
"""
import torch
from models.dualmod.rope import apply_rope


class GraphDecoder:
    def __init__(self, model, B, P_max, dtype=torch.bfloat16):
        self.m = model
        self.dev = next(model.parameters()).device
        assert P_max <= model.cfg.max_seq_len, (P_max, model.cfg.max_seq_len)
        a0 = model.blocks[0].attn
        L, h, dh = len(model.blocks), a0.n_heads, a0.head_dim
        self.B, self.P, self.dt = B, P_max, dtype
        self.K = [torch.zeros(B, h, P_max, dh, device=self.dev, dtype=dtype) for _ in range(L)]
        self.V = [torch.zeros(B, h, P_max, dh, device=self.dev, dtype=dtype) for _ in range(L)]
        self.tok = torch.zeros(B, 1, dtype=torch.long, device=self.dev)      # input token (per step)
        self.pos = torch.zeros(1, dtype=torch.long, device=self.dev)         # absolute position
        self.ar = torch.arange(P_max, device=self.dev)
        self.logits = torch.zeros(B, 1, model.cfg.vocab_size, device=self.dev)   # fp32 out
        self.next = torch.zeros(B, 1, dtype=torch.long, device=self.dev)     # argmax out
        self.graph = None

    # ---- one layer's attention step on the static cache -------------------------
    def _attn(self, attn, xhat, l, cos_p, sin_p):
        q = attn._split(attn.wq(xhat))
        k = attn._split(attn.wk(xhat))
        v = attn._split(attn.project_v(xhat))
        q_r = apply_rope(q, cos_p, sin_p)
        k_r = apply_rope(k, cos_p, sin_p)
        Kb, Vb = self.K[l], self.V[l]
        Kb.index_copy_(2, self.pos, k_r.to(Kb.dtype))       # raw self key/value in slot pos
        Vb.index_copy_(2, self.pos, v.to(Vb.dtype))
        s = (q_r @ Kb.transpose(-1, -2)) * attn.scale         # [B,h,1,P]
        s = s.masked_fill(self.ar > self.pos, float("-inf"))  # only slots <= pos are live
        a = torch.softmax(s.float(), dim=-1)
        o = a.to(Vb.dtype) @ Vb                               # [B,h,1,dh]
        o_cat = attn._merge(o)
        kpp, vpp = attn.refine_kv(xhat, o_cat, attn._merge(k), attn._merge(v))
        kpp_r = apply_rope(attn._split(kpp), cos_p, sin_p)   # rotate blended key at write
        Kb.index_copy_(2, self.pos, kpp_r.to(Kb.dtype))       # refined pair replaces the raw slot
        Vb.index_copy_(2, self.pos, attn._split(vpp).to(Vb.dtype))
        return attn.wo(o_cat)

    def _step(self):
        m = self.m
        cos_p = m.rope_cos.index_select(0, self.pos)
        sin_p = m.rope_sin.index_select(0, self.pos)
        x = m.tok_emb(self.tok)
        for l, blk in enumerate(m.blocks):
            x = x + self._attn(blk.attn, blk.attn_norm(x), l, cos_p, sin_p)
            x = x + blk.mlp(blk.mlp_norm(x))
        lg = m.lm_head(m.norm_f(x))
        self.logits.copy_(lg.float())
        self.next.copy_(lg.argmax(-1))
        self.pos += 1

    def _run_step(self):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            self._step()

    def capture(self):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):                                # warmup (allocator, autocast, kernels)
                self._run_step()
        torch.cuda.current_stream().wait_stream(s)
        self.reset()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._run_step()
        self.reset()
        self.graph = g
        return self

    def reset(self):
        self.pos.zero_()
        for Kb, Vb in zip(self.K, self.V):
            Kb.zero_(); Vb.zero_()

    def step(self):
        if self.graph is not None:
            self.graph.replay()
        else:
            self._run_step()

    # ---- greedy generation -------------------------------------------------------
    @torch.no_grad()
    def generate(self, prompt, max_new, eos_id=None, stop_fn=None, check_every=16):
        """prompt: [B, L] long on device (all rows same L; left-pad upstream).
        Returns gen [B, max_new] long (CPU) and n_steps actually run. stop_fn(gen_cpu[:, :k])
        -> bool tensor [B] 'row finished' (eos/until); generation stops when all rows are."""
        B, L = prompt.shape
        assert B == self.B and L + max_new <= self.P, (B, L, max_new, self.P)
        self.reset()
        for p in range(L):                                    # prefill: one replay per prompt token
            self.tok.copy_(prompt[:, p:p + 1])
            self.step()
        gen = torch.zeros(B, max_new, dtype=torch.long, device=self.dev)
        done = torch.zeros(B, dtype=torch.bool)
        k = 0
        while k < max_new:
            gen[:, k].copy_(self.next[:, 0])                  # token predicted at the previous step
            k += 1
            if k % check_every == 0 or k == max_new:
                g_cpu = gen[:, :k].cpu()
                if eos_id is not None:
                    done |= (g_cpu == eos_id).any(1)
                if stop_fn is not None:
                    done |= stop_fn(g_cpu)
                if bool(done.all()):
                    break
            if k < max_new:
                self.tok.copy_(self.next)
                self.step()
        return gen[:, :k].cpu(), k
