"""lm-eval-harness adapter for DualModLM checkpoints (flagship_run.md §6).

DualModEval (eager): likelihoods via the eager sequential scan (attn_mode="sequential"),
generation via model.decode_init/decode_step. Wave3DEval (engine): likelihoods AND generation
run the wave3d training operator at --engine_K (generation = one full engine forward per
emitted token, no KV cache); K=24 is the trained operator, K=64 the fixed point (exact by
nilpotency at C=64). Generation is K-independent in practice: DM-4K JRT recall at K=24 vs K=64
(2026-09-20) = SWDE 58.2/58.4, FDA 70.2/74.0, SQuAD 49.2/48.9, TQA 66.5/65.3, NQ 30.2/30.1, DROP 23.8/24.0.

The eager scan is latency-bound in T and ~free in batch rows, so likelihood
requests are packed into wide row-batches (default 64) bucketed by length.
Right-padding is safe: causal attention means positions after the scored
span cannot affect it.

Usage:
  python lm/eval/run_lm_eval.py --ckpt out/FS-DM-s1337/snap_009536.pt \
      --tasks piqa,hellaswag,winogrande,arc_easy,arc_challenge,lambada_openai,wikitext
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.distributed as dist
import torch.nn.functional as F

from lm_eval.api.model import LM
from lm_eval.api.instance import Instance

from models.dualmod.config import DualModConfig
from models.dualmod.model import DualModLM



_GEN_IMPL_VER = 7   # bump when generation-path semantics change (invalidates GEN_CACHE_DIR entries)


def _bos_prefix(bos):
    """GEN_NO_BOS=1: no BOS at position 0. The hero data = [BOS]+doc packed into 4K windows at
    arbitrary offsets => BOS-at-position-0 is ~1/1000 in training; a mid-text start is the norm."""
    return [] if os.environ.get("GEN_NO_BOS") == "1" else [bos]


def _gen_ctx(ctx):
    """Strip TRAILING SPACES from a generation prompt (GEN_KEEP_TRAILING_SPACE=1 disables).
    SentencePiece/Llama: a prompt ending in ' ' tokenizes to a lone '▁' token, which in training
    text precedes only DIGITS (words carry their own '▁'), so the model answers with a number:
    Based doc-grounded TQA/NQ/DROP prompts all end in ' ' and 83-98% of generations began with a
    digit vs 3-15% of targets (2026-09-08 dump). Stripping lets the model emit '▁Word' itself."""
    if os.environ.get("GEN_KEEP_TRAILING_SPACE") == "1":
        return ctx
    return ctx.rstrip(" ") if ctx.endswith(" ") else ctx


def _trunc_prompt(toks, cap):
    """GEN_TRUNC=head: an over-long prompt keeps the DOCUMENT HEAD + its last 128 tokens (the
    question/key line) instead of the last `cap` tokens (default 'tail' = lm-eval convention).
    Based/GDN-2 recall docs "truncated to 2K" carry their answers near the START (FDA: median
    doc 2428 tok, answer at median tok 218 => tail-truncation cuts 70% of FDA answers;
    2026-09-08 audit)."""
    if len(toks) <= cap or os.environ.get("GEN_TRUNC", "tail") != "head":
        return toks
    keep_tail = min(128, cap // 4)
    return toks[:cap - keep_tail] + toks[-keep_tail:]


class DualModEval(LM):
    def __init__(self, ckpt, batch_rows=64, device="cuda",
                 tok_dir="data/llama2_tok", max_length=2048, ctx_len=None):
        """ctx_len: context extension for length-extrapolation evals (RULER):
        rebuilds RoPE tables at the longer length (raw extrapolation, no
        scaling — the plan's protocol) and lifts the T<=max_seq_len assert.
        GDN has no positional tables; only max_length changes."""
        super().__init__()
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(tok_dir)
        ck = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)   # 32 ranks x 16GB: mmap, not RAM
        self.arch = os.environ.get("EVAL_ARCH") or ck.get("arch", "dualmod")
        self.ckpt_engine_kwargs = ck.get("engine_kwargs")   # raw_diag / local_window (operator identity)
        if ctx_len:
            max_length = ctx_len
        if self.arch == "gdn2lit":        # official GDN-2 lit_gpt layout (3rd-party FineWeb-Edu ckpts); needs the lit_gpt env-gdn2
            from lm.eval import gdn2lit_shim  # noqa: F401  (flash_attn shim + sys.path)
            from lit_gpt.config import Config as _LitCfg
            from lit_gpt.model import GPT as _LitGPT
            cfg = _LitCfg.from_name(os.environ.get("GDN2LIT_CONFIG", "gdn2_1.3B"))
            assert cfg.gdn2_per_layer == 1, "gdn2lit path supports pure-recurrent GDN-2 only (no attention layers)"
            st = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
            with torch.device(device):
                self.model = _LitGPT(cfg)
            self.model.load_state_dict({k.replace("_forward_module.", "").replace("module.", ""): v for k, v in st.items()}, strict=True)
            self.model = self.model.to(device).bfloat16().eval()
            self._lit_max_seq = max(cfg.block_size, max_length)
        elif self.arch == "gdn":            # fla checkpoint — needs flash-linear-attention
            from fla.models import GatedDeltaNetConfig, GatedDeltaNetForCausalLM
            cfg = GatedDeltaNetConfig(**{k: v for k, v in ck["model_cfg"].items()
                                         if k not in ("model_type", "architectures")})
            self.model = GatedDeltaNetForCausalLM(cfg).to(device).eval()
            self.model.load_state_dict(ck["model"])
        else:
            cfg = DualModConfig(**ck["model_cfg"])
            if cfg.attn_mode != "vanilla":  # E-controls stay on the SDPA path
                cfg.attn_mode = "sequential"
            if ctx_len:
                cfg.max_seq_len = ctx_len   # RoPE tables extend at build
            max_length = min(max_length, cfg.max_seq_len)
            self.model = DualModLM(cfg).to(device).eval()
            self.model.load_state_dict(ck["model"])   # RoPE buffers are
            # persistent=False, so extended tables never clash with the ckpt
        self.dev = device
        self.batch_rows = batch_rows
        self.max_length = max_length
        self.bos = self.tok.bos_token_id
        # ---- data-parallel over ranks (torchrun): lm-eval 0.4.x shards every task's
        # requests by (rank, world_size) and gathers samples/metrics on rank 0 through
        # these hooks -> one job spans all GPUs (2026-09-08; before: 1 GPU per job).
        if dist.is_available() and dist.is_initialized():
            self._rank = dist.get_rank()
            self._world_size = dist.get_world_size()

        # Token-precise generation prefix (attention-dilution discrimination test,
        # retrieval_protocol_investigation.md §8). Inserted right AFTER bos, before ctx.
        #   EVAL_PREFIX_KIND = space | random | bos   ·   EVAL_PREFIX_NTOK = N
        # Resumable generation cache (EVAL_GEN_CACHE=<jsonl path>): every finished generation is
        # appended as {"k": md5(prompt|until|max_new|prefix), "g": text}; on start the file is
        # loaded and cached requests are served without running the model. Lets a multi-day
        # generation eval (NIAH at exact length) survive reboots and be re-scored from a partial
        # cache: EVAL_CACHE_ONLY=1 skips uncached requests (returned as "<UNCACHED>", so the TEXT
        # dump can be scored over cached docs only).
        self._gcache_path = os.environ.get("EVAL_GEN_CACHE", "")
        self._gcache_only = os.environ.get("EVAL_CACHE_ONLY", "0") == "1"
        self._gcache = {}
        if self._gcache_path and os.path.exists(self._gcache_path):
            import json as _json
            with open(self._gcache_path) as f:
                for line in f:
                    try:
                        d = _json.loads(line); self._gcache[d["k"]] = d["g"]
                    except Exception:
                        pass
            print(f"[gen-cache] loaded {len(self._gcache)} generations from {self._gcache_path}", flush=True)
        self._pfx_kind = os.environ.get("EVAL_PREFIX_KIND", "")
        self._pfx_n = int(os.environ.get("EVAL_PREFIX_NTOK", "0"))
        self.pfx_ids = []
        if self._pfx_kind and self._pfx_n > 0:
            if self._pfx_kind == "bos":
                self.pfx_ids = [self.bos] * self._pfx_n
            elif self._pfx_kind == "space":
                sp = self.tok(" ", add_special_tokens=False)["input_ids"]
                self.pfx_ids = (sp * self._pfx_n)[:self._pfx_n] if sp else []
            elif self._pfx_kind in ("text", "prose"):
                # text: in-distribution SlimPajama val stream (happens to be C++ code);
                # prose: a fixed English paragraph. Both = "another document precedes": first N tokens of the SlimPajama
                # val stream (after its leading BOS), i.e. "some other document
                # precedes the passage" — the packed-training majority regime.
                ids = None
                if self._pfx_kind == "text":
                  try:
                      d = torch.load(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "slimpj627_val.pt"),
                          map_location="cpu", weights_only=False)
                      t = d if torch.is_tensor(d) else d.get("tokens")
                      t = t.flatten().tolist()
                      t = t[1:] if t and t[0] == self.bos else t
                      ids = [x for x in t if x != self.bos][:self._pfx_n]
                  except Exception:
                    ids = None
                if not ids:
                    para = ("The village sits at the mouth of a shallow river valley, where the "
                            "old mill road bends toward the coast. For most of the nineteenth "
                            "century its economy depended on wool, and the long stone sheds "
                            "along the quay were built to store fleeces before shipment. A "
                            "railway branch arrived in 1871, and the population roughly doubled "
                            "over the following two decades as quarrying expanded on the ridge.")
                    ids = (self.tok(para, add_special_tokens=False)["input_ids"] * 4)[:self._pfx_n]
                self.pfx_ids = ids
            elif self._pfx_kind == "random":
                import random as _r; _r.seed(0)
                lo, hi = 100, min(30000, self.tok.vocab_size - 1)
                self.pfx_ids = [_r.randint(lo, hi) for _ in range(self._pfx_n)]

    # ---- lm-eval distributed hooks ------------------------------------------

    @property
    def device(self):
        return torch.device(self.dev if self.dev != "cuda" else f"cuda:{torch.cuda.current_device()}")

    def all_gather(self, tensor):
        if self._world_size == 1:
            return tensor
        t = torch.as_tensor(tensor).to(self.device)
        outs = [torch.zeros_like(t) for _ in range(self._world_size)]
        dist.all_gather(outs, t)
        return torch.stack(outs, 0) if t.dim() == 0 else torch.cat(outs, 0)

    def gather_object(self, obj, dst=0):
        # all_gather_object, NOT dist.gather_object: the NCCL point-to-point gather died with
        # "NCCL Error 2: unhandled system error" on one node after 50 min of RULER generation
        # (2026-09-08) and the other 24 ranks hung to the 12h timeout; the ring all_gather is
        # the collective every phase already exercises.
        if self._world_size == 1:
            return [obj]
        outs = [None] * self._world_size
        dist.all_gather_object(outs, obj)
        return outs if self._rank == dst else None

    def barrier(self):
        if self._world_size > 1:
            dist.barrier()

    # ---- generation cache ---------------------------------------------------
    def _gkey(self, ctx, until, max_new):
        import hashlib
        return hashlib.md5(f"{ctx}\x00{list(until)}\x00{max_new}\x00{self.pfx_ids}\x00{os.environ.get('GEN_REP_PENALTY','1.0')}\x00{os.environ.get('GEN_NO_REPEAT_NGRAM','0')}".encode()).hexdigest()

    def _gcache_put(self, key, text):
        self._gcache[key] = text
        if self._gcache_path:
            import json as _json
            with open(self._gcache_path, "a") as f:
                f.write(_json.dumps({"k": key, "g": text}) + "\n")

    # ---- scoring core ------------------------------------------------------

    @torch.no_grad()
    def _logits_rows(self, rows):
        """rows: list of int lists (<= max_length). Returns per-row logits on
        CPU fp32 [len(row)-... ] lazily via padded batch forward."""
        L = max(len(r) for r in rows)
        x = torch.zeros(len(rows), L, dtype=torch.long, device=self.dev)
        for i, r in enumerate(rows):
            x[i, :len(r)] = torch.tensor(r, dtype=torch.long)
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=self.dev == "cuda"):
            if self.arch == "gdn":
                logits = self.model(input_ids=x).logits
            elif self.arch == "gdn2lit":
                logits = self.model(x, max_seq_length=self._lit_max_seq)
            else:
                logits, _ = self.model(x)
        return logits.float()

    def _score(self, pairs):
        """pairs: list of (ctx_tokens, cont_tokens). Returns
        [(sum_logprob, is_greedy)] in order."""
        out = [None] * len(pairs)
        order = sorted(range(len(pairs)),
                       key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))
        for b0 in range(0, len(order), self.batch_rows):
            idxs = order[b0:b0 + self.batch_rows]
            rows, spans = [], []
            for i in idxs:
                ctx, cont = pairs[i]
                seq = (ctx + cont)[-(self.max_length + 1):]
                n_cont = len(cont)
                rows.append(seq)
                spans.append(n_cont)
            # feed seq[:-1]: the last target never needs to be an input, and
            # rolling windows are max_length+1 long (input must fit max_length)
            logits = self._logits_rows([r[:-1] for r in rows])
            for j, i in enumerate(idxs):
                seq, n_cont = rows[j], spans[j]
                lp = F.log_softmax(logits[j, :len(seq) - 1], dim=-1)
                tgt = torch.tensor(seq[1:], device=lp.device)
                tok_lp = lp.gather(-1, tgt[:, None]).squeeze(-1)
                cont_lp = tok_lp[-n_cont:]
                greedy = (lp[-n_cont:].argmax(-1) == tgt[-n_cont:]).all().item()
                out[i] = (cont_lp.sum().item(), bool(greedy))
            del logits
            if self.dev == "cuda":
                torch.cuda.empty_cache()
        return out

    # ---- lm-eval API -------------------------------------------------------

    def loglikelihood(self, requests):
        """Continuations are sliced from the JOINT encoding of ctx+cont:
        SentencePiece tokenizes a leading-space continuation in isolation as
        ['▁', ...] — a bare '▁' that never occurs in running text and
        corrupts absolute logprobs (lambada ppl 42M on GDN before this fix;
        multiple-choice accs survived only because all choices shared the
        artifact)."""
        pairs = []
        for req in requests:
            ctx, cont = req.args
            whole = self.tok(ctx + cont, add_special_tokens=False)["input_ids"]
            ctx_ids = self.tok(ctx, add_special_tokens=False)["input_ids"] \
                if ctx else []
            n_cont = len(whole) - len(ctx_ids)
            if n_cont <= 0:              # boundary merged into ctx: back off
                whole = ctx_ids + self.tok(cont, add_special_tokens=False)["input_ids"]
                n_cont = len(whole) - len(ctx_ids)
            seq = [self.bos] + whole
            pairs.append((seq[:-n_cont], seq[-n_cont:]))
        return self._score(pairs)

    def _generate_hf(self, requests):
        """gdn arch: BATCHED greedy generation via HF generate (fla supports caching).
        The old batch-1 loop left the GPU at ~12% util (recurrent decode has no
        chunk-kernel; one 2K-context sequence at a time is latency-bound) — TQA/DROP
        were intractable. Here we LEFT-PAD batch_rows prompts + pass attention_mask so
        the recurrent state ignores pad, and decode the whole batch in one generate()."""
        pad_id = self.tok.eos_token_id
        enc = []
        for req in requests:
            ctx, kw = req.args
            max_new = kw.get("max_gen_toks", 128)
            toks = [self.bos] + self.pfx_ids + self.tok(_gen_ctx(ctx), add_special_tokens=False)["input_ids"]
            enc.append((toks[-(self.max_length - max_new):], kw.get("until", []), max_new))
        res = [None] * len(enc)
        for b0 in range(0, len(enc), self.batch_rows):
            sub = enc[b0:b0 + self.batch_rows]
            L = max(len(t) for t, _, _ in sub)
            max_new = max(m for _, _, m in sub)
            ids = torch.full((len(sub), L), pad_id, dtype=torch.long)
            am = torch.zeros((len(sub), L), dtype=torch.long)
            for i, (toks, _, _) in enumerate(sub):
                ids[i, L - len(toks):] = torch.tensor(toks)   # LEFT pad
                am[i, L - len(toks):] = 1
            ids, am = ids.to(self.dev), am.to(self.dev)
            with torch.no_grad(), torch.autocast(
                    "cuda", dtype=torch.bfloat16, enabled=self.dev == "cuda"):
                out = self.model.generate(ids, attention_mask=am, max_new_tokens=max_new,
                                          do_sample=False, eos_token_id=pad_id,
                                          pad_token_id=pad_id)
            gen = out[:, L:]
            for i, (_, until, _) in enumerate(sub):
                text = self.tok.decode(gen[i].tolist(), skip_special_tokens=True)
                for u in until:
                    if u in text:
                        text = text.split(u)[0]
                res[b0 + i] = text
        return res

    def loglikelihood_rolling(self, requests):
        """Disjoint max_length windows (standard rolling protocol), but ALL
        windows across ALL docs scored in one batched _score call — the scan
        is latency-bound, so per-window scoring costs ~75s each regardless
        of rows; batched it's the same 75s per 32-64 windows."""
        pairs, owner = [], []
        for di, req in enumerate(requests):
            (text,) = req.args
            toks = [self.bos] + self.tok(text, add_special_tokens=False)["input_ids"]
            for w0 in range(0, len(toks) - 1, self.max_length):
                seq = toks[w0:w0 + self.max_length + 1]
                if len(seq) < 2:
                    break
                pairs.append((seq[:1], seq[1:]))
                owner.append(di)
        scores = self._score(pairs)
        res = [0.0] * len(requests)
        for di, (s, _) in zip(owner, scores):
            res[di] += s
        return res

    # ---- CUDA-graph decode (lm/eval/graph_decode.py; opt-in, validated path) ----

    graph_decode = False          # set True (after validate_graph_decode.py PASS) to use it

    def _generate_graph(self, requests):
        """Same bucketing/padding/truncation as generate_until, but every bucket runs on
        GraphDecoder: one captured graph per decode step, one launch per token."""
        from lm.eval.graph_decode import GraphDecoder
        Q = 32
        items = []
        for i, req in enumerate(requests):
            ctx, kw = req.args
            until = kw.get("until", [])
            max_new = kw.get("max_gen_toks", 128)
            toks = [self.bos] + self.tok(_gen_ctx(ctx), add_special_tokens=False)["input_ids"]
            cap = ((self.max_length - max_new) // Q) * Q
            toks = _trunc_prompt(toks, cap)
            # round the prompt UP to a Q-multiple with left BOS padding (never drop content: the
            # old floor-rounding dropped up to Q-1 leading passage tokens, 2026-09-08)
            L = min(cap, max(Q, ((len(toks) + Q - 1) // Q) * Q))
            row = toks[-L:]
            if len(row) < L:
                row = [self.bos] * (L - len(row)) + row
            items.append((L, row, tuple(until), max_new, i))
        res = [None] * len(requests)
        from collections import defaultdict
        buckets = defaultdict(list)
        for it in items:
            buckets[(it[0], it[2], it[3])].append(it)
        dec, dec_key = None, None
        eos = self.tok.eos_token_id
        for (L, until, max_new), group in sorted(buckets.items(), key=lambda kv: kv[0][0]):
            P = min(self.max_length, ((L + max_new + 255) // 256) * 256)
            key = (self.batch_rows, P)
            if key != dec_key:                       # one live decoder (22GB @P8448/B16): rebuild on change
                del dec; torch.cuda.empty_cache()
                dec = GraphDecoder(self.model, self.batch_rows, P).capture(); dec_key = key
            def stop_fn(g_cpu, until=until):
                if not until:
                    return torch.zeros(g_cpu.shape[0], dtype=torch.bool)
                return torch.tensor([any(u in self.tok.decode(g_cpu[b].tolist()) for u in until)
                                     for b in range(g_cpu.shape[0])])
            for g0 in range(0, len(group), self.batch_rows):
                sub = group[g0:g0 + self.batch_rows]
                rows = [it[1] for it in sub]
                while len(rows) < self.batch_rows:       # pad the batch with copies (ignored)
                    rows.append(rows[-1])
                x = torch.tensor(rows, device=self.dev)
                gen, _ = dec.generate(x, max_new, eos_id=eos, stop_fn=stop_fn)
                for b, it in enumerate(sub):
                    ids = gen[b].tolist()
                    if eos in ids:
                        ids = ids[:ids.index(eos)]
                    text = self.tok.decode(ids)
                    for u in until:
                        if u in text:
                            text = text.split(u)[0]
                    res[it[4]] = text
        del dec; torch.cuda.empty_cache()
        return res

    def _generate_reforward(self, requests):
        """Greedy generation by FULL re-forward per token (no decode cache): every prompt at its exact
        positions [0, len), rows bucketed by T = 64-multiple covering len + max_new. Used for models
        without a decode API (gdn2lit). Per-rank cache like the engine path."""
        import hashlib, pickle
        cache_dir = os.environ.get("GEN_CACHE_DIR"); cpath = None
        if cache_dir:
            h = hashlib.sha1((f"v{_GEN_IMPL_VER}:rf:{os.environ.get('GEN_TRUNC', 'tail')}:" + repr([(r.args[0], sorted(r.args[1].items())) for r in requests])).encode()).hexdigest()[:12]
            cpath = os.path.join(cache_dir, f"{os.environ.get('GEN_TAG', 'gen')}_w{self._world_size}_r{self._rank}_{h}.pkl")
            if os.path.exists(cpath):
                with open(cpath, "rb") as f:
                    res = pickle.load(f)
                if len(res) == len(requests):
                    print(f"[gen r{self._rank}] CACHE HIT {cpath}", flush=True); return res
        items = []
        for i, req in enumerate(requests):
            ctx, kw = req.args
            until = kw.get("until", []); max_new = kw.get("max_gen_toks", 128)
            toks = [self.bos] + getattr(self, "pfx_ids", []) + self.tok(_gen_ctx(ctx), add_special_tokens=False)["input_ids"]
            cap = self.max_length - max_new
            toks = _trunc_prompt(toks, cap)[-cap:]
            T = min(self.max_length, ((len(toks) + max_new + 63) // 64) * 64)
            items.append((T, toks, tuple(until), max_new, i))
        res = [None] * len(requests)
        from collections import defaultdict
        buckets = defaultdict(list)
        for it in items:
            buckets[(it[0], it[2], it[3])].append(it)
        eos = self.tok.eos_token_id; B = max(1, getattr(self, "gen_rows", 4))
        import time as _time
        _t0 = _time.time(); _done = 0
        print(f"[gen r{self._rank}] {len(requests)} requests in {len(buckets)} buckets (re-forward)", flush=True)
        for (T, until, max_new), group in sorted(buckets.items(), key=lambda kv: kv[0][0]):
            group = sorted(group, key=lambda it: len(it[1]))
            for g0 in range(0, len(group), B):
                sub = group[g0:g0 + B]; lens = [len(it[1]) for it in sub]; nb = len(sub)
                x = torch.zeros(nb, T, dtype=torch.long, device=self.dev)
                for b, it in enumerate(sub):
                    x[b, :lens[b]] = torch.tensor(it[1], dtype=torch.long, device=self.dev)
                pos = torch.tensor(lens, device=self.dev); ar = torch.arange(nb, device=self.dev)
                outs = [[] for _ in range(nb)]; alive = [True] * nb
                for k in range(max_new):
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        lg = self.model(x, max_seq_length=self._lit_max_seq) if self.arch == "gdn2lit" else self.model(x)[0]
                    nxt = self._pick_next(lg[ar, pos - 1], outs, alive)
                    for b in range(nb):
                        if not alive[b]:
                            continue
                        t = int(nxt[b])
                        if t == eos:
                            alive[b] = False; continue
                        outs[b].append(t)
                        if until and any(u in self.tok.decode(outs[b][-8:]) for u in until):
                            alive[b] = False
                    if not any(alive) or int(pos.max()) >= T:
                        break
                    wr = pos < T
                    x[ar[wr], pos[wr]] = nxt[wr]; pos = pos + 1
                for b, it in enumerate(sub):
                    text = self.tok.decode(outs[b])
                    for u in until:
                        if u in text:
                            text = text.split(u)[0]
                    res[it[4]] = text
                _done += nb
            print(f"[gen r{self._rank}] bucket T={T} n={len(group)} done {_done}/{len(requests)} {(_time.time()-_t0)/60:.1f} min", flush=True)
        if cpath:
            os.makedirs(cache_dir, exist_ok=True)
            with open(cpath + ".tmp", "wb") as f:
                pickle.dump(res, f)
            os.replace(cpath + ".tmp", cpath)
        return res


    # ---- decoding controls (env): GEN_REP_PENALTY (>1 penalises already-generated tokens, CTRL-style), GEN_NO_REPEAT_NGRAM
    def _pick_next(self, lg_rows, outs, alive):
        """lg_rows: [B, V] next-token logits (one row per sequence); outs: generated ids per row. Greedy with optional
        repetition penalty and no-repeat-ngram, applied identically to every model (protocol knob, default off)."""
        import os as _os
        pen = float(_os.environ.get("GEN_REP_PENALTY", "1.0")); ng = int(_os.environ.get("GEN_NO_REPEAT_NGRAM", "0"))
        if pen == 1.0 and ng == 0:
            return lg_rows.argmax(-1)
        lg = lg_rows.float().clone()
        for b in range(lg.shape[0]):
            if b < len(alive) and not alive[b]:
                continue
            seen = outs[b]
            if pen != 1.0 and seen:
                idx = torch.tensor(sorted(set(seen)), device=lg.device)
                v = lg[b, idx]; lg[b, idx] = torch.where(v > 0, v / pen, v * pen)
            if ng and len(seen) >= ng - 1:
                key = tuple(seen[-(ng - 1):]) if ng > 1 else ()
                banned = {seen[i + ng - 1] for i in range(len(seen) - ng + 1) if tuple(seen[i:i + ng - 1]) == key}
                if banned:
                    lg[b, torch.tensor(sorted(banned), device=lg.device)] = float("-inf")
        return lg.argmax(-1)

    def generate_until(self, requests):
        """Greedy batched generation. Rows are bucketed by EXACT prompt
        length so one scalar RoPE position serves the whole batch —
        decode_step is [B,1] batched and prefill+generation amortize across
        the bucket.

        EVAL_LEN_QUANT (default 1): bucket quantum. The old default of 32
        LEFT-TRUNCATED every prompt to a multiple of 32 tokens, i.e. silently
        deleted 0-31 tokens from the HEAD of every prompt (mean ~15). For
        stripped SQuAD that is the opening of the passage — 17% of docs lost
        the gold answer span outright (retrieval_protocol_investigation.md
        §11). With a leading whitespace pad the cut only ate whitespace, which
        is what made built-in/no-strip look "better". Q=1 = no truncation
        (only the max_length cap applies). Q=32 reproduces the old numbers."""
        if self.arch == "gdn":
            return self._generate_hf(requests)
        if self.arch == "gdn2lit":
            return self._generate_reforward(requests)
        if self.graph_decode:
            return self._generate_graph(requests)
        Q = int(os.environ.get("EVAL_LEN_QUANT", "1"))   # kept for the cap rounding only; lengths are exact
        items = []      # (bucket_len, toks, until, max_new, orig_idx)
        res = [None] * len(requests)
        keys = [None] * len(requests)
        n_hit = 0
        for i, req in enumerate(requests):
            ctx, kw = req.args
            until = kw.get("until", [])
            max_new = kw.get("max_gen_toks", 128)
            keys[i] = self._gkey(ctx, until, max_new)
            if keys[i] in self._gcache:
                res[i] = self._gcache[keys[i]]; n_hit += 1; continue
            if self._gcache_only:
                res[i] = "<UNCACHED>"; continue
            toks = [self.bos] + self.pfx_ids + self.tok(_gen_ctx(ctx), add_special_tokens=False)["input_ids"]
            cap = ((self.max_length - max_new) // Q) * Q
            toks = _trunc_prompt(toks, cap)
            L = min(len(toks), cap)     # EXACT length: no rounding, no dropped tokens, no BOS padding
            row = toks[-L:]             # (2026-09-13: floor-to-32 dropped BOS + up to 31 head tokens;
                                        #  round-up BOS padding is OOD (RULER SN3). Buckets = exact length.)
            items.append((L, row, tuple(until), max_new, i))
        if self._gcache_path:
            print(f"[gen-cache] {n_hit} served from cache, {len(items)} to generate", flush=True)
        from collections import defaultdict
        buckets = defaultdict(list)
        for it in items:
            buckets[(it[0], it[2], it[3])].append(it)
        for (L, until, max_new), group in buckets.items():
            for g0 in range(0, len(group), self.batch_rows):
                sub = group[g0:g0 + self.batch_rows]
                B = len(sub)
                x = torch.tensor([it[1] for it in sub], device=self.dev)
                caches = self.model.decode_init()
                outs = [[] for _ in range(B)]
                alive = [True] * B
                with torch.no_grad(), torch.autocast(
                        "cuda", dtype=torch.bfloat16,
                        enabled=self.dev == "cuda"):
                    logits = None
                    for pos in range(L):
                        logits = self.model.decode_step(x[:, pos:pos + 1],
                                                        pos, caches)
                    for k in range(max_new):
                        nxt = self._pick_next(logits[:, -1], outs, alive)          # [B]
                        for b in range(B):
                            if not alive[b]:
                                continue
                            t = nxt[b].item()
                            if t == self.tok.eos_token_id:
                                alive[b] = False
                                continue
                            outs[b].append(t)
                            if until and any(
                                    u in self.tok.decode(outs[b][-8:])
                                    for u in until):
                                alive[b] = False
                        if not any(alive) or L + k + 1 >= self.max_length:
                            break
                        logits = self.model.decode_step(
                            nxt[:, None], L + k, caches)
                for b, it in enumerate(sub):
                    text = self.tok.decode(outs[b])
                    for u in until:
                        if u in text:
                            text = text.split(u)[0]
                    res[it[4]] = text
                    self._gcache_put(keys[it[4]], text)
        return res


class Wave3DEval(DualModEval):
    """Teacher-forced (loglikelihood) scoring through the wave3d TRAINING engine
    (engines.wave3d.wave3d_x, C=64) at a chosen K: K=64 is exact by nilpotency at
    C=64, K=24 scores the model at its final training operator. Orders of
    magnitude faster than the eager sequential scan for likelihood tasks.
    Generation (generate_until) ALSO runs through the engine at the same K ("trained-operator
    decode": full re-forward of prompt+generated per token, no decode cache); see module docstring
    for the K=24 vs K=64 parity check.
    """

    def __init__(self, ckpt, tok_dir, batch_rows, T_eval, K, ctx_len=None,
                 engine_kwargs=None, compile_mode="all", gen_rows=4):
        T_eval = ((T_eval + 63) // 64) * 64
        super().__init__(ckpt, batch_rows=batch_rows, device="cuda",
                         tok_dir=tok_dir, max_length=T_eval, ctx_len=ctx_len)
        assert self.arch != "gdn", "engine scoring is DualModLM-only"
        from engines.wave3d.wave3d import make_engine
        kw = dict(C=64, graphs=False)                       # TF engine: validated default path (stash on)
        # operator identity travels with the checkpoint (2026-09-08): raw_diag / local_window
        ck_kw = self.ckpt_engine_kwargs or {}
        for k_ in ("raw_diag", "local_window"):
            if ck_kw.get(k_):
                kw[k_] = ck_kw[k_]
        lw = getattr(self.model.cfg, "local_window", 0) if hasattr(self.model, "cfg") else 0
        if lw and not kw.get("local_window"):
            kw["local_window"] = int(lw)
        kw.update(engine_kwargs or {})
        self._op_kw = {k_: kw[k_] for k_ in ("raw_diag", "local_window") if kw.get(k_)}
        if self._op_kw:
            print(f"[Wave3DEval] operator: {self._op_kw}", flush=True)
        self.engine = make_engine(self.model, batch_rows, T_eval, engine="x", **kw)
        if compile_mode and compile_mode != "none":
            self.engine.set_compile(compile_mode)
        self.K = int(K)
        self.T_eval = T_eval
        self.engine_compile = compile_mode
        self.gen_rows = gen_rows

    def generate_until(self, requests):
        return self._generate_engine(requests)

    # ---- generation THROUGH THE ENGINE (2026-09-08) ---------------------------------
    # The engine's within-chunk Jacobi sweeps do NOT converge to the sequential fixed point
    # (self term refined; K128 != K64 at several layers), so no per-token decode reproduces
    # the trained operator. The window mask is causal, so one engine forward over
    # [prompt + generated so far, right-padded] gives exactly the engine's logits for the
    # last real token: greedy generation = one forward per token at the training K.
    def _engine_for(self, B, T):
        key = (B, T)
        if getattr(self, "_gen_key", None) != key:
            from engines.wave3d.wave3d import make_engine
            if getattr(self, "_gen_eng", None) is not None:
                del self._gen_eng
                torch.cuda.empty_cache()
            op = getattr(self, "_op_kw", {})
            kw = dict(C=64, graphs=False, stash_attn=bool(op), **op)   # raw_diag/window need the grouped path
            self._gen_eng = make_engine(self.model, B, T, engine="x", **kw)
            if getattr(self, "engine_compile", "all") not in (None, "none"):
                self._gen_eng.set_compile(self.engine_compile)
            self._gen_key = key
        return self._gen_eng

    def _generate_engine(self, requests):
        # per-rank generation cache: a failed post-generation collective must not lose the
        # 50-min shard again. Keyed by (tag, rank, world, hash of the request args).
        import hashlib, pickle
        cache_dir = os.environ.get("GEN_CACHE_DIR")
        cpath = None
        if cache_dir:
            # _GEN_IMPL_VER salts the key: v2 = round-UP/BOS-kept prompts + GEN_TRUNC mode (2026-09-08)
            salt = f"v{_GEN_IMPL_VER}:{os.environ.get('GEN_TRUNC', 'tail')}:{os.environ.get('GEN_NO_BOS', '0')}:{os.environ.get('GEN_LEGACY_ROUND', '0')}:"
            h = hashlib.sha1((salt + repr([(r.args[0], sorted(r.args[1].items())) for r in requests])).encode()).hexdigest()[:12]
            cpath = os.path.join(cache_dir, f"{os.environ.get('GEN_TAG', 'gen')}_w{self._world_size}_r{self._rank}_{h}.pkl")
            if os.path.exists(cpath):
                with open(cpath, "rb") as f:
                    res = pickle.load(f)
                if len(res) == len(requests):
                    print(f"[gen r{self._rank}] CACHE HIT {cpath} ({len(res)} generations)", flush=True)
                    return res
        res = self._generate_engine_impl(requests)
        if cpath:
            os.makedirs(cache_dir, exist_ok=True)
            with open(cpath + ".tmp", "wb") as f:
                pickle.dump(res, f)
            os.replace(cpath + ".tmp", cpath)
            print(f"[gen r{self._rank}] cached -> {cpath}", flush=True)
        return res

    def _generate_engine_impl(self, requests):
        """Greedy generation through the training engine, PER-ROW positions: every prompt keeps
        its exact tokens (BOS + text, no left padding, no dropped leading tokens) at positions
        [0, len) and generates from its own end. Rows are bucketed by (T, until, max_new) only,
        T = the 64-multiple covering len + max_new. (v3, 2026-09-08: the round-DOWN path dropped
        BOS + up to 31 haystack tokens; the round-UP path prepended up to 31 BOS -- both moved
        RULER/SQuAD by 5-25 points.)"""
        items = []
        for i, req in enumerate(requests):
            ctx, kw = req.args
            until = kw.get("until", [])
            max_new = kw.get("max_gen_toks", 128)
            toks = _bos_prefix(self.bos) + getattr(self, "pfx_ids", []) + self.tok(_gen_ctx(ctx), add_special_tokens=False)["input_ids"]
            cap = self.max_length - max_new
            toks = _trunc_prompt(toks, cap)[-cap:]
            if os.environ.get("GEN_LEGACY_ROUND") == "1":     # diagnostic: emulate the old floor-to-32 drop
                toks = toks[-max(32, (len(toks) // 32) * 32):]
            T = min(self.max_length, ((len(toks) + max_new + 63) // 64) * 64)
            items.append((T, toks, tuple(until), max_new, i))
        res = [None] * len(requests)
        from collections import defaultdict
        buckets = defaultdict(list)
        for it in items:
            buckets[(it[0], it[2], it[3])].append(it)
        eos = self.tok.eos_token_id
        B = self.gen_rows
        import time as _time
        _t0 = _time.time(); _done = 0; _nfwd = 0
        print(f"[gen r{self._rank}] {len(requests)} requests in {len(buckets)} buckets", flush=True)
        for (T, until, max_new), group in sorted(buckets.items(), key=lambda kv: kv[0][0]):
            eng = self._engine_for(B, T)
            group = sorted(group, key=lambda it: len(it[1]))
            for g0 in range(0, len(group), B):
                sub = group[g0:g0 + B]
                lens = [len(it[1]) for it in sub]
                x = torch.zeros(B, T, dtype=torch.long, device=self.dev)
                for b, it in enumerate(sub):
                    x[b, :lens[b]] = torch.tensor(it[1], dtype=torch.long, device=self.dev)
                for b in range(len(sub), B):                      # dead rows: copy of the last real row
                    x[b] = x[len(sub) - 1]
                pos = torch.tensor(lens + [lens[-1]] * (B - len(sub)), device=self.dev)   # next position per row
                outs = [[] for _ in range(B)]
                alive = [True] * len(sub)
                ar = torch.arange(B, device=self.dev)
                for k in range(max_new):
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        lg = eng.forward_logits(x, self.K)
                    nxt = self._pick_next(lg[ar, pos - 1], outs, alive)                  # prediction for each row's own position
                    for b in range(len(sub)):
                        if not alive[b]:
                            continue
                        t = int(nxt[b])
                        if t == eos:
                            alive[b] = False
                            continue
                        outs[b].append(t)
                        if until and any(u in self.tok.decode(outs[b][-8:]) for u in until):
                            alive[b] = False
                    if not any(alive) or int(pos.max()) >= T:
                        break
                    wr = pos < T
                    x[ar[wr], pos[wr]] = nxt[wr]
                    pos = pos + 1
                if hasattr(eng, "_stash"):
                    eng._stash = {}
                _nfwd += k + 1
                for b, it in enumerate(sub):
                    text = self.tok.decode(outs[b])
                    for u in until:
                        if u in text:
                            text = text.split(u)[0]
                    res[it[4]] = text
                _done += len(sub)
            print(f"[gen r{self._rank}] bucket T={T} n={len(group)} done {_done}/{len(requests)} fwd {_nfwd} {(_time.time()-_t0)/60:.1f} min", flush=True)
        return res

    @torch.no_grad()
    def _logits_rows(self, rows):
        L = max(len(r) for r in rows)
        assert L <= self.T_eval, f"{L} > {self.T_eval}"
        x = torch.zeros(self.batch_rows, self.T_eval, dtype=torch.long, device=self.dev)
        for i, r in enumerate(rows):
            x[i, :len(r)] = torch.tensor(r, dtype=torch.long)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = self.engine.forward_logits(x, self.K)     # [batch_rows, T_eval, V]
        if hasattr(self.engine, "_stash"):                  # eager no_grad memo (177GB @T4096 measured)
            self.engine._stash = {}
        return lg[:len(rows), :L].float()

