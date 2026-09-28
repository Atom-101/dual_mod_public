"""DualModConfig — the full configuration surface from plan.md §7."""

from dataclasses import dataclass, field, asdict


@dataclass
class DualModConfig:
    # architecture
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8                 # query heads
    n_kv_heads: int = 0              # 0 => = n_heads (MHA); < n_heads => GQA (vanilla path)
    head_dim_override: int = 0       # 0 => d_model//n_heads; else fixes head_dim, decoupling d_attn
    vocab_size: int = 50257          # GPT-2 BPE
    max_seq_len: int = 512
    local_window: int = 0            # sliding-window attention, W tokens incl. self (0 = full causal); GDN-2 SWA convention
    rope_theta: float = 10000.0
    use_rope: bool = True             # False = NoPE (identity rotation; length-gen protocol)
    mlp_ratio: str = "swiglu_8_3"    # SwiGLU, hidden = round(8/3 * d) to multiple of 64
    mlp_hidden_override: int = 0     # 0 => derive from mlp_ratio (E-match param knob)
    tie_embeddings: bool = True

    # dual-mod
    attn_mode: str = "sequential"    # "vanilla" | "sequential" | "deq"
    enable_key_mod: bool = True
    enable_value_mod: bool = True
    gate_type: str = "convex_sigmoid"  # "independent" | "unbounded" as options, untrained in v0
    # VALUE-gate convexity (independent of the legacy `gate_type`, which stays
    # convex for keys ALWAYS). "convex": v'' = g*v_raw + (1-g)*v_ctx (raw/ctx
    # rivalry per dim). "independent": w_gv widens to 2d, two sigmoids
    #   g_raw, g_ctx = sig(gv[:d]), sig(gv[d:]);  v'' = g_raw*v_raw + g_ctx*v_ctx
    # Adds the none/both states; the box [0,1]^2 is amplitude-safe (<=2x row
    # bound) BECAUSE v_norm(+tanh ctx) bounds the channels and norm_ctx cuts
    # per-hop amplification. Costs 2d^2/layer (param-compensate via mlp).
    gate_type_v: str = "convex"      # "convex" | "independent"
    gate_input: str = "xu"           # "xu" (concat x̂,u) | "u" | "x"
    gate_bias_init: float = 4.0
    # structured per-dim gate-bias inits (plans/asym_gate.md). "scalar" uses
    # gate_bias_init; "ugi" = uniform-gate-init spread over [eps,1-eps] at mean 0.5.
    gate_bias_init_k: str = "scalar"  # "scalar" | "ugi" | "committed"
    gate_bias_init_v: str = "scalar"  # "scalar" | "ugi" | "band"
    ugi_eps: float = 0.04
    key_commit_bias: float = 3.0      # G-B: committed key dims at ±3 (sigma 0.953/0.047)
    key_frontier_pairs: int = 6       # G-B: mobile frontier RoPE pairs per head
    key_frontier_hi: float = 1.5
    value_band_lo: float = 0.2        # G-B: value UGI band U[lo,hi]
    value_band_hi: float = 0.8
    ctx_reserved_init_std: float = 0.02  # G-B: W_Kctx rows for ctx-committed (reserved) dims
    gate_init_seed: int = 1337        # tied to training seed in train.py
    ctx_proj_init_std: float = 0.006  # ~0.02/sqrt(2L)-style small init
    rmsnorm_on_o: bool = True        # do not disable in v0 runs
    deq_sweeps: int = 4              # used only in deq mode / T4 test
    kmod_vraw_heads: int = 0         # first N heads: keys modulated, values
                                     # RAW (anti-flooding head: addressing
                                     # kept, verbatim content preserved)
    ctx_mod: str = "linear"          # ctx-path transition strength (LSTM-gap
                                     # arms): linear | vmlp (v'=MLP(u)) |
                                     # fusion (v'=MLP([xhat;u])) | kmlp
                                     # (k'=MLP(u), falsification)
    kmod_vraw_bound: str = "none"    # "none" | "tanh": bound the raw-value
                                     # channel of kmod heads. Unbounded raw v
                                     # = ungated loudness -> L6-h0 flooding
                                     # loop, deterministic divergence ~step
                                     # 800 at dev scale (CMK-lin, Jul 14)
    kmod_mode: str = "raw"           # "raw" | "gated_norm" (kmod_vraw_heads
                                     # > 0 only; intended with v_norm=rms).
                                     # raw: the kmod head commits v_raw
                                     # verbatim (dead v-gate rows, +30 pin in
                                     # the engines). gated_norm: the head's
                                     # value is a WRITE GATE on the (normed)
                                     # raw value — v'' = sigmoid(gv_logits)
                                     # * v_raw, no ctx term. Implemented as:
                                     # gate logits stay LIVE (no pin) and the
                                     # head's v_ctx slice is zeroed, so the
                                     # standard convex blend degenerates to
                                     # the write gate exactly. w_gv head rows
                                     # come alive; w_vctx head rows go dead.
    gate_rank: int = 0               # >0: LoRA-factored gates (gate_in ->
                                     # r -> d, no bias on the down-proj).
                                     # Gate params 4d^2 -> ~0.75d^2 at r=d/8;
                                     # runs on the verbatim engine path.
    ctx_rank: int = 0                # >0: LoRA-factor the four full dxd ctx
                                     # operators — w_kctx, w_vctx (ctx
                                     # projections) and v_gate, v_map
                                     # (mult_res transition) — as down: d->r
                                     # then up: r->d (r = ctx_rank). The fused
                                     # engine still materializes the full dxd
                                     # weight (up@down) for its K-sweep kernels
                                     # and projects the composed dW back onto
                                     # the factors, exactly like the LoRA gates
                                     # (gate_rank). 0 = off (byte-identical:
                                     # the matrices stay dense nn.Linear).
    v_norm: str = "none"             # "none" | "rms": per-head RMS-norm the
                                     # raw value projection (fp32, eps 1e-6,
                                     # learned per-head scalar gain).
                                     # Constraint: unbounded per-token value
                                     # amplitude is a softmax-EXEMPT salience
                                     # channel (flooding pathology); norming
                                     # moves amplitude coding into the
                                     # bounded gates. Applies BEFORE the
                                     # kmod_vraw_heads splice — the verbatim
                                     # channel becomes unit-scale and its
                                     # loudness role migrates to the gates.
    v_ctx_norm: str = "none"         # "none" | "rms": per-head RMS-norm the
                                     # ctx value candidate v_ctx (the OUTPUT
                                     # of w_vctx, every ctx_mod) in fp32
                                     # (eps 1e-6) with a learned per-head
                                     # gain ctx_gain ([n_heads], init 1.0,
                                     # 1-D -> no-decay group) — mirrors
                                     # project_v's treatment of v_raw.
                                     # Rationale: bounds the write amplitude
                                     # of the ctx channel (fusion's candidate
                                     # is unbounded — identity+silu);
                                     # direction preserved (unlike tanh) so
                                     # depth-seeking dynamics survive; with
                                     # v_norm + independent gates this
                                     # completes the symmetric write law
                                     # v'' = g_r*N(v_raw) + g_c*N(v_ctx).

    # mult_res-lambda (the structural non-expansivity fix): leak the identity
    # carry in the mult_res ctx candidate,
    #   v_in = lambda * u + sigmoid(v_gate) * tanh(v_map(u)),
    # so the identity's contribution to the chain Jacobian is < 1 (contractive)
    # while the tanh bounds the branch output. 1.0 = current mult_res
    # (byte-identical). Recommended arm: 0.97 (fixed). See rho_retrodict.md.
    mult_res_lambda: float = 1.0
    # K-norm (read-side storm governor): per-head RMSNorm + learnable per-head
    # gain (k_gain) on the ctx KEY candidate k_ctx = w_kctx(u), BEFORE the
    # blend k'' = g*k_raw + (1-g)*N(k_ctx). Identity at g=1 (raw path / vanilla
    # FA untouched — the g=1 exactness contract). Caps the only key-side
    # channel inside the refinement recurrence (loop var), where the read-side
    # logit escalation lives (logit_decomp: ctx-term = 65-77% of top logits).
    # k_gain init = survivors' equilibrium k_ctx RMS (~1.75 on DM-d832).
    # "none" = off (byte-identical). The k-side twin of v_ctx_norm.
    k_ctx_norm: str = "none"         # "none" | "rms"
    # Length-adaptive logit scale (Chiang & Cholak 2022): multiply attention logits by ln(n_keys)/ln(attn_logn_ref),
    # n_keys = number of attended keys at the position (incl. self). Identity at n_keys == ref; 0 = off.
    # Eager paths only (attend_one / static / deq); the WaveScan engine does not implement it.
    attn_logn_ref: int = 0
    k_gain_init: float = 1.75
    # v'' entry-norm clamp (the storm governor, NOT a norm): the written
    # value is capped at a per-(position,head) L2 radius tau —
    #   v'' <- v'' * min(1, tau/||v''||_head)
    # bit-exact identity for every in-range head (the vast majority), active
    # only on the p99.9 amplitude tail the coherent mode must ride. tau is
    # calibrated from a HEALTHY run's p99.9 (vtail/p999_* telemetry). 0 = off.
    # An EVENT governor (projection onto the ball), the opposite of v_norm
    # (an always-on sphere projection = regime change). See
    # analysis/rho_retrodict.md.
    v_clamp_tau: float = 0.0
    # Deterministic fused-sweep backward: routes the FusedSweepCell backward's
    # float atomic_add accumulators (RMS-norm weight grad; and, when a chunk is
    # split across row tiles, the dK/dV cache-grad and dK_in/dV_in merges)
    # through a run-to-run bit-identical reduction (per-program partials +
    # fixed-order torch sum) instead of tl.atomic_add. False (default) =
    # BYTE-IDENTICAL to the historical non-deterministic path (the live runs).
    # True selects the separate deterministic path. Used to test whether the
    # chain_rho storms are kernel-noise-driven. FusedSweepCell only.
    deterministic_bwd: bool = False
    # training
    checkpoint_chunk: int = 64       # scan checkpointing granularity; 0 = off
    dtype: str = "bf16"              # fp32 softmax + fp32 gate sigmoid inside autocast

    def __post_init__(self):
        if self.head_dim_override == 0:
            assert self.d_model % self.n_heads == 0
        assert self.n_kv_heads_eff <= self.n_heads and self.n_heads % self.n_kv_heads_eff == 0
        # decoupled d_attn / GQA is only wired for the vanilla SDPA path (the F controls)
        if self.attn_mode != "vanilla":
            assert self.head_dim_override == 0 and self.n_kv_heads == 0, \
                "n_kv_heads / head_dim_override are vanilla-path only"
        assert self.attn_mode in ("vanilla", "sequential", "deq", "static")
        assert self.gate_type in ("convex_sigmoid", "independent", "unbounded")
        assert self.gate_type_v in ("convex", "independent")
        assert self.gate_bias_init_k in ("scalar", "ugi", "committed")
        assert self.gate_bias_init_v in ("scalar", "ugi", "band")
        assert self.gate_input in ("xu", "u", "x")
        assert self.kmod_vraw_bound in ("none", "tanh")
        assert self.v_norm in ("none", "rms")
        assert self.v_ctx_norm in ("none", "rms")
        assert self.k_ctx_norm in ("none", "rms")
        assert self.kmod_mode in ("raw", "gated_norm")
        if self.gate_rank > 0:
            assert self.gate_bias_init_k == "scalar" \
                and self.gate_bias_init_v == "scalar", \
                "structured gate-bias inits assume full-rank gate matrices"
        if self.ctx_rank > 0:
            # both paths write into the DENSE w_kctx/w_vctx rows at init,
            # which is a read-only property on the factored operators.
            assert self.gate_bias_init_k != "committed", \
                "committed key init writes dense W_Kctx rows; incompatible with ctx_rank"
            assert not self.ctx_mod.endswith("_hw"), \
                "_hw re-inits dense w_vctx; incompatible with ctx_rank"
        # the tanh loudness cap belongs to the raw-splice channel only;
        # gated_norm has no raw splice (amplitude lives in v_norm + gate)
        assert not (self.kmod_mode == "gated_norm"
                    and self.kmod_vraw_bound != "none"), \
            "kmod_vraw_bound applies to kmod_mode='raw' only"
        assert self.mlp_ratio == "swiglu_8_3", "only swiglu_8_3 is implemented"

    @property
    def head_dim(self) -> int:
        return self.head_dim_override or (self.d_model // self.n_heads)

    @property
    def n_kv_heads_eff(self) -> int:
        return self.n_kv_heads or self.n_heads

    @property
    def d_attn(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def mlp_hidden(self) -> int:
        # hidden = round(8/3 * d) to nearest multiple of 64;
        # override used by param-matched controls (flagship E-match)
        if self.mlp_hidden_override:
            return self.mlp_hidden_override
        return int(round(self.d_model * 8 / 3 / 64) * 64)

    @property
    def gate_in_dim(self) -> int:
        return 2 * self.d_model if self.gate_input == "xu" else self.d_model

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainConfig:
    run_name: str = "B-dualmod"
    out_dir: str = "out"
    data_dir: str = "data"

    batch_size: int = 128            # sequences per step (scan throughput comes from B)
    grad_accum: int = 1
    tokens_target: float = 300e6     # matched token budget for runs A and B
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_frac: float = 0.01
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    seed: int = 1337                 # model init / training seed
    data_seed: int = 42              # identical token stream across runs A/B/C

    log_interval: int = 10
    eval_interval: int = 250         # val loss
    eval_iters: int = 32
    telemetry_interval: int = 500    # §10 telemetry cadence
    ckpt_interval: int = 2000

    wandb_log: bool = False
    wandb_project: str = "dualmod"
    compile: bool = False
    device: str = "cuda:0"

    def to_dict(self) -> dict:
        return asdict(self)
