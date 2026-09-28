"""DM 1.3B / 100B-tok FineWeb-Edu hero config — matches GatedDeltaNet-2's
swa_gdn2_1.3B macro-shape (n_layer 18, n_embd 2304, n_head 18-equiv, T=4096,
vocab 32k) at ~1.30B total, so our number drops into their published table.

Arch decision (2026-09-01): d=2304 EXACT (GDN-2 hidden dim; clean 9x256 kernel
tiling), L=18, 36 heads x head_dim 64, LoRA rank 144 (= "lora-16", d/16) on BOTH
gates and ctx operators. lora-8 (d/8=288) would OVERSHOOT to 1.388B; lora-16
lands at 1.304B (+4M vs GDN-2's 1.300B). The MRK stability stack (mult_res ctx,
k-mod head, k-norm, v-clamp) is carried from the from-scratch flagship.

Optimizer NOTE: use STANDARD fused AdamW (fp32 moments), NOT 8-bit — the 8B run
used PagedAdamW8bit but the GDN-2 table comparison must be apples-to-apples.
fp32 moments at 1.3B = ~10.4GB, trivial in 268GB.
"""

from models.dualmod.config import DualModConfig

VOCAB = 32000          # TinyLlama/Llama-2 32k (GDN-2 table tokenizer)
D_MODEL = 2304         # == swa_gdn2_1.3B n_embd
N_LAYERS = 18          # == swa_gdn2_1.3B n_layer
N_HEADS = 36           # head_dim = 2304/36 = 64
SEQ_LEN = 4096         # == GDN-2 block_size
LORA_RANK = 144        # d/16; gate_rank == ctx_rank -> 1.304B total


def build_config(seq_len: int = SEQ_LEN) -> DualModConfig:
    return DualModConfig(
        # --- macro shape (GDN-2 match) ---
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        vocab_size=VOCAB,
        max_seq_len=seq_len,
        rope_theta=10000.0,
        mlp_ratio="swiglu_8_3",   # h = round(8/3*d/64)*64 = 6144
        tie_embeddings=True,
        # --- DM MRK stability stack (carried from FS flagship) ---
        attn_mode="sequential",
        enable_key_mod=True,
        enable_value_mod=True,
        ctx_mod="mult_res",
        gate_rank=LORA_RANK,
        ctx_rank=LORA_RANK,
        k_ctx_norm="rms",
        k_gain_init=1.75,
        kmod_vraw_heads=1,        # anti-flooding raw-value head
        v_clamp_tau=150.0,        # entry-norm clamp storm governor
        gate_bias_init=4.0,       # NOTE: revisit for COLD from-scratch (A5 escape)
        checkpoint_chunk=64,
        dtype="bf16",
    )


if __name__ == "__main__":
    from models.dualmod.model import DualModLM
    cfg = build_config()
    m = DualModLM(cfg)
    bd = m.param_breakdown()
    print(f"d_model={cfg.d_model} L={cfg.n_layers} heads={cfg.n_heads} "
          f"head_dim={cfg.head_dim} mlp_hidden={cfg.mlp_hidden} "
          f"lora_rank={cfg.gate_rank}")
    print(f"base={bd['base']/1e6:.1f}M  dualmod_extra={bd['dualmod_extra']/1e6:.1f}M  "
          f"total={bd['total']/1e9:.4f}B")
    print(f"non_embedding={m.num_params(non_embedding=True)/1e9:.4f}B")
