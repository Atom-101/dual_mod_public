"""Gated DeltaNet baseline (Yang et al. 2025) via flash-linear-attention (`pip install flash-linear-attention`).

The formal-language runs build it with num_heads = h/2, head_dim 64, expand_v 1, hidden_ratio 4, tied embeddings,
i.e. the same width/depth as the Transformer++ and DM arms of the same `--scale`.
"""


def build_gdn(vocab, d, L, h):
    from fla.models import GatedDeltaNetConfig, GatedDeltaNetForCausalLM
    cfg = GatedDeltaNetConfig(hidden_size=d, num_hidden_layers=L, num_heads=h // 2, head_dim=64, expand_v=1,
                              hidden_ratio=4, vocab_size=vocab, tie_word_embeddings=True)
    return GatedDeltaNetForCausalLM(cfg)
