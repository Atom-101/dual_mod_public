# Baselines

| arm (paper name) | implementation | harness flag |
|---|---|---|
| Transformer++ | `models/dualmod/model.py` `DualModLM` with `attn_mode="vanilla"` (same LLaMA-style trunk as DM: RMSNorm pre-norm, RoPE, SwiGLU, tied embeddings; the dual-modulation branch is disabled) | `--arch vanilla` |
| LSTM | `lstm_lm.py` (`torch.nn.LSTM`, tied embedding/head) | `--arch lstm` |
| Gated DeltaNet | `gdn.py` (`fla.models.GatedDeltaNetForCausalLM`) | `--arch gdn` |
| Looped transformer, iteration-agnostic and K∝length | `models/looped/looped_lm.py` | `--arch looped` |
| Recurrent-depth (Geiping et al.) | `models/recurrent_depth/recurrent_depth_lm.py` | `--arch looped --loop_huginn` |
| Full-bandwidth transformer (Wang et al.) | `models/fbt/fbt_lm.py` | `--arch fbt` |

The 340M language-modeling baselines (`lm/train_340m/train_baseline.py`) use the same two classes:
`--arch vanilla340` is `DualModLM(attn_mode="vanilla")` at d=1024, and `--arch gdn` loads
`lm/train_340m/gdn340m_config.json` into `fla`. The 1.3B GDN-2 comparison uses the third-party checkpoint and
the official lit_gpt code (see `lm/eval/gdn2lit_shim.py`), not a re-implementation.
