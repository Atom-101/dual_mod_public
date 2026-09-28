"""Recurrent-depth transformer (Geiping et al. 2025, "Scaling up test-time compute with latent reasoning").

The recipe, as implemented in models/looped/looped_lm.py (LoopedLM with huginn=True):
  * prelude: the token embedding, scaled by gamma = sqrt(d) (their Sec. 3.2), through one non-looped block;
  * the loop state s_0 is drawn from a truncated normal with variance 2/5 (their Sec. 4.1);
  * every iteration: adapter A: R^{2d} -> R^d on [s; e] (the embedding is re-injected each step), then the
    weight-tied core blocks with sandwich RMSNorm (n2(x + Attn(n1 x)), n4(x + MLP(n3 x)));
  * no step embedding; the iteration count is sampled per training step by the harness as
    1 + Poisson(log-normal) around --loop_k_mean, with truncated backpropagation through the last
    --loop_bp iterations (they use 8); a coda block and the LM head read the final state.

Every reported number used --loop_layers {1,2}, --loop_k 128, --loop_bp 8, --loop_k_mean 32 at lr 1e-4.
The paper's submitted rows were run WITHOUT the embedding scale (emb_scale=False); the camera-ready rows use
emb_scale=True (harness flag --loop_emb_scale), which changed the keyed-A5 result (see docs/loop_numbers.md).
"""
from models.dualmod.config import DualModConfig
from models.looped.looped_lm import LoopedLM


class RecurrentDepthLM(LoopedLM):
    """LoopedLM pinned to the recurrent-depth recipe."""

    def __init__(self, cfg: DualModConfig, n_unique: int = 2, k_max: int = 128, bp_iters: int = 8,
                 emb_scale: bool = True):
        super().__init__(cfg, n_unique=n_unique, k_max=k_max, inject=True, huginn=True,
                         bp_iters=bp_iters, emb_scale=emb_scale)
