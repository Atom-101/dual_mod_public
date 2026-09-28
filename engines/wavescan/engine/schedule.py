"""Schedule pool (megakernel.md §10).

A schedule is a static list of cells [(pos0, c, n_sweeps), ...] with
sum(c) == T, drawn once at init and baked into a captured graph. `exact
sequential` (all c=1) is the always-available oracle path through the same
cell code.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class WaveScanSchedule:
    """Defaults are the MEASURED frontier, not the spec's guess: the
    pre-registered pool (c8/K2, c4/K1) failed T-K7 by 30x (Δ≈+0.12 on the
    trained gate-0.5 checkpoint — K=1 means raw intra reads, the catastrophic
    m=1 regime of the DEQ depth study). The (c,K) grid (bench/k7_grid.py)
    shows K/c = 0.5 with c >= 8 is exact to bf16 noise at half the serial
    ticks; K/c = 0.25 costs +0.02, K=1 costs +0.3..0.6."""
    chunk_probs: dict = field(default_factory=lambda: {8: 0.2, 16: 0.4, 32: 0.4})
    sweeps: dict = field(default_factory=lambda: {8: 4, 16: 8, 32: 16})
    pool_size: int = 8
    random_offset: bool = True
    seed: int = 0

    def __post_init__(self):
        assert abs(sum(self.chunk_probs.values()) - 1.0) < 1e-9
        for c in self.chunk_probs:
            assert c in self.sweeps, f"no sweep count for chunk size {c}"

    def _sweeps_for(self, c: int) -> int:
        # truncated chunks fall back to exactness (K=c is exact by nilpotency)
        return min(self.sweeps.get(c, c), c)

    def draw(self, T: int, rng: np.random.Generator) -> list:
        sizes = np.array(sorted(self.chunk_probs))
        probs = np.array([self.chunk_probs[c] for c in sizes])
        cells, pos = [], 0
        if self.random_offset:
            off = int(rng.integers(0, int(sizes.max())))
            if off > 0:
                cells.append((0, off, self._sweeps_for(off)))
                pos = off
        while pos < T:
            c = int(rng.choice(sizes, p=probs))
            c = min(c, T - pos)
            cells.append((pos, c, self._sweeps_for(c)))
            pos += c
        return cells

    def draw_pool(self, T: int) -> list:
        rng = np.random.default_rng(self.seed)
        return [self.draw(T, rng) for _ in range(self.pool_size)]


def exact_sequential(T: int) -> list:
    """The oracle schedule: every position its own exact cell."""
    return [(j, 1, 1) for j in range(T)]


def uniform_chunks(T: int, c: int, n_sweeps: int | None = None) -> list:
    """Fixed-size chunks; n_sweeps=None means K=c (exact by nilpotency)."""
    cells = []
    for pos in range(0, T, c):
        cc = min(c, T - pos)
        cells.append((pos, cc, cc if n_sweeps is None else min(n_sweeps, cc)))
    return cells
