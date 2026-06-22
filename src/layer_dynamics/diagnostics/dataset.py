"""Load collected residual streams and derive the quantities the diagnostics fit.

``Δ_ℓ = v_{ℓ+1} − v_ℓ`` is computed here rather than stored (Step 2 keeps only
``v``). A single example-level train/test split is reused across all layer pairs
so held-out R² values are comparable across depth.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_residual_stream(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(residual_stream[N, L+1, D], relation_class[N], relation[N])``."""
    import h5py

    with h5py.File(path, "r") as f:
        rs = f["residual_stream"][:].astype(np.float32)
        relation_class = f["relation_class"].asstr()[:]
        relation = f["relation"].asstr()[:]
    return rs, relation_class, relation


def deltas(rs: np.ndarray) -> np.ndarray:
    """Layer-to-layer updates ``Δ[:, ℓ] = v_{ℓ+1} − v_ℓ`` for ℓ in 0..L-1.

    Input ``[N, L+1, D]`` -> output ``[N, L, D]``.
    """
    return rs[:, 1:, :] - rs[:, :-1, :]


def train_test_split_idx(
    n: int, *, test_frac: float = 0.2, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic example-level split into (train_idx, test_idx)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = int(round(n * test_frac))
    return perm[n_test:], perm[:n_test]
