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


def group_disjoint_split_idx(
    groups: np.ndarray, *, test_frac: float = 0.2, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Group-disjoint example split: no group (e.g. CounterFact subject) straddles
    train and test (review item R5 — the plain split shares entities across the
    boundary because subjects/objects recur).

    Whole groups are assigned to test in shuffled order until at least
    ``test_frac`` of the *examples* are covered; both index arrays come back
    shuffled. Groups are compared as strings.
    """
    groups = np.asarray(groups).astype(str)
    n = len(groups)
    rng = np.random.default_rng(seed)
    uniq = rng.permutation(np.unique(groups))
    target = int(round(n * test_frac))
    test_groups: set[str] = set()
    covered = 0
    for g in uniq:
        if covered >= target:
            break
        test_groups.add(g)
        covered += int((groups == g).sum())
    is_test = np.isin(groups, sorted(test_groups))
    test_idx = rng.permutation(np.flatnonzero(is_test))
    train_idx = rng.permutation(np.flatnonzero(~is_test))
    return train_idx, test_idx


def entity_split_idx(
    n: int, parquet_path: str | Path, column: str, *, test_frac: float = 0.2, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Entity-disjoint split keyed on a ``filtered_triples.parquet`` column (R5).

    The h5 stores rows in parquet order (``example_id`` monotone), so the parquet
    column aligns with activation rows directly. ``column='object'`` is the split
    that matters on CounterFact: only ~205 unique objects cover the 1555 examples
    ("French" alone appears 163×), so a random split shares answers across the
    boundary; subjects are ~unique (1534/1555) and barely differ from random.
    """
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    if len(df) != n:
        raise ValueError(f"parquet rows ({len(df)}) != activation rows ({n})")
    return group_disjoint_split_idx(df[column].to_numpy(), test_frac=test_frac, seed=seed)
