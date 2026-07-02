"""Paired bootstrap CIs over test examples for pooled-R² gaps (review item R0).

Every headline reconstruction gap in the project (Gate-3 SSM-vs-baseline, E3 SSM-vs-CLT)
is a difference of pooled R² values computed on the *same* test examples, so the honest
uncertainty is a **paired** bootstrap: resample test examples with replacement, recompute
both pooled R² on the resample, take percentiles of the gap. Pooled R² follows the
project convention ``1 − Σ ss_res / Σ ss_tot`` with ``ss_tot`` against the per-depth mean
(held fixed across resamples — the mean is not re-estimated per resample).

Inputs are per-example sums of squares ``ss_res[N]`` / ``ss_tot[N]`` (each example's SSE
summed over layers × dims), which every decoder in the project can produce cheaply.
"""

from __future__ import annotations

import numpy as np

__all__ = ["per_example_ss", "pooled_r2", "paired_gap_ci"]


def per_example_ss(pred: np.ndarray, target: np.ndarray, mean: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Per-example residual / total sums of squares.

    ``pred``/``target`` are ``[N, L, D]``; ``mean`` is the per-depth baseline ``[L, D]``
    (or broadcastable). Returns ``(ss_res[N], ss_tot[N])`` summed over layers and dims.
    """
    ss_res = ((target - pred) ** 2).reshape(target.shape[0], -1).sum(axis=1)
    ss_tot = ((target - mean) ** 2).reshape(target.shape[0], -1).sum(axis=1)
    return ss_res, ss_tot


def pooled_r2(ss_res: np.ndarray, ss_tot: np.ndarray) -> float:
    """Project-convention pooled R² from per-example sums of squares."""
    tot = float(ss_tot.sum())
    return 1.0 - float(ss_res.sum()) / tot if tot > 0 else 0.0


def paired_gap_ci(
    ss_res_a: np.ndarray,
    ss_res_b: np.ndarray,
    ss_tot: np.ndarray,
    *,
    n_boot: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """Percentile CI for the pooled-R² gap ``R²(a) − R²(b)`` under paired resampling.

    The same bootstrap indices are applied to both arms (paired — the arms share test
    examples), so example-level difficulty cancels out of the gap distribution.
    """
    n = len(ss_tot)
    if not (len(ss_res_a) == len(ss_res_b) == n):
        raise ValueError("per-example arrays must share a length")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    ra = ss_res_a[idx].sum(axis=1)
    rb = ss_res_b[idx].sum(axis=1)
    tt = ss_tot[idx].sum(axis=1)
    gaps = (rb - ra) / tt  # R²_a − R²_b == (ss_res_b − ss_res_a) / ss_tot
    lo, hi = np.quantile(gaps, [alpha / 2, 1 - alpha / 2])
    point = pooled_r2(ss_res_a, ss_tot) - pooled_r2(ss_res_b, ss_tot)
    return {
        "gap": point,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "alpha": alpha,
        "n_boot": n_boot,
        "n_test": n,
        "boot_seed": seed,
        "r2_a": pooled_r2(ss_res_a, ss_tot),
        "r2_b": pooled_r2(ss_res_b, ss_tot),
        "significant": bool(lo > 0 or hi < 0),
    }
