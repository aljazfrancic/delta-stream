"""Tests for the Phase 2 Step 3 cross-position kernel diagnostics."""

from __future__ import annotations

import numpy as np

from layer_dynamics.diagnostics.dataset import train_test_split_idx
from layer_dynamics.diagnostics.token_position import (
    classify_position_invariance,
    cross_position_r2,
    fit_position_kernels,
    kernel_comparisons,
    offdiag_retention,
    shared_position_basis,
)


def _ar1(rot, *, n=600, n_delta=8, d=6, seed=0):
    """Planted AR(1) on Δ with transition ``rot`` (direct on Δ, no ±v tautology)."""
    rng = np.random.default_rng(seed)
    mu = rng.standard_normal((n_delta, d)) * 1.5
    dlt = np.empty((n, n_delta, d))
    dlt[:, 0] = rng.standard_normal((n, d))
    for ell in range(1, n_delta):
        dlt[:, ell] = mu[ell] + dlt[:, ell - 1] @ rot.T + 0.1 * rng.standard_normal((n, d))
    return dlt


def _rot(d, seed):
    q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((d, d)))
    return 0.9 * q


def test_same_dynamics_transfers_across_positions():
    # Two "positions" with the *same* transition -> off-diagonal R² ≈ diagonal.
    rot = _rot(6, 0)
    deltas_by_pos = {"a": _ar1(rot, seed=1), "b": _ar1(rot, seed=2)}
    n = deltas_by_pos["a"].shape[0]
    tr, te = train_test_split_idx(n, seed=0)
    basis = shared_position_basis(deltas_by_pos, tr, 6)
    depths = list(range(1, 8))
    kernels = fit_position_kernels(deltas_by_pos, tr, te, 1, 6, basis, depths)
    conf = cross_position_r2(kernels, deltas_by_pos, tr, te, 1, basis, depths)
    summary = offdiag_retention(conf)
    assert summary["mean_diagonal_r2"] > 0.5
    assert summary["worst_retention"] > 0.85          # foreign kernel keeps ~all R²
    verdict = classify_position_invariance(summary, kernel_comparisons(kernels, 1, 6))
    assert verdict["verdict"] == "position_invariant"


def test_different_dynamics_is_position_specific():
    # Different (near-orthogonal) transitions -> foreign kernel collapses off-diagonal.
    deltas_by_pos = {"a": _ar1(_rot(6, 0), seed=1), "b": _ar1(_rot(6, 99), seed=2)}
    n = deltas_by_pos["a"].shape[0]
    tr, te = train_test_split_idx(n, seed=0)
    basis = shared_position_basis(deltas_by_pos, tr, 6)
    depths = list(range(1, 8))
    kernels = fit_position_kernels(deltas_by_pos, tr, te, 1, 6, basis, depths)
    conf = cross_position_r2(kernels, deltas_by_pos, tr, te, 1, basis, depths)
    summary = offdiag_retention(conf)
    assert summary["mean_diagonal_r2"] > 0.5          # each fits itself well
    assert summary["worst_retention"] < 0.5           # but not each other
    verdict = classify_position_invariance(summary, kernel_comparisons(kernels, 1, 6))
    assert verdict["verdict"] == "query_position_specific"


def test_thin_signal_when_no_structure():
    rng = np.random.default_rng(3)
    deltas_by_pos = {
        "a": rng.standard_normal((400, 6, 5)),
        "b": rng.standard_normal((400, 6, 5)),
    }
    tr, te = train_test_split_idx(400, seed=0)
    basis = shared_position_basis(deltas_by_pos, tr, 5)
    depths = list(range(1, 6))
    kernels = fit_position_kernels(deltas_by_pos, tr, te, 1, 5, basis, depths)
    conf = cross_position_r2(kernels, deltas_by_pos, tr, te, 1, basis, depths)
    summary = offdiag_retention(conf)
    verdict = classify_position_invariance(summary, kernel_comparisons(kernels, 1, 5))
    assert verdict["verdict"] == "thin_signal"


def test_offdiag_retention_shapes():
    conf = {"a": {"a": 0.8, "b": 0.4}, "b": {"a": 0.2, "b": 0.6}}
    s = offdiag_retention(conf)
    assert np.isclose(s["mean_diagonal_r2"], 0.7)
    assert np.isclose(s["mean_offdiagonal_r2"], 0.3)
    # retention a->b = conf[a][b]/conf[b][b] = 0.4/0.6; b->a = 0.2/0.8 = 0.25 (worst).
    assert np.isclose(s["worst_retention"], 0.25)
