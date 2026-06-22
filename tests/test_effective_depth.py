"""Tests for the Phase 2 Step 2 effective-kernel-depth diagnostics."""

from __future__ import annotations

import numpy as np

from layer_dynamics.diagnostics.autoregress import fit_delta_pca
from layer_dynamics.diagnostics.effective_depth import (
    band_r2_curve,
    depth_r2_curve,
    feasible_k,
    incremental_over_lag1,
    kstar_from_curve,
)


def test_feasible_k_caps_at_depth():
    assert feasible_k(3, (1, 2, 3, 5, 10)) == [1, 2, 3]
    assert feasible_k(24, (1, 2, 3, 5, 10, 15, 20)) == [1, 2, 3, 5, 10, 15, 20]
    assert feasible_k(1, (1, 2, 3)) == [1]


def test_kstar_saturating_declining_flat():
    assert kstar_from_curve({1: 0.5, 2: 0.9, 3: 0.92, 5: 0.93}) == 2   # saturates at 2
    assert kstar_from_curve({1: 0.9, 2: 0.85, 3: 0.8}) == 1            # declines -> peak at 1
    assert kstar_from_curve({1: 0.9, 2: 0.9, 3: 0.9}) == 1            # flat -> smallest


def test_incremental_over_lag1():
    assert np.isclose(incremental_over_lag1({1: 0.8, 2: 0.9, 3: 0.85}), 0.1)
    assert incremental_over_lag1({1: 0.9}) == 0.0


def _make_ar(lag, *, n=800, n_delta=10, d=8, seed=0):
    """Planted process Δ_ℓ = μ_ℓ + R·Δ_{ℓ-lag} + ε (direct on Δ, no ±v_ℓ tautology).

    ``lag=1`` is AR(1) (k*=1). ``lag=2`` depends *only* on the lag-2 update — the
    immediate predecessor is uninformative — so a k=1 fit must underfit and k* ≥ 2.
    """
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((d, d)))  # orthogonal -> variance-preserving
    r = 0.9 * q
    mu = rng.standard_normal((n_delta, d)) * 2.0
    dlt = np.empty((n, n_delta, d))
    for j in range(lag):
        dlt[:, j] = rng.standard_normal((n, d))
    for ell in range(lag, n_delta):
        dlt[:, ell] = mu[ell] + dlt[:, ell - lag] @ r.T + 0.1 * rng.standard_normal((n, d))
    return dlt


def test_depth_r2_curve_ar1_saturates_at_one():
    dlt = _make_ar(1)
    tr, te = dlt[:600], dlt[600:]
    basis = fit_delta_pca(tr, 8)
    curve = depth_r2_curve(tr, te, 6, sweep=(1, 2, 3, 5), rank=8, basis=basis, input_pcs=8)
    assert set(curve) == {1, 2, 3, 5}
    assert curve[1] > 0.5
    assert kstar_from_curve(curve) == 1                      # one lag suffices
    assert incremental_over_lag1(curve) < 0.1               # lags >=2 add ~nothing


def test_depth_r2_curve_lag2_needs_two_lags():
    # Power check: when the law depends only on lag 2, k* must detect it (not stuck at 1).
    dlt = _make_ar(2)
    tr, te = dlt[:600], dlt[600:]
    basis = fit_delta_pca(tr, 8)
    curve = depth_r2_curve(tr, te, 6, sweep=(1, 2, 3, 5), rank=8, basis=basis, input_pcs=8)
    assert curve[2] > curve[1] + 0.1                         # lag 2 genuinely helps
    assert kstar_from_curve(curve) >= 2


def test_band_r2_curve_caps_k_at_min_depth():
    dlt = _make_ar(1, n_delta=12)
    tr, te = dlt[:400], dlt[400:]
    basis = fit_delta_pca(tr, 8)
    # Band {5,6,7}: k must stay <= 5 to keep every target feasible.
    curve = band_r2_curve(tr, te, [5, 6, 7], sweep=(1, 2, 3, 5, 10), rank=8, basis=basis,
                          input_pcs=8)
    assert set(curve) == {1, 2, 3, 5}
    assert max(curve) == 5
