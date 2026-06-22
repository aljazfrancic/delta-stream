"""Tests for the Phase 1 Step 5 AR(k) / Granger diagnostics."""

from __future__ import annotations

import numpy as np
from scipy import stats

from layer_dynamics.diagnostics.autoregress import (
    feasible_depths,
    fit_ar_pooled,
    granger_lag_grid,
    pca_project,
    scramble_per_depth,
    wilks_rao_f,
)


def _make_ar1(n=400, n_delta=8, d=6, seed=0):
    """A depth-series with a genuine AR(1) law Δ_ℓ = μ_ℓ + 0.7·R Δ_{ℓ-1} + ε."""
    rng = np.random.default_rng(seed)
    r = 0.7 * rng.standard_normal((d, d)) / np.sqrt(d)
    mu = rng.standard_normal((n_delta, d)) * 2.0  # per-depth intercepts
    dlt = np.empty((n, n_delta, d), dtype=np.float64)
    dlt[:, 0] = rng.standard_normal((n, d))
    for ell in range(1, n_delta):
        dlt[:, ell] = mu[ell] + dlt[:, ell - 1] @ r.T + 0.1 * rng.standard_normal((n, d))
    return dlt


def test_feasible_depths():
    assert feasible_depths(24, 1) == list(range(1, 24))
    assert feasible_depths(24, 20) == [20, 21, 22, 23]
    assert feasible_depths(5, 5) == []


def test_fit_ar_pooled_recovers_real_structure_and_dies_on_scramble():
    dlt = _make_ar1()
    tr, te = dlt[:320], dlt[320:]
    fit = fit_ar_pooled(tr, te, k=1, ranks=(2, 4, 6))
    assert fit.best_r2 > 0.5  # the planted AR(1) law is recovered out of sample

    # Scrambling the example axis per depth destroys the cross-depth law -> R²≈0.
    s_tr = scramble_per_depth(dlt[:320], seed=1)
    s_te = scramble_per_depth(dlt[320:], seed=2)
    s_fit = fit_ar_pooled(s_tr, s_te, k=1, ranks=(2, 4, 6))
    assert s_fit.best_r2 < 0.1


def test_fit_ar_pooled_per_depth_intercept_not_folded_into_kernel():
    # Large per-depth intercepts but NO AR dependence: a correct per-depth-intercept
    # fit must score ≈0 (the offsets are absorbed by μ_ℓ, not fitted by the kernel).
    rng = np.random.default_rng(3)
    n, n_delta, d = 400, 8, 6
    mu = rng.standard_normal((n_delta, d)) * 5.0
    dlt = mu[None] + 0.5 * rng.standard_normal((n, n_delta, d))
    fit = fit_ar_pooled(dlt[:320], dlt[320:], k=1, ranks=(2, 4, 6))
    assert abs(fit.best_r2) < 0.1


def test_wilks_rao_f_matches_partial_f_in_univariate_case():
    # For q=1 the Wilks/Rao F reduces exactly to the ordinary partial F-test.
    rng = np.random.default_rng(0)
    n = 200
    x_red = np.column_stack([np.ones(n), rng.standard_normal((n, 2))])  # intercept + 2
    x_add = rng.standard_normal((n, 1))
    y = (x_red @ rng.standard_normal((3, 1)) + 1.5 * x_add + rng.standard_normal((n, 1)))
    x_full = np.column_stack([x_red, x_add])

    def sscp(z):
        beta, *_ = np.linalg.lstsq(z, y, rcond=None)
        r = y - z @ beta
        return r.T @ r

    e_red, e_full = sscp(x_red), sscp(x_full)
    ve = n - x_full.shape[1]
    res = wilks_rao_f(e_full, e_red, p=1, vh=1, ve=ve)

    sse_r, sse_f = e_red.item(), e_full.item()
    f_ref = ((sse_r - sse_f) / 1) / (sse_f / ve)
    p_ref = float(stats.f.sf(f_ref, 1, ve))
    assert np.isclose(res["F"], f_ref, rtol=1e-6)
    assert np.isclose(res["p_value"], p_ref, rtol=1e-6)
    assert res["df1"] == 1 and np.isclose(res["df2"], ve)


def test_granger_grid_significant_on_ar_and_null_on_scramble():
    dlt = _make_ar1(n=600, n_delta=6, d=6)
    grid = granger_lag_grid(dlt, lags=(1, 2, 3), q_pca=5)
    nlp = grid["neg_log10_p"]
    # Lag-1 cells (ℓ≥1) should be overwhelmingly significant for a true AR(1).
    lag1 = nlp[1:, 0]
    assert np.nanmean(lag1) > 5.0  # −log10 p ≫ Bonferroni threshold

    null = granger_lag_grid(scramble_per_depth(dlt, seed=7), lags=(1, 2, 3), q_pca=5)
    # Under the scramble almost no cell clears even an uncorrected α=0.05.
    frac_sig = np.nanmean(null["neg_log10_p"] > -np.log10(0.05))
    assert frac_sig < 0.2


def test_pca_project_shape_and_variance_order():
    rng = np.random.default_rng(0)
    dlt = rng.standard_normal((50, 8, 20))
    proj = pca_project(dlt, q=4)
    assert proj.shape == (50, 8, 4)
    # Components are variance-ordered (first PC carries the most).
    v = proj.reshape(-1, 4).var(axis=0)
    assert v[0] >= v[1] >= v[2] >= v[3]


def test_scramble_preserves_marginals_and_uses_distinct_perms():
    rng = np.random.default_rng(0)
    dlt = rng.standard_normal((100, 5, 3))
    s = scramble_per_depth(dlt, seed=0)
    # Per-depth marginal (sorted rows) is preserved exactly — it's a permutation.
    for ell in range(dlt.shape[1]):
        a = np.sort(dlt[:, ell, 0])
        b = np.sort(s[:, ell, 0])
        np.testing.assert_allclose(a, b)
    # Different depths get different permutations (cross-depth structure broken).
    assert not np.allclose(s[:, 0, :], dlt[:, 0, :]) or not np.allclose(s[:, 1, :], dlt[:, 1, :])
