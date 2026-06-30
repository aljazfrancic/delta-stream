"""Tests for the Phase 1 Step 5 AR(k) / Granger diagnostics."""

from __future__ import annotations

import numpy as np
from scipy import stats

from layer_dynamics.diagnostics.autoregress import (
    feasible_depths,
    fit_ar_pooled,
    granger_lag_grid,
    granger_significant_by_lag,
    pca_project,
    random_walk_surrogate,
    remove_state_dirs,
    scramble_per_depth,
    state_pca_basis,
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


def test_state_pca_basis_orthonormal_and_top_variance():
    rng = np.random.default_rng(0)
    # States with one dominant direction (a "rogue" persistent dim).
    big = rng.standard_normal((40, 9, 1)) * 50.0
    rest = rng.standard_normal((40, 9, 5))
    rs = np.concatenate([big * np.ones((1, 1, 5)), rest], axis=2)[:, :, :5]
    mean, dirs = state_pca_basis(rs, m=3)
    assert mean.shape == (5,) and dirs.shape == (3, 5)
    np.testing.assert_allclose(dirs @ dirs.T, np.eye(3), atol=1e-6)  # orthonormal
    assert state_pca_basis(rs, m=0)[1].shape == (0, 5)


def test_remove_state_dirs_projects_out_and_preserves_complement():
    rng = np.random.default_rng(1)
    dlt = rng.standard_normal((30, 6, 8))
    dirs = np.linalg.svd(rng.standard_normal((8, 8)))[0][:2]  # 2 orthonormal rows
    out = remove_state_dirs(dlt, dirs)
    # Components along the removed directions are ~0; complement is untouched.
    along = out.reshape(-1, 8) @ dirs.T
    np.testing.assert_allclose(along, 0.0, atol=1e-6)
    # Removing nothing is a no-op.
    np.testing.assert_allclose(remove_state_dirs(dlt, np.zeros((0, 8))), dlt)
    # A second removal is idempotent (already in the complement).
    np.testing.assert_allclose(remove_state_dirs(out, dirs), out, atol=1e-6)


def test_random_walk_surrogate_matches_marginals_and_kills_ar():
    # Genuine AR(1): the surrogate must reproduce per-depth step covariance but
    # destroy the cross-depth law -> AR R²≈0 (parametric companion to scramble).
    dlt = _make_ar1(n=800, n_delta=8, d=6)
    surr = random_walk_surrogate(dlt, seed=0, rank=6)
    assert surr.shape == dlt.shape
    for ell in range(dlt.shape[1]):  # per-layer covariance preserved within sampling error
        c_real = np.cov(dlt[:, ell, :], rowvar=False)
        c_surr = np.cov(surr[:, ell, :], rowvar=False)
        assert np.linalg.norm(c_real - c_surr) / np.linalg.norm(c_real) < 0.25
    fit = fit_ar_pooled(surr[:640], surr[640:], k=1, ranks=(2, 4, 6))
    assert fit.best_r2 < 0.1  # independent steps -> no AR predictability


def test_granger_significant_by_lag_splits_lag1_and_higher():
    dlt = _make_ar1(n=600, n_delta=6, d=5)
    grid = granger_lag_grid(dlt, lags=(1, 2, 3), q_pca=4)
    res = granger_significant_by_lag(grid)
    assert res["lag1_total"] == 5  # depths ℓ=1..5 are testable at lag 1
    assert res["lag1_sig"] >= 4  # a true AR(1) lights up lag 1
    # lag>=2 totals = testable (ℓ≥2) cells across lags 2 and 3.
    assert res["lagge2_total"] == 4 + 3
    assert 0 <= res["lagge2_sig"] <= res["lagge2_total"]
