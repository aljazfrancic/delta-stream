"""Tests for the Phase 2 Step 1 across-depth stationarity diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from layer_dynamics.diagnostics.autoregress import (
    eval_kernel_on_band,
    feasible_depths,
    fit_ar_pooled,
    fit_delta_pca,
)
from layer_dynamics.diagnostics.stationarity import (
    band_depths,
    bootstrap_within_band,
    classify_stationarity,
    cross_band_r2,
    cross_band_r2_by_rank,
    fit_band_kernels,
    per_lag_frobenius,
    principal_angles_deg,
    relative_frobenius,
    select_k,
    top_subspaces,
)


def _make_banded_ar(r_map, *, n=500, n_delta=12, d=8, seed=0):
    """Depth-series with AR(1) law Δ_ℓ = μ_ℓ + R(ℓ)·Δ_{ℓ-1} + ε, R per target depth."""
    rng = np.random.default_rng(seed)
    mu = rng.standard_normal((n_delta, d)) * 2.0
    dlt = np.empty((n, n_delta, d))
    dlt[:, 0] = rng.standard_normal((n, d))
    for ell in range(1, n_delta):
        dlt[:, ell] = mu[ell] + dlt[:, ell - 1] @ r_map(ell).T + 0.1 * rng.standard_normal((n, d))
    return dlt


def _rand_map(d, scale, seed):
    rng = np.random.default_rng(seed)
    return scale * rng.standard_normal((d, d)) / np.sqrt(d)


# --- band partition + k selection ---------------------------------------------


def test_band_depths_pythia410m():
    bands = band_depths(24, 5)
    assert bands["early"] == [5, 6, 7]            # ⌊L/3⌋−k = 3 feasible early depths
    assert bands["middle"] == list(range(8, 16))
    assert bands["late"] == list(range(16, 24))


def test_band_depths_feasibility_filters_early():
    # k=7 would leave only one early depth (<3) — the cap exists to prevent this.
    bands = band_depths(24, 7)
    assert bands["early"] == [7]  # only ℓ=7 feasible; cap in select_k keeps k≤5


def _granger_df(n_delta, sig_up_to, lags=(1, 2, 3, 5, 10, 15, 20)):
    rows = []
    for j in lags:
        for ell in feasible_depths(n_delta, j):
            nlp = 10.0 if j <= sig_up_to else 0.0
            rows.append({"depth": ell, "lag": j, "neg_log10_p": nlp})
    return pd.DataFrame(rows)


def test_select_k_caps_when_granger_reaches_max_lag():
    # Significant to the largest lag (the real Step 5 case) -> capped to ⌊L/3⌋−3=5.
    sel = select_k(_granger_df(24, sig_up_to=20), 24)
    assert sel["granger_selected_lag"] == 20
    assert sel["feasibility_cap"] == 5
    assert sel["k"] == 5 and sel["capped"] is True
    assert sel["n_granger_cells"] == 112  # Σ_j (L−j) over the sweep


def test_select_k_uses_granger_lag_when_under_cap():
    sel = select_k(_granger_df(24, sig_up_to=3), 24)
    assert sel["granger_selected_lag"] == 3
    assert sel["k"] == 3 and sel["capped"] is False


# --- distance / subspace primitives -------------------------------------------


def test_relative_frobenius_bounds_and_symmetry():
    a = np.random.default_rng(0).standard_normal((10, 6))
    assert relative_frobenius(a, a) == 0.0
    assert np.isclose(relative_frobenius(a, -a), 2.0)        # equal norm, opposite
    b = np.random.default_rng(1).standard_normal((10, 6))
    assert np.isclose(relative_frobenius(a, b), relative_frobenius(b, a))


def test_per_lag_frobenius_splits_blocks():
    k, q, d = 3, 4, 5
    a = np.zeros((k * q, d))
    b = a.copy()
    b[q : 2 * q] = 1.0  # perturb only the lag-2 block
    dists = per_lag_frobenius(a, b, k)
    assert len(dists) == k
    assert dists[0] == 0.0 and dists[2] == 0.0
    assert dists[1] > 0.0


def test_principal_angles_identical_and_orthogonal():
    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.standard_normal((10, 6)))
    a, b = q[:, :3], q[:, 3:6]
    assert np.allclose(principal_angles_deg(a, a), 0.0, atol=1e-6)      # same subspace
    assert np.allclose(principal_angles_deg(a, b), 90.0, atol=1e-4)     # orthogonal


def test_top_subspaces_shapes_and_orthonormal():
    coef = np.random.default_rng(0).standard_normal((20, 8))
    u, v = top_subspaces(coef, 4)
    assert u.shape == (20, 4) and v.shape == (8, 4)
    np.testing.assert_allclose(u.T @ u, np.eye(4), atol=1e-6)
    np.testing.assert_allclose(v.T @ v, np.eye(4), atol=1e-6)


# --- AR fit on a depth band + cross-band evaluation ---------------------------


def test_fit_ar_pooled_depths_subset_runs_and_eval_matches():
    dlt = _make_banded_ar(lambda _ell: _rand_map(8, 0.7, 99))
    tr, te = dlt[:400], dlt[400:]
    basis = fit_delta_pca(tr, 8)  # q=d: lossless coordinate system
    depths = [8, 9, 10, 11]
    fit = fit_ar_pooled(tr, te, k=1, ranks=(8,), input_basis=basis, depths=depths)
    assert fit.depths == tuple(depths)
    # The within-band R² recomputed via eval_kernel_on_band matches the fit's own R².
    r2_via_eval = eval_kernel_on_band(fit.coef_at_rank(8), tr, te, depths, 1, basis)
    assert np.isclose(r2_via_eval, fit.r2[8], atol=1e-8)


def test_fit_ar_pooled_rejects_infeasible_depths():
    dlt = _make_banded_ar(lambda _ell: _rand_map(8, 0.5, 1), n_delta=12)
    try:
        fit_ar_pooled(dlt[:50], dlt[50:], k=5, depths=[3], input_pcs=8)
    except ValueError as e:
        assert "infeasible" in str(e)
    else:
        raise AssertionError("expected ValueError for depth < k")


def test_stationary_law_transfers_across_bands():
    # One shared AR map at every depth -> cross-band R² ≈ within-band R².
    dlt = _make_banded_ar(lambda _ell: _rand_map(8, 0.7, 7))
    tr, te = dlt[:400], dlt[400:]
    basis = fit_delta_pca(tr, 8)
    bands = band_depths(12, 1)
    kernels = fit_band_kernels(tr, te, 1, 8, basis, bands)
    cross = cross_band_r2(kernels, tr, te, 1, basis)
    # Every off-diagonal keeps almost all of the within-band fit.
    for fnm in bands:
        for enm in bands:
            assert cross[fnm][enm] / cross[enm][enm] > 0.8


def test_phase_transition_breaks_cross_band_transfer():
    # R0 on early+middle target depths, a different R1 on late -> late kernel diverges.
    r0, r1 = _rand_map(8, 0.7, 11), _rand_map(8, 0.7, 22)
    dlt = _make_banded_ar(lambda ell: r0 if ell < 8 else r1)
    tr, te = dlt[:400], dlt[400:]
    basis = fit_delta_pca(tr, 8)
    bands = band_depths(12, 1)
    kernels = fit_band_kernels(tr, te, 1, 8, basis, bands)
    cross = cross_band_r2(kernels, tr, te, 1, basis)
    # early kernel predicts late much worse than late predicts itself.
    assert cross["early"]["late"] < cross["late"]["late"] - 0.2
    # and the kernel distance is concentrated at the middle|late boundary.
    d_em = relative_frobenius(kernels["early"].coef_r, kernels["middle"].coef_r)
    d_ml = relative_frobenius(kernels["middle"].coef_r, kernels["late"].coef_r)
    assert d_ml > d_em


def test_cross_band_r2_by_rank_keeps_diagonal_positive():
    # On a stationary law the diagonal stays positive across ranks (sanity of the sweep).
    dlt = _make_banded_ar(lambda _ell: _rand_map(8, 0.7, 7))
    tr, te = dlt[:400], dlt[400:]
    basis = fit_delta_pca(tr, 8)
    bands = band_depths(12, 1)
    by_rank = cross_band_r2_by_rank(tr, te, 1, (2, 4, 8), basis, bands)
    assert set(by_rank) == {2, 4, 8}
    for r in (2, 4, 8):
        for nm in bands:
            assert by_rank[r][nm][nm] > 0.0


def test_bootstrap_spread_smaller_than_real_band_difference():
    r0, r1 = _rand_map(8, 0.7, 11), _rand_map(8, 0.7, 22)
    dlt = _make_banded_ar(lambda ell: r0 if ell < 8 else r1)
    tr, te = dlt[:400], dlt[400:]
    basis = fit_delta_pca(tr, 8)
    bands = band_depths(12, 1)
    kernels = fit_band_kernels(tr, te, 1, 8, basis, bands)
    spread = bootstrap_within_band(
        tr, te, bands["late"], 1, 8, basis, kernels["late"].coef_r, n_boot=20, seed=0
    )
    between = relative_frobenius(kernels["middle"].coef_r, kernels["late"].coef_r)
    # The genuine between-band difference exceeds within-band sampling noise.
    assert between > np.percentile(spread, 95)


# --- decision matrix ----------------------------------------------------------


def _angles(mean):
    return {"output": {"mean_deg": mean, "median_deg": mean, "max_deg": mean},
            "input": {"mean_deg": mean, "median_deg": mean, "max_deg": mean}}


def test_classify_stationary():
    cross = {b: {c: (1.0 if b == c else 0.97) for c in ("early", "middle", "late")}
             for b in ("early", "middle", "late")}
    fro = {"early-middle": 0.1, "middle-late": 0.1, "early-late": 0.12}
    angles = {p: _angles(8.0) for p in fro}
    boot = {p: {"separated": False} for p in fro}
    out = classify_stationarity(cross, fro, angles, boot)
    assert out["verdict"] == "stationary"
    assert out["phase3_branch"].startswith("A")


def test_classify_phase_transition():
    cross = {
        "early": {"early": 1.0, "middle": 0.95, "late": 0.4},
        "middle": {"early": 0.95, "middle": 1.0, "late": 0.4},
        "late": {"early": 0.4, "middle": 0.4, "late": 1.0},
    }
    fro = {"early-middle": 0.1, "middle-late": 0.6, "early-late": 0.6}
    angles = {"early-middle": _angles(5.0), "middle-late": _angles(60.0),
              "early-late": _angles(60.0)}
    boot = {"early-middle": {"separated": False}, "middle-late": {"separated": True},
            "early-late": {"separated": True}}
    out = classify_stationarity(cross, fro, angles, boot)
    assert out["verdict"] == "phase_transition"
    assert out["segment_boundary"] == "middle|late"


def test_classify_smooth_drift():
    cross = {
        "early": {"early": 1.0, "middle": 0.8, "late": 0.5},
        "middle": {"early": 0.8, "middle": 1.0, "late": 0.8},
        "late": {"early": 0.5, "middle": 0.8, "late": 1.0},
    }
    fro = {"early-middle": 0.3, "middle-late": 0.3, "early-late": 0.55}
    angles = {p: _angles(40.0) for p in fro}
    boot = {p: {"separated": True} for p in fro}
    out = classify_stationarity(cross, fro, angles, boot)
    assert out["verdict"] == "smooth_drift"
    assert out["phase3_branch"].startswith("B")
