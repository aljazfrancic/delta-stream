"""Tests for the Phase 1 diagnostics (Step 3)."""

from __future__ import annotations

import numpy as np

from layer_dynamics.diagnostics.cca import fit_cca, regularized_cca, shrink_cov
from layer_dynamics.diagnostics.dataset import deltas, train_test_split_idx
from layer_dynamics.diagnostics.lowrank_ridge import fit_lowrank_ridge, pooled_r2
from layer_dynamics.diagnostics.tuned_lens import tl_implied_deltas


def test_pooled_r2_perfect_and_mean():
    rng = np.random.default_rng(0)
    y = rng.standard_normal((50, 8))
    ref = np.broadcast_to(y.mean(0), y.shape)
    assert pooled_r2(y, y, ref) == 1.0                      # perfect prediction
    assert abs(pooled_r2(y, ref, ref)) < 1e-12              # mean predictor -> 0


def test_deltas_shape_and_values():
    rs = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3)  # [N=2, L+1=4, D=3]
    d = deltas(rs)
    assert d.shape == (2, 3, 3)
    np.testing.assert_allclose(d[0, 0], rs[0, 1] - rs[0, 0])


def test_train_test_split_disjoint():
    tr, te = train_test_split_idx(100, test_frac=0.2, seed=1)
    assert len(tr) == 80 and len(te) == 20
    assert set(tr).isdisjoint(set(te))
    assert set(tr) | set(te) == set(range(100))


def test_fit_lowrank_ridge_recovers_low_rank_signal():
    rng = np.random.default_rng(0)
    d, r_true = 40, 4
    m = rng.standard_normal((d, r_true)) @ rng.standard_normal((r_true, d))  # rank-4 map
    x = rng.standard_normal((400, d))
    y = x @ m.T
    res = fit_lowrank_ridge(x[:320], y[:320], x[320:], y[320:], ranks=(2, 4, 16))
    assert res[4] > 0.9            # rank >= true rank recovers the signal
    assert res[4] > res[2]         # too-low rank underfits
    assert "full" in res and "alpha" in res


class _IdentityLens:
    """Fake lens whose translator is the identity, so r̂_ℓ = v_ℓ."""

    def transform_hidden(self, h, idx):  # noqa: ARG002
        return h


def test_tl_implied_deltas_identity_equals_true_deltas():
    rng = np.random.default_rng(0)
    rs = rng.standard_normal((5, 7, 6)).astype(np.float32)  # [N, L+1, D]
    tl = tl_implied_deltas(_IdentityLens(), rs)
    # Identity translator -> implied update is the true update exactly.
    np.testing.assert_allclose(tl, deltas(rs), atol=1e-5)


def test_shrink_cov_preserves_trace_and_improves_conditioning():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((6, 30))
    cov = a @ a.T  # 6x6, rank-deficient-ish, ill-conditioned
    shr = shrink_cov(cov, 0.3)
    # Target has the same trace, so shrinkage preserves total variance.
    assert abs(np.trace(shr) - np.trace(cov)) < 1e-8
    # ... while pulling the eigenvalues together (better conditioned).
    assert np.linalg.cond(shr) < np.linalg.cond(cov)
    np.testing.assert_allclose(shrink_cov(cov, 0.0), cov)  # gamma=0 is identity


def test_fit_cca_recovers_planted_shared_latent():
    rng = np.random.default_rng(0)
    n = 600
    z = rng.standard_normal((n, 1))  # shared latent driving one direction each
    x = rng.standard_normal((n, 10))
    y = rng.standard_normal((n, 8))
    x[:, 0] += 3.0 * z[:, 0]
    y[:, 0] += 3.0 * z[:, 0]
    _, _, corr = fit_cca(x, y, n_components=3, gamma=0.05)
    assert corr[0] > 0.7  # the shared latent is found
    assert corr[2] < 0.4  # nothing shared beyond it
    assert corr[0] >= corr[1] >= corr[2]  # SVD order is descending


def test_regularized_cca_heldout_collapses_without_real_structure():
    # p ≈ N_train makes in-sample CCA degenerate (correlations inflate to ~1),
    # but with no true shared structure the held-out correlations collapse.
    rng = np.random.default_rng(1)
    x = rng.standard_normal((100, 45))
    y = rng.standard_normal((100, 45))
    res = regularized_cca(x[:60], y[:60], x[60:], y[60:], n_components=5, gamma=0.05)
    assert res["train"][0] > 0.8  # in-sample inflated by the degeneracy
    assert res["test"][0] < 0.6   # held-out exposes the absence of structure


def test_regularized_cca_heldout_survives_real_structure():
    rng = np.random.default_rng(2)
    n = 500
    z = rng.standard_normal((n, 2))
    x = rng.standard_normal((n, 12))
    y = rng.standard_normal((n, 12))
    x[:, :2] += 2.5 * z
    y[:, :2] += 2.5 * z  # two genuinely shared directions
    res = regularized_cca(x[:400], y[:400], x[400:], y[400:], n_components=4, gamma=0.1)
    assert res["test"][0] > 0.6   # real shared structure generalises
    assert res["test"][3] < 0.4   # but only the two planted directions do
