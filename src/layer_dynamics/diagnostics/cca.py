"""Regularised (shrinkage) CCA backstop for shared-subspace structure (Phase 1 Step 4).

Step 3's ridge asks the directional question "can a linear map predict Δ_ℓ from
v_ℓ?". CCA asks the weaker, symmetric one: "is there a shared linear subspace
between v_ℓ and Δ_ℓ at all?" — structure the regression could miss if it is
symmetric (a shared latent variable driving both) rather than predictive. The
plan's interpretation key reads the two together: high R² *and* high canonical
correlation means the linear predictability is real; low R² *but* high canonical
correlation means ridge is underfitting genuine (possibly symmetric) structure.

D = 1024 with only ~1244 training examples makes the within-block covariances
ill-conditioned, so raw CCA is degenerate: its in-sample canonical correlations
saturate near 1 by chance. Two defences, both applied here:

* **Shrinkage** of each within-block covariance toward a scaled identity
  (Ledoit–Wolf-style target ``(tr C / p) I``) before whitening.
* **Held-out evaluation.** Canonical directions are fit on the train split and
  the correlations recomputed on the *test* split — the same split Step 3 uses,
  so they are directly comparable to the ridge held-out R². Held-out
  correlations cannot be inflated by overfitting; an in-sample correlation near
  1 whose held-out value collapses is exactly the degeneracy this guards
  against, so both are reported.
"""

from __future__ import annotations

import numpy as np


def shrink_cov(cov: np.ndarray, gamma: float) -> np.ndarray:
    """Shrink a covariance toward ``(tr cov / p) I`` by fraction ``gamma``.

    ``gamma=0`` is the raw covariance; ``gamma=1`` is the spherical target. The
    target has the same trace as ``cov``, so shrinkage changes the conditioning
    without changing total variance.
    """
    p = cov.shape[0]
    mu = float(np.trace(cov)) / p
    return (1.0 - gamma) * cov + gamma * mu * np.eye(p, dtype=cov.dtype)


def _inv_sqrt(cov: np.ndarray) -> np.ndarray:
    """Symmetric inverse square root ``cov^{-1/2}`` via eigendecomposition.

    Eigenvalues are floored at a small positive value so a (numerically)
    semidefinite shrunk covariance still yields a finite whitener.
    """
    w, u = np.linalg.eigh(cov)
    w = np.clip(w, 1e-12, None)
    return (u / np.sqrt(w)) @ u.T


def fit_cca(
    x: np.ndarray, y: np.ndarray, *, n_components: int, gamma: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Regularised CCA on ``x`` (N×p) and ``y`` (N×q).

    Returns ``(A, B, corr)``: canonical directions ``A`` (p×k) and ``B`` (q×k),
    and the in-sample canonical correlations ``corr`` (k,), for
    ``k = min(n_components, p, q)``. The directions map *centred* data to
    canonical variates, i.e. variate ``i`` is ``(x - x.mean(0)) @ A[:, i]``.
    """
    xc = x - x.mean(axis=0)
    yc = y - y.mean(axis=0)
    n = xc.shape[0]
    sxx = shrink_cov(xc.T @ xc / (n - 1), gamma)
    syy = shrink_cov(yc.T @ yc / (n - 1), gamma)
    sxy = xc.T @ yc / (n - 1)

    wx = _inv_sqrt(sxx)  # symmetric, so wx.T == wx
    wy = _inv_sqrt(syy)
    m = wx @ sxy @ wy
    u, s, vt = np.linalg.svd(m, full_matrices=False)

    k = min(n_components, m.shape[0], m.shape[1])
    a = wx @ u[:, :k]
    b = wy @ vt[:k].T
    corr = np.clip(s[:k], 0.0, 1.0)
    return a, b, corr


def heldout_correlations(
    x_test: np.ndarray,
    y_test: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    *,
    means: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """Per-component Pearson correlation of held-out canonical variates.

    ``means = (x_mean, y_mean)`` are the *train* means used to centre the test
    data, so the held-out variates live in the same frame as the fitted
    directions. Returns a length-``k`` array; values are signed, so a near-zero
    or negative entry against a large in-sample correlation flags a
    non-generalising (overfit) canonical direction.
    """
    x_mean, y_mean = means
    xv = (x_test - x_mean) @ a  # [N_te, k]
    yv = (y_test - y_mean) @ b
    k = a.shape[1]
    out = np.empty(k)
    for i in range(k):
        out[i] = np.corrcoef(xv[:, i], yv[:, i])[0, 1]
    return out


def regularized_cca(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_te: np.ndarray,
    y_te: np.ndarray,
    *,
    n_components: int = 10,
    gamma: float = 0.1,
) -> dict:
    """Fit shrinkage CCA on train; report train and held-out correlations.

    Returns ``{"train": corr_tr (k,), "test": corr_te (k,), "gamma": gamma}``
    with ``k = min(n_components, p, q)``. ``corr_tr`` is sorted descending (the
    SVD order); ``corr_te`` keeps that same component order so the two align
    component-for-component.
    """
    a, b, corr_tr = fit_cca(x_tr, y_tr, n_components=n_components, gamma=gamma)
    corr_te = heldout_correlations(
        x_te, y_te, a, b, means=(x_tr.mean(axis=0), y_tr.mean(axis=0))
    )
    return {"train": corr_tr, "test": corr_te, "gamma": gamma}
