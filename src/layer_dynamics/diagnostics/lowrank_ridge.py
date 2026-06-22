"""Low-rank ridge regression for depth-local predictability (Phase 1 Step 3).

For a layer pair, fit ``Δ_ℓ ≈ v_ℓ Wᵀ`` with a rank-constrained ``W``. Following
the plan, the easy path is taken first: fit a full-rank ridge (alpha chosen by
efficient leave-one-out CV across a grid), then truncate the coefficient matrix
by SVD to each requested rank. An intercept absorbs the per-layer mean, so a
positive R² means structure beyond the layer-wide bias.

R² is pooled (Frobenius) — "fraction of variance in Δ_ℓ" across all D dims —
and measured against the **train mean** of Δ_ℓ, which makes the mean-predictor
score exactly 0 and the zero-predictor score generally negative, as the plan's
baseline table specifies.
"""

from __future__ import annotations

import numpy as np

DEFAULT_ALPHAS = np.logspace(1.0, 6.0, 11)


def pooled_r2(y_true: np.ndarray, y_pred: np.ndarray, y_ref: np.ndarray) -> float:
    """1 − ‖y_true − y_pred‖²_F / ‖y_true − y_ref‖²_F (pooled over all dims)."""
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_ref) ** 2))
    return 1.0 - ss_res / ss_tot


def fit_lowrank_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    ranks: tuple[int, ...] = (16, 32, 64, 128),
    alphas: np.ndarray = DEFAULT_ALPHAS,
) -> dict:
    """Fit ridge and report held-out pooled R² at each rank plus full rank.

    Returns ``{rank: r2, "full": r2, "alpha": float}`` where ``"full"`` is the
    untruncated ridge (the saturation reference for the rank sweep).
    """
    from sklearn.linear_model import RidgeCV

    ridge = RidgeCV(alphas=alphas, fit_intercept=True).fit(x_train, y_train)
    coef = ridge.coef_  # [D_out, D_in]
    intercept = ridge.intercept_

    ref = np.broadcast_to(y_train.mean(axis=0), y_test.shape)
    out: dict = {"alpha": float(np.atleast_1d(ridge.alpha_)[0])}

    pred_full = x_test @ coef.T + intercept
    out["full"] = pooled_r2(y_test, pred_full, ref)

    u, s, vt = np.linalg.svd(coef, full_matrices=False)
    max_rank = s.shape[0]
    for r in ranks:
        rr = min(r, max_rank)
        coef_r = (u[:, :rr] * s[:rr]) @ vt[:rr]
        pred = x_test @ coef_r.T + intercept
        out[r] = pooled_r2(y_test, pred, ref)
    return out


def identity_r2(
    v_ell_train: np.ndarray,
    v_next_train: np.ndarray,
    v_ell_test: np.ndarray,
    v_next_test: np.ndarray,
    *,
    alphas: np.ndarray = DEFAULT_ALPHAS,
) -> float:
    """Transparency baseline: R² of regressing ``v_{ℓ+1}`` on ``v_ℓ`` (full rank).

    Sits near 1 because the residual stream is additive (v_{ℓ+1} ≈ v_ℓ + a small
    update), so the regression is dominated by the identity component. Reported
    so a reader sees the bulk of v_{ℓ+1}'s variance is the residual connection,
    NOT as a comparator for the Δ_ℓ fit (different target) — see plan Step 3.
    """
    from sklearn.linear_model import RidgeCV

    reg = RidgeCV(alphas=alphas, fit_intercept=True).fit(v_ell_train, v_next_train)
    ref = np.broadcast_to(v_next_train.mean(axis=0), v_next_test.shape)
    return pooled_r2(v_next_test, reg.predict(v_ell_test), ref)
