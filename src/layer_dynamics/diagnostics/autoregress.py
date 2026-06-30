"""AR(k) on Δ over depth, and the nested-F (Granger-style) lag test (Phase 1 Step 5).

The "time" axis here is **depth** ℓ, not token position: each example contributes
a depth-series Δ_0, …, Δ_{L-1}, and we ask whether Δ_ℓ is better predicted from the
``k`` preceding updates Δ_{ℓ-1}, …, Δ_{ℓ-k} than from the per-depth mean alone, and
how far back in depth that dependence reaches.

Two estimators, matching the plan:

1. **Pooled AR(k) fit** — a single low-rank kernel ``W = [W_1 … W_k]`` shared across
   depths, with a **per-depth intercept** μ_ℓ ∈ ℝ^D. The intercept is handled by the
   within-group ("fixed-effects") transform: center predictors and target by their
   per-depth train means, fit the shared slope on the stacked centered data, and put
   μ_ℓ back as ``ȳ_ℓ − W x̄_ℓ`` at evaluation. Held-out R² is measured against the
   per-depth mean (the intercept-only model), so R²=0 ⟺ the lags add nothing beyond
   the per-depth bias — exactly the quantity the plan calls the "pooled AR R²".

   Ridge α is chosen by **GCV** off one economy SVD of the centered design (cheap for
   every α, no extra data split). Low-rank ``W`` is obtained by SVD-truncating the
   fitted coefficient, the same easy path Step 3 takes.

2. **Nested-F lag test** — at each depth ℓ and lag j (ℓ ≥ j), compare the multivariate
   regression of Δ_ℓ on lags {1…j-1} (reduced) vs {1…j} (full), both with intercept.
   Because D=1024 ≫ N makes a full-D multivariate F degenerate (the error covariance is
   singular), the test runs on a top-``q`` PCA projection of Δ (a conservative,
   power-limited detector that keeps Wilks' Λ well-defined). The statistic is Wilks' Λ
   with **Rao's F approximation**, which yields the F value and its (df1, df2) the plan
   asks to be saved per cell. This is, strictly, AR-order selection on one series
   (reported as Granger-style per the plan's own framing).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_ALPHAS = np.logspace(1.0, 6.0, 11)
LAG_SWEEP = (1, 2, 3, 5, 10, 15, 20)


def feasible_depths(n_delta: int, k: int) -> list[int]:
    """Target depths ℓ for which all k lags Δ_{ℓ-1..ℓ-k} exist (ℓ ≥ k)."""
    return list(range(k, n_delta))


def fit_delta_pca(dlt: np.ndarray, q: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-``q`` PCA basis of Δ pooled over all depths -> ``(mean[D], comps[q, D])``.

    Used to compress the AR **predictors** (the lagged Δ's). The full ``k·D`` design
    is too large for an economy SVD at high ``k`` (the cost is hours at D=1024); a
    top-q projection of the predictors makes the fit seconds-fast while leaving the
    target full-D, so the held-out R² is still measured in the full Δ space. q is
    chosen ≥ the rank at which Step 3's ridge saturated (128), so the predictability
    regime that matters is preserved with headroom.
    """
    n, n_delta, d = dlt.shape
    flat = dlt.reshape(n * n_delta, d)
    mean = flat.mean(axis=0)
    _, _, vt = np.linalg.svd(flat - mean, full_matrices=False)
    return mean, vt[: min(q, d)]


def _project(dlt: np.ndarray, basis: tuple[np.ndarray, np.ndarray] | None) -> np.ndarray:
    """Project Δ onto a PCA basis (no-op when ``basis`` is None)."""
    if basis is None:
        return dlt
    mean, comps = basis
    n, n_delta, d = dlt.shape
    return ((dlt.reshape(n * n_delta, d) - mean) @ comps.T).reshape(n, n_delta, comps.shape[0])


def _lagged_design(dlt: np.ndarray, ell: int, k: int) -> np.ndarray:
    """Stack [Δ_{ℓ-1}, …, Δ_{ℓ-k}] for all examples -> ``[N, k·q]`` (j=1..k order)."""
    return np.concatenate([dlt[:, ell - j, :] for j in range(1, k + 1)], axis=1)


def _gcv_alpha(u: np.ndarray, s: np.ndarray, y: np.ndarray, alphas: np.ndarray) -> float:
    """Pick ridge α by generalized cross-validation using a precomputed economy SVD.

    With ``X = U diag(s) Vᵀ`` the ridge hat matrix has eigenvalues ``f = s²/(s²+α)``
    along the left singular vectors. GCV(α) = ‖(I−A)Y‖² / (M − tr A)², evaluated in
    closed form (no per-α solve), summed over output dimensions for the multi-output
    target.
    """
    m = u.shape[0]
    uty = u.T @ y  # [P', D]
    row_sq = (uty**2).sum(axis=1)  # [P'] energy of Y along each left singular vector
    total = float((y**2).sum())
    best_alpha, best_gcv = float(alphas[0]), np.inf
    for alpha in alphas:
        f = s**2 / (s**2 + alpha)
        ss_res = total - float((2 * f - f**2) @ row_sq)
        denom = (m - float(f.sum())) ** 2
        gcv = ss_res / denom if denom > 0 else np.inf
        if gcv < best_gcv:
            best_gcv, best_alpha = gcv, float(alpha)
    return best_alpha


@dataclass
class ARFit:
    """Result of one pooled AR(k) fit: held-out R² per rank, plus provenance.

    ``coef`` is the full fitted kernel ``[k·q, D]`` (predictor-PCA coordinates ->
    full Δ space); ``coef_svd`` is its economy SVD so any rank-truncated kernel is
    recoverable via :meth:`coef_at_rank` without refitting. ``depths`` records the
    target depths the kernel was fit on (the band, in Phase 2 Step 1) and ``basis``
    the shared input PCA basis — both needed to apply this kernel to *other* depths
    with :func:`eval_kernel_on_band`.
    """

    k: int
    alpha: float
    n_train_rows: int
    n_depths: int
    r2: dict  # {rank: r2, "full": r2}
    best_rank: int
    best_r2: float
    coef: np.ndarray | None = None
    coef_svd: tuple | None = None  # (U, s, Vt) of coef
    depths: tuple[int, ...] = ()
    basis: tuple | None = None  # (mean, comps) shared input PCA basis

    def coef_at_rank(self, r: int) -> np.ndarray:
        """SVD-truncate the fitted kernel to rank ``r`` (≤ rank(coef))."""
        if self.coef_svd is None:
            raise ValueError("coef_svd not stored on this ARFit")
        bu, bs, bvt = self.coef_svd
        rr = min(r, bs.shape[0])
        return (bu[:, :rr] * bs[:rr]) @ bvt[:rr]


def _depth_means(
    pred_tr: np.ndarray, dlt_train: np.ndarray, depths: list[int], k: int
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Per-depth predictor/target train means ``{ℓ: (x̄_ℓ, ȳ_ℓ)}`` (the μ_ℓ carrier)."""
    means: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for ell in depths:
        x = _lagged_design(pred_tr, ell, k)
        means[ell] = (x.mean(axis=0), dlt_train[:, ell, :].mean(axis=0))
    return means


def _pooled_r2_from_means(
    coef: np.ndarray,
    pred_te: np.ndarray,
    dlt_test: np.ndarray,
    means: dict[int, tuple[np.ndarray, np.ndarray]],
    depths: list[int],
    k: int,
) -> float:
    """Held-out pooled R² of ``coef`` over ``depths``, vs the per-depth mean ȳ_ℓ.

    The per-depth means in ``means`` are the intercept μ_ℓ: prediction at depth ℓ is
    ``ȳ_ℓ + (x_ℓ − x̄_ℓ) coef``. When ``coef`` was fit on a *different* depth band
    but ``means`` is recomputed on ``depths`` (see :func:`eval_kernel_on_band`), this
    isolates the kernel — the offsets are matched, only the transition map differs.
    """
    ss_res = ss_tot = 0.0
    for ell in depths:
        xbar, ybar = means[ell]
        yte = dlt_test[:, ell, :]
        pred = ybar + (_lagged_design(pred_te, ell, k) - xbar) @ coef
        ss_res += float(np.sum((yte - pred) ** 2))
        ss_tot += float(np.sum((yte - ybar) ** 2))
    return 1.0 - ss_res / ss_tot


def eval_kernel_on_band(
    coef: np.ndarray,
    dlt_train: np.ndarray,
    dlt_test: np.ndarray,
    depths: list[int],
    k: int,
    basis: tuple[np.ndarray, np.ndarray] | None,
) -> float:
    """Apply kernel ``coef`` to target ``depths``, with per-depth means refit there.

    This is the cross-band prediction of Phase 2 Step 1: take a kernel fit on one
    band and score it on another band's updates. The per-depth intercepts μ_ℓ are
    recomputed from the *train* split at the evaluation depths, so the reported R²
    degradation reflects the kernel mismatch alone, not a mean offset between bands.
    """
    pred_tr = _project(dlt_train, basis)
    pred_te = _project(dlt_test, basis)
    means = _depth_means(pred_tr, dlt_train, depths, k)
    return _pooled_r2_from_means(coef, pred_te, dlt_test, means, depths, k)


def fit_ar_pooled(
    dlt_train: np.ndarray,
    dlt_test: np.ndarray,
    k: int,
    *,
    ranks: tuple[int, ...] = (16, 32, 64, 128),
    alphas: np.ndarray = DEFAULT_ALPHAS,
    input_pcs: int | None = 256,
    input_basis: tuple[np.ndarray, np.ndarray] | None = None,
    depths: list[int] | None = None,
) -> ARFit:
    """Fit the shared low-rank AR(k) kernel with a per-depth intercept.

    ``dlt_*`` are ``[N, n_delta, D]`` arrays of Δ. The **predictors** (lagged Δ's)
    are compressed to ``input_pcs`` principal components (basis fit on the training
    Δ unless ``input_basis`` is supplied — pass one to share it across the k sweep,
    or across depth bands so the fitted kernels live in a common coordinate system);
    the **target** stays full-D, so R² is still the held-out fraction of full Δ_ℓ
    variance (against the per-depth mean) at each rank and at full predictor rank.
    Set ``input_pcs=None`` to fit in the raw ``k·D`` predictor space (much slower).

    ``depths`` restricts the **target** depths the shared kernel is fit over (Phase 2
    Step 1 fits one kernel per early/middle/late band); it defaults to every feasible
    depth ℓ ≥ k (the Phase 1 Step 5 pooled fit). The fitted kernel and its SVD are
    returned on the :class:`ARFit` so it can be compared and cross-applied.
    """
    n_delta = dlt_train.shape[1]
    if depths is None:
        depths = feasible_depths(n_delta, k)
    else:
        depths = list(depths)
        bad = [e for e in depths if e < k or e >= n_delta]
        if bad:
            raise ValueError(f"depths {bad} infeasible for k={k}, n_delta={n_delta}")
    if not depths:
        raise ValueError(f"k={k} leaves no feasible depths for n_delta={n_delta}")

    basis = input_basis
    if basis is None and input_pcs is not None:
        basis = fit_delta_pca(dlt_train, input_pcs)
    pred_tr = _project(dlt_train, basis)  # [N, n_delta, q] predictor coordinates
    pred_te = _project(dlt_test, basis)

    # Per-depth centering (within transform) absorbs μ_ℓ before the shared fit. The
    # predictor design is in PCA space; the target Δ_ℓ stays full-D.
    means = _depth_means(pred_tr, dlt_train, depths, k)
    xc_blocks, yc_blocks = [], []
    for ell in depths:
        xbar, ybar = means[ell]
        xc_blocks.append(_lagged_design(pred_tr, ell, k) - xbar)
        yc_blocks.append(dlt_train[:, ell, :] - ybar)
    xc = np.concatenate(xc_blocks, axis=0)  # [M, kq]
    yc = np.concatenate(yc_blocks, axis=0)  # [M, D]

    u, s, vt = np.linalg.svd(xc, full_matrices=False)
    alpha = _gcv_alpha(u, s, yc, alphas)
    d_inv = s / (s**2 + alpha)
    coef = vt.T @ (d_inv[:, None] * (u.T @ yc))  # [kq, D], maps predictors -> Δ

    # Rank sweep by SVD-truncating the fitted coefficient (Step 3's easy path).
    bu, bs, bvt = np.linalg.svd(coef, full_matrices=False)
    max_rank = bs.shape[0]

    r2: dict = {"full": _pooled_r2_from_means(coef, pred_te, dlt_test, means, depths, k)}
    for r in ranks:
        rr = min(r, max_rank)
        coef_r = (bu[:, :rr] * bs[:rr]) @ bvt[:rr]
        r2[r] = _pooled_r2_from_means(coef_r, pred_te, dlt_test, means, depths, k)

    best_rank = max(ranks, key=lambda r: r2[r])
    return ARFit(
        k=k,
        alpha=alpha,
        n_train_rows=xc.shape[0],
        n_depths=len(depths),
        r2=r2,
        best_rank=best_rank,
        best_r2=r2[best_rank],
        coef=coef,
        coef_svd=(bu, bs, bvt),
        depths=tuple(depths),
        basis=basis,
    )


def wilks_rao_f(
    e_full: np.ndarray, e_reduced: np.ndarray, *, p: int, vh: int, ve: int
) -> dict:
    """Wilks' Λ nested-model test with Rao's F approximation.

    ``e_full`` / ``e_reduced`` are the residual sum-of-squares-and-cross-products
    (SSCP) matrices (q×q) of the full and reduced multivariate regressions. ``p`` is
    the response dimension, ``vh`` the hypothesis df (number of added predictor
    columns), ``ve`` the error df of the full model. Returns ``{lambda, F, df1, df2,
    p_value}``. Rao's F is exact when ``p`` or ``vh`` equals 1 and a close
    approximation otherwise (Rao 1973; Rencher 2002).
    """
    from scipy import stats

    sign_f, logdet_f = np.linalg.slogdet(e_full)
    sign_r, logdet_r = np.linalg.slogdet(e_reduced)
    if sign_f <= 0 or sign_r <= 0:
        # Singular error SSCP — test is undefined (too few error df for q).
        return {"lambda": np.nan, "F": np.nan, "df1": p * vh, "df2": np.nan, "p_value": np.nan}
    lam = float(np.exp(logdet_f - logdet_r))  # det(E_full)/det(E_reduced) ∈ (0,1]

    denom = p**2 + vh**2 - 5
    t = np.sqrt((p**2 * vh**2 - 4) / denom) if denom > 0 else 1.0
    w = ve - (p - vh + 1) / 2.0
    df1 = p * vh
    df2 = w * t - (p * vh - 2) / 2.0
    lam_t = lam ** (1.0 / t)
    f_stat = (1.0 - lam_t) / lam_t * df2 / df1
    p_value = float(stats.f.sf(f_stat, df1, df2)) if df2 > 0 else np.nan
    return {
        "lambda": lam, "F": float(f_stat), "df1": int(df1), "df2": float(df2),
        "p_value": p_value,
    }


def _resid_sscp(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Residual SSCP ``RᵀR`` (q×q) of the OLS fit of ``y`` (n×q) on ``z`` (n×p)."""
    beta, *_ = np.linalg.lstsq(z, y, rcond=None)
    r = y - z @ beta
    return r.T @ r


def pca_project(dlt: np.ndarray, q: int) -> np.ndarray:
    """Project Δ onto its top-``q`` principal components (fit on all depths pooled).

    Centering uses the global Δ mean; the per-depth intercept in the Granger test is
    carried by each regression's own intercept column, so this only needs to supply a
    compact, full-rank-friendly coordinate system for the nested F.
    """
    n, n_delta, d = dlt.shape
    flat = dlt.reshape(n * n_delta, d)
    mean = flat.mean(axis=0)
    _, _, vt = np.linalg.svd(flat - mean, full_matrices=False)
    basis = vt[:q]  # [q, D]
    proj = (dlt.reshape(n * n_delta, d) - mean) @ basis.T
    return proj.reshape(n, n_delta, q)


def granger_lag_grid(
    dlt: np.ndarray, *, lags: tuple[int, ...] = LAG_SWEEP, q_pca: int = 32
) -> dict:
    """Per-(depth ℓ, lag j) nested-F lag test on the top-``q_pca`` PCA projection.

    Returns dict of arrays indexed ``[depth, lag_index]`` (NaN where ℓ < j): the F
    statistic, the −log10 p-value, and df1/df2. The test uses all examples (it is a
    hypothesis test on coefficients, not a generalization measure).
    """
    n, n_delta, _ = dlt.shape
    proj = pca_project(dlt, q_pca)  # [N, n_delta, q]
    ones = np.ones((n, 1))

    shape = (n_delta, len(lags))
    f_arr = np.full(shape, np.nan)
    nlp_arr = np.full(shape, np.nan)  # −log10 p
    df1_arr = np.full(shape, np.nan)
    df2_arr = np.full(shape, np.nan)
    n_tests = 0

    for ji, j in enumerate(lags):
        for ell in range(j, n_delta):
            y = proj[:, ell, :]  # [N, q]
            # reduced: intercept + lags 1..j-1 ; full: + lag j.
            red_cols = [ones] + [proj[:, ell - jj, :] for jj in range(1, j)]
            z_red = np.concatenate(red_cols, axis=1)
            z_full = np.concatenate([z_red, proj[:, ell - j, :]], axis=1)
            e_red = _resid_sscp(z_red, y)
            e_full = _resid_sscp(z_full, y)
            ve = n - z_full.shape[1]
            res = wilks_rao_f(e_full, e_red, p=q_pca, vh=q_pca, ve=ve)
            f_arr[ell, ji] = res["F"]
            df1_arr[ell, ji] = res["df1"]
            df2_arr[ell, ji] = res["df2"]
            pv = res["p_value"]
            if pv is not None and np.isfinite(pv):
                nlp_arr[ell, ji] = -np.log10(max(pv, 1e-300))
            n_tests += 1

    return {
        "F": f_arr,
        "neg_log10_p": nlp_arr,
        "df1": df1_arr,
        "df2": df2_arr,
        "lags": list(lags),
        "n_tests": n_tests,
        "q_pca": q_pca,
    }


def scramble_per_depth(dlt: np.ndarray, seed: int) -> np.ndarray:
    """Null 1: permute the example axis independently at each depth.

    Destroys within-example cross-depth structure (so any AR/Granger signal vanishes)
    while preserving each Δ_ℓ's marginal exactly. A *different* permutation per depth
    is essential — a shared permutation would leave the dynamics intact.
    """
    out = np.empty_like(dlt)
    for ell in range(dlt.shape[1]):
        rng = np.random.default_rng(seed + ell)
        out[:, ell, :] = dlt[rng.permutation(dlt.shape[0]), ell, :]
    return out


def state_pca_basis(rs: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-``m`` principal directions of the **states** v_ℓ, pooled over depth.

    Returns ``(mean[D], dirs[m, D])`` with ``dirs`` orthonormal. These are the
    persistent, high-variance ("rogue") directions the residual stream is dominated
    by (top-1 ≈ 28 %, top-20 ≈ 85 % of variance on Pythia-410m). T1 projects the
    updates onto the *complement* of this subspace to ask whether the AR-on-Δ
    predictability survives where state persistence is no longer dominant — i.e.
    whether it is genuine depth structure or the mechanical ±v_ℓ tautology
    (Δ_ℓ ⊃ −v_ℓ ⊂ Δ_{ℓ-1}).
    """
    n, n_states, d = rs.shape
    flat = rs.reshape(n * n_states, d).astype(np.float64)
    mean = flat.mean(axis=0)
    if m <= 0:
        return mean, np.zeros((0, d))
    _, _, vt = np.linalg.svd(flat - mean, full_matrices=False)
    return mean, vt[: min(m, d)]


def remove_state_dirs(dlt: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """Project the updates Δ onto the orthogonal complement of ``dirs`` ([m, D]).

    Removing the dominant **state** directions from every Δ_ℓ strips the persistent
    ±v_ℓ component that inflates the AR-on-Δ R². No mean-centering of Δ is needed —
    the per-depth intercept μ_ℓ in :func:`fit_ar_pooled` already absorbs offsets; here
    we only remove variance *directions*. (Projection then differencing equals
    differencing then projecting, since both are linear with the same projector, so
    this is exactly "re-difference the persistence-removed states".)
    """
    if dirs.shape[0] == 0:
        return dlt
    n, n_delta, d = dlt.shape
    flat = dlt.reshape(n * n_delta, d)
    comp = (flat @ dirs.T) @ dirs  # component along the removed directions
    return (flat - comp).reshape(n, n_delta, d).astype(dlt.dtype)


def random_walk_surrogate(dlt: np.ndarray, seed: int, *, rank: int = 256) -> np.ndarray:
    """Null T2: per-layer marginal-matched **independent-step** surrogate.

    For each depth ℓ independently, draw new updates from a Gaussian matched to that
    layer's empirical mean and (rank-truncated) covariance. The resulting Δ̂_ℓ are
    *independent across depth* — equivalently a random walk v̂_ℓ = v̂_{ℓ-1} + ξ_ℓ
    whose increments carry no cross-layer dependence — while every per-layer step
    marginal (1st + 2nd moment, including the persistent high-variance directions) is
    preserved.

    This is the parametric persistence-matched companion to the scramble null. For
    AR-on-Δ it is expected to collapse to ≈0: with independent steps the shared ±v_ℓ
    terms cancel in expectation (Var(v_ℓ) contributions to Cov(Δ_ℓ, Δ_{ℓ-1}) net to
    zero), so any AR-on-Δ predictability *above* this floor cannot be a mere
    state-persistence artefact — it requires genuine step-to-step structure.
    """
    rng = np.random.default_rng(seed)
    n, n_delta, d = dlt.shape
    out = np.empty_like(dlt)
    for ell in range(n_delta):
        x = dlt[:, ell, :].astype(np.float64)
        mu = x.mean(axis=0)
        u, s, vt = np.linalg.svd(x - mu, full_matrices=False)
        r = min(rank, s.shape[0])
        factor = (s[:r] / np.sqrt(max(n - 1, 1)))[:, None] * vt[:r]  # [r, D], cov^½
        z = rng.standard_normal((n, r))
        out[:, ell, :] = (mu + z @ factor).astype(dlt.dtype)
    return out


def granger_significant_by_lag(
    grid: dict, *, bonferroni: bool = True, alpha: float = 0.05
) -> dict:
    """Count Bonferroni-significant Granger cells split by lag==1 vs lag>=2.

    The ±v_ℓ tautology lives **entirely at lag 1** (Δ_ℓ shares v_ℓ only with Δ_{ℓ-1};
    Δ_{ℓ-2}=v_{ℓ-1}−v_{ℓ-2} shares no v_ℓ or v_{ℓ+1} term), so the lag≥2 significant
    count is an intrinsically tautology-free read of depth dependence (T3). Returns
    ``{n_tests, threshold, lag1_sig, lag1_total, lagge2_sig, lagge2_total}``.
    """
    nlp = grid["neg_log10_p"]
    lags = grid["lags"]
    n_tests = int(grid["n_tests"])
    thr = -np.log10(alpha / n_tests if bonferroni else alpha)
    lag1_sig = lag1_tot = ge2_sig = ge2_tot = 0
    for ji, j in enumerate(lags):
        col = nlp[:, ji]
        finite = np.isfinite(col)
        sig = int(np.nansum(col[finite] > thr))
        tot = int(finite.sum())
        if j == 1:
            lag1_sig, lag1_tot = sig, tot
        else:
            ge2_sig += sig
            ge2_tot += tot
    return {
        "n_tests": n_tests, "threshold": float(thr),
        "lag1_sig": lag1_sig, "lag1_total": lag1_tot,
        "lagge2_sig": ge2_sig, "lagge2_total": ge2_tot,
    }
