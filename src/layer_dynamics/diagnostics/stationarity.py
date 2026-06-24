"""Across-depth stationarity of the AR(k) kernel (Phase 2 Step 1).

Phase 1 Step 5 found that a single low-rank AR(k) kernel — Δ_ℓ predicted from
Δ_{ℓ-1..ℓ-k} with a per-depth intercept μ_ℓ — fits the residual updates well, and
that the structure survives both nulls. This module asks the H2 question: *is it the
same kernel at every depth?* It fits one band-shared kernel on each of the early,
middle, and late depth bands (via :func:`layer_dynamics.diagnostics.autoregress.fit_ar_pooled`
restricted to the band's target depths, with a **shared input PCA basis** so the
kernels live in one coordinate system) and compares the three with the four checks
the plan fixes:

1. **Frobenius distance** between kernels, per-lag block W_j and on the concatenated
   stack, normalised so 0 = identical and ~√2 = orthogonal-of-equal-norm.
2. **Subspace overlap** — principal angles between the top-r singular subspaces, in
   both Δ-output space and lagged-predictor space.
3. **Cross-band prediction** — score each band's kernel on the *other* bands' updates
   (per-depth means refit at the target band, so only the kernel mismatch shows). A
   3×3 confusion of held-out R² whose off-diagonal drop is the non-stationarity.
4. **Bootstrap** — resample examples within each band, refit, and ask whether the
   between-band distance exceeds the within-band sampling spread.

The AR-on-Δ stationarity tested here is a strong *proxy* for H2's MA/generative
stationarity, not a logical equivalent (plan Step 1); the MA recheck is Phase 3 Step 4.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from layer_dynamics.diagnostics.autoregress import (
    LAG_SWEEP,
    eval_kernel_on_band,
    fit_ar_pooled,
)

BAND_NAMES = ("early", "middle", "late")


def band_depths(n_delta: int, k: int) -> dict[str, list[int]]:
    """Feasible AR(k) **target** depths in each early/middle/late third.

    Bands partition ℓ ∈ {0..L-1} into [0,⌊L/3⌋), [⌊L/3⌋,⌊2L/3⌋), [⌊2L/3⌋,L). Only
    depths with ℓ ≥ k are feasible (all k lags exist). On Pythia-410m (L=24) with k=5
    this is early {5,6,7}, middle {8..15}, late {16..23}.
    """
    lo1, lo2 = n_delta // 3, 2 * n_delta // 3
    spans = {"early": range(0, lo1), "middle": range(lo1, lo2), "late": range(lo2, n_delta)}
    return {name: [e for e in span if e >= k] for name, span in spans.items()}


def select_k(
    granger: object,
    n_delta: int,
    *,
    sweep: tuple[int, ...] = LAG_SWEEP,
    bonferroni_alpha: float = 0.05,
    min_early_depths: int = 3,
) -> dict:
    """Pick the AR order k for Step 1 from the Phase 1 Step 5 Granger profile.

    ``granger`` is a DataFrame with columns ``depth, lag, neg_log10_p`` (Step 5's
    ``granger_real.csv``). Following the plan's two-stage rule:

    - **Granger-selected lag** = the largest lag j in ``sweep`` at which *any* depth
      cleared the Bonferroni threshold (α / number of feasible cells, the count of
      rows — strictly the upper-triangular ℓ ≥ j region, not L·|lags|).
    - **Feasibility cap** = ⌊L/3⌋ − ``min_early_depths`` so the early band keeps at
      least ``min_early_depths`` valid target depths for the across-band comparison.
    - **k** = the Granger-selected lag if it fits under the cap; otherwise (including
      the "still significant at the largest lag tested" case) the largest ``sweep``
      value ≤ cap. The capped case means Step 1 tests only *short-horizon*
      stationarity — flag it for the Gate 2 memo; long-horizon is deferred to the
      Phase 3 Step 4 MA recheck.
    """
    n_tests = int(len(granger))
    thr = -np.log10(bonferroni_alpha / n_tests)
    sig_lags = sorted({int(j) for j in granger.loc[granger["neg_log10_p"] > thr, "lag"]})
    granger_lag = max(sig_lags) if sig_lags else 0
    cap = n_delta // 3 - min_early_depths
    in_sweep_capped = [j for j in sweep if j <= cap]
    if not in_sweep_capped:
        raise ValueError(f"no lag in sweep {sweep} satisfies the feasibility cap {cap}")
    capped = granger_lag > cap
    k = granger_lag if not capped else max(in_sweep_capped)
    return {
        "k": int(k),
        "granger_selected_lag": int(granger_lag),
        "feasibility_cap": int(cap),
        "capped": bool(capped),
        "bonferroni_threshold_nlp": float(thr),
        "n_granger_cells": n_tests,
    }


def relative_frobenius(a: np.ndarray, b: np.ndarray) -> float:
    """‖a − b‖_F / (½(‖a‖_F + ‖b‖_F)): 0 = identical, ~√2 = orthogonal & equal-norm."""
    denom = 0.5 * (np.linalg.norm(a) + np.linalg.norm(b))
    return float(np.linalg.norm(a - b) / denom) if denom > 0 else 0.0


def per_lag_frobenius(coef_a: np.ndarray, coef_b: np.ndarray, k: int) -> list[float]:
    """Relative Frobenius distance for each lag block W_j (rows split into k blocks)."""
    q = coef_a.shape[0] // k
    return [relative_frobenius(coef_a[j * q : (j + 1) * q], coef_b[j * q : (j + 1) * q])
            for j in range(k)]


def top_subspaces(coef: np.ndarray, r: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-r left (predictor-space [kq,r]) and right (Δ-space [D,r]) singular vectors."""
    u, _, vt = np.linalg.svd(coef, full_matrices=False)
    rr = min(r, u.shape[1])
    return u[:, :rr], vt[:rr].T


def principal_angles_deg(basis_a: np.ndarray, basis_b: np.ndarray) -> np.ndarray:
    """Principal angles (degrees, ascending) between two orthonormal column spaces."""
    s = np.linalg.svd(basis_a.T @ basis_b, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1.0, 1.0)))


def _angle_summary(angles: np.ndarray) -> dict:
    return {
        "mean_deg": float(np.mean(angles)),
        "median_deg": float(np.median(angles)),
        "max_deg": float(np.max(angles)),
    }


@dataclass
class BandKernel:
    """A band's fitted kernel: name, target depths, rank-r coef, within-band R²."""

    name: str
    depths: list[int]
    coef_r: np.ndarray
    within_r2: float
    alpha: float


def fit_band_kernels(
    dlt_train: np.ndarray,
    dlt_test: np.ndarray,
    k: int,
    r: int,
    basis: tuple[np.ndarray, np.ndarray],
    bands: dict[str, list[int]],
) -> dict[str, BandKernel]:
    """Fit one shared low-rank AR(k) kernel per band against the common ``basis``."""
    out: dict[str, BandKernel] = {}
    for name, depths in bands.items():
        fit = fit_ar_pooled(
            dlt_train, dlt_test, k, ranks=(r,), input_basis=basis, depths=depths
        )
        out[name] = BandKernel(
            name=name, depths=depths, coef_r=fit.coef_at_rank(r),
            within_r2=fit.r2[r], alpha=fit.alpha,
        )
    return out


def cross_band_r2(
    kernels: dict[str, BandKernel],
    dlt_train: np.ndarray,
    dlt_test: np.ndarray,
    k: int,
    basis: tuple[np.ndarray, np.ndarray],
) -> dict[str, dict[str, float]]:
    """Held-out R² of every (fit band -> eval band) pair: diagonal within, off-diag cross."""
    mat: dict[str, dict[str, float]] = {}
    for fit_name, bk in kernels.items():
        mat[fit_name] = {}
        for eval_name, target in kernels.items():
            mat[fit_name][eval_name] = eval_kernel_on_band(
                bk.coef_r, dlt_train, dlt_test, target.depths, k, basis
            )
    return mat


def cross_band_r2_by_rank(
    dlt_train: np.ndarray,
    dlt_test: np.ndarray,
    k: int,
    ranks: tuple[int, ...],
    basis: tuple[np.ndarray, np.ndarray],
    bands: dict[str, list[int]],
) -> dict[int, dict[str, dict[str, float]]]:
    """Cross-band R² confusion at several truncation ranks (rank-robustness check).

    The spot-check can promote the working rank (more capacity → better pooled fit
    but a noisier, more band-specialised tail). Recomputing the cross-band confusion
    across ranks shows whether the (non-)stationarity verdict is an artefact of the
    chosen rank: a verdict that flips with rank is rank-driven; one that holds from a
    small rank up is real. Returns ``{rank: {fit_band: {eval_band: R²}}}``.
    """
    out: dict[int, dict[str, dict[str, float]]] = {}
    for r in ranks:
        kernels = fit_band_kernels(dlt_train, dlt_test, k, r, basis, bands)
        out[r] = cross_band_r2(kernels, dlt_train, dlt_test, k, basis)
    return out


def bootstrap_within_band(
    dlt_train: np.ndarray,
    dlt_test: np.ndarray,
    depths: list[int],
    k: int,
    r: int,
    basis: tuple[np.ndarray, np.ndarray],
    ref_coef: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> np.ndarray:
    """Within-band sampling spread: ‖W^(i) − Ŵ‖ over example-resampled refits.

    Resamples the example axis with replacement, refits the band kernel against the
    fixed shared ``basis``, truncates to rank r, and returns the relative Frobenius
    distance of each replicate from the point estimate ``ref_coef``.
    """
    n = dlt_train.shape[0]
    rng = np.random.default_rng(seed)
    spreads = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        fit = fit_ar_pooled(
            dlt_train[idx], dlt_test, k, ranks=(r,), input_basis=basis, depths=depths
        )
        spreads[b] = relative_frobenius(fit.coef_at_rank(r), ref_coef)
    return spreads


def classify_stationarity(
    cross_r2: dict[str, dict[str, float]],
    between_fro: dict[str, float],
    between_angles: dict[str, dict],
    bootstrap: dict[str, dict],
    *,
    retention_tol: float = 0.10,
    angle_tol_deg: float = 30.0,
    collinearity_thresh: float = 1.5,
) -> dict:
    """Apply the plan's decision matrix to the four diagnostics.

    Returns a verdict in {``stationary``, ``smooth_drift``, ``pervasive_drift``,
    ``phase_transition``} with the supporting numbers. The tolerances are **reporting
    aids for the memo, not pre-registered cutoffs** (mirroring Step 4's regime
    thresholds): the full architecture branch is locked at Gate 2 after Steps 2–5,
    with this step the primary determinant.

    - **stationary** (→ Branch A) — every off-diagonal cross-band R² keeps ≥ (1−tol)
      of the within-band fit, the mean output-subspace principal angle is ≤
      ``angle_tol`` for all pairs, and no between-band distance is bootstrap-separated.
    - **phase_transition** (→ Branch C) — bands differ, and the difference is
      concentrated at one boundary (one adjacent gap dominates the other, ratio
      ≥ 2); the boundary is reported.
    - **smooth_drift** (→ Branch B, depth-smoothed) — bands differ but drift
      *collinearly* with depth: the two adjacent gaps are comparable **and** the
      early↔late gap is roughly their sum (``triangle_ratio`` ≥ ``collinearity_thresh``),
      i.e. early→middle→late lie along a path. Slowly-varying / interpolated control
      depths are defensible here.
    - **pervasive_drift** (→ Branch B, fully depth-indexed) — bands differ
      *equidistantly*: all three pairwise distances are comparable and the early↔late
      gap is **not** larger than the adjacent gaps (``triangle_ratio`` <
      ``collinearity_thresh``), i.e. the band kernels form a near-equilateral triangle.
      Adjacent bands are about as different as distant ones, so there is no smoothness
      for interpolation to exploit — free per-layer operators are required. This is the
      410m Step-1 outcome ("pervasive, not gradual"); it is a *sub-case* of the plan's
      smoothly-varying Branch B, distinguished here so the verdict matches the
      architecture lock instead of mislabelling the geometry as smooth.
    """
    pairs = [("early", "middle"), ("middle", "late"), ("early", "late")]

    # Cross-band retention: worst off-diagonal fraction of the within-band R² kept.
    retentions = {}
    for a, b in pairs:
        for src, dst in ((a, b), (b, a)):
            within = cross_r2[dst][dst]
            cross = cross_r2[src][dst]
            retentions[f"{src}->{dst}"] = cross / within if within > 0 else np.nan
    worst_retention = float(np.nanmin(list(retentions.values())))

    max_angle = max(between_angles[f"{a}-{b}"]["output"]["mean_deg"] for a, b in pairs)
    any_separated = any(bootstrap[f"{a}-{b}"]["separated"] for a, b in pairs)

    stationary = (
        worst_retention >= 1.0 - retention_tol
        and max_angle <= angle_tol_deg
        and not any_separated
    )

    # The two adjacent gaps decide drift-vs-transition (one dominating → a boundary).
    d_em, d_ml = between_fro["early-middle"], between_fro["middle-late"]
    adj_lo, adj_hi = sorted((d_em, d_ml))
    boundary_ratio = adj_hi / adj_lo if adj_lo > 0 else np.inf

    # Within "drift" (no dominant boundary), the early↔late distance separates
    # *collinear* drift (early→mid→late along a path → early↔late ≈ sum of adjacent
    # gaps → ratio ≈ 2) from *equidistant* / pervasive non-stationarity (near-
    # equilateral band triangle → early↔late ≈ the adjacent gaps → ratio ≈ 1).
    d_el = between_fro["early-late"]
    triangle_ratio = d_el / adj_hi if adj_hi > 0 else np.inf

    if stationary:
        verdict, branch = "stationary", "A (block-Toeplitz convolutive kernel)"
        boundary = None
    elif boundary_ratio >= 2.0:
        verdict, branch = "phase_transition", "C (one kernel per segment)"
        boundary = "early|middle" if d_em > d_ml else "middle|late"
    elif triangle_ratio >= collinearity_thresh:
        verdict, branch = "smooth_drift", "B (state-space / depth-smoothed kernel)"
        boundary = None
    else:
        verdict, branch = "pervasive_drift", "B (state-space / fully depth-indexed kernel)"
        boundary = None

    return {
        "verdict": verdict,
        "phase3_branch": branch,
        "segment_boundary": boundary,
        "worst_cross_band_retention": worst_retention,
        "cross_band_retention": retentions,
        "max_output_angle_deg": float(max_angle),
        "any_bootstrap_separated": bool(any_separated),
        "boundary_ratio": float(boundary_ratio),
        "triangle_ratio": float(triangle_ratio),
        "between_frobenius": between_fro,
        "tolerances": {
            "retention_tol": retention_tol,
            "angle_tol_deg": angle_tol_deg,
            "collinearity_thresh": collinearity_thresh,
        },
    }
