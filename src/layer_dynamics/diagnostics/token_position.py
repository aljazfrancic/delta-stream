"""Token-position invariance of the AR(k*) kernel (Phase 2 Step 3).

Step 1/2 characterised the depth dynamics at the final prompt token. This module
asks whether that kernel is the *same* at other token positions, or specific to
the query position where context aggregation concentrates. It mirrors the Step 1
across-band stationarity machinery, but the comparison axis is **token position**
rather than depth band: one AR(k*) kernel is fit per position (pooled across all
feasible depths, with a per-depth intercept μ_ℓ), and the four positions are
compared with the same diagnostics —

1. **Cross-position prediction** — kernel fit at position *p*, scored on position
   *q*'s updates (per-depth means refit at *q*, so only the kernel mismatch shows).
   The confusion's diagonal is the within-position fit; the off-diagonal drop is
   the position dependence.
2. **Frobenius distance** between kernels (shared input PCA basis → one coordinate
   system), per-lag and on the stack.
3. **Subspace overlap** — principal angles between the kernels' top-r output
   subspaces.

Because the bands→positions substitution is the only change, the low-level fit
(:func:`~layer_dynamics.diagnostics.autoregress.fit_ar_pooled`), cross-eval
(:func:`~layer_dynamics.diagnostics.autoregress.eval_kernel_on_band`), and kernel
comparators (:mod:`~layer_dynamics.diagnostics.stationarity`) are reused verbatim;
the difference is that each position is a *separate* Δ array, not a depth slice of
one array.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from layer_dynamics.diagnostics.autoregress import (
    eval_kernel_on_band,
    feasible_depths,
    fit_ar_pooled,
    fit_delta_pca,
)
from layer_dynamics.diagnostics.stationarity import (
    per_lag_frobenius,
    principal_angles_deg,
    relative_frobenius,
    top_subspaces,
)


@dataclass
class PositionKernel:
    """A position's fitted kernel: name, rank-r coef, within-position held-out R²."""

    name: str
    coef_r: np.ndarray
    within_r2: float
    alpha: float


def shared_position_basis(
    deltas_by_pos: dict[str, np.ndarray], tr: np.ndarray, input_pcs: int
) -> tuple[np.ndarray, np.ndarray]:
    """Top-``input_pcs`` PCA basis fit on training Δ pooled over **all** positions.

    A single shared predictor basis puts every position's kernel in one coordinate
    system so Frobenius distances and principal angles are meaningful across
    positions (the Step 1 across-band rationale, applied to the position axis).
    """
    pooled = np.concatenate([dlt[tr] for dlt in deltas_by_pos.values()], axis=0)
    return fit_delta_pca(pooled, input_pcs)


def fit_position_kernels(
    deltas_by_pos: dict[str, np.ndarray],
    tr: np.ndarray,
    te: np.ndarray,
    k: int,
    r: int,
    basis: tuple[np.ndarray, np.ndarray],
    depths: list[int],
) -> dict[str, PositionKernel]:
    """Fit one shared low-rank AR(k) kernel per position against the common ``basis``."""
    out: dict[str, PositionKernel] = {}
    for name, dlt in deltas_by_pos.items():
        fit = fit_ar_pooled(
            dlt[tr], dlt[te], k, ranks=(r,), input_basis=basis, depths=depths
        )
        out[name] = PositionKernel(
            name=name, coef_r=fit.coef_at_rank(r), within_r2=fit.r2[r], alpha=fit.alpha
        )
    return out


def cross_position_r2(
    kernels: dict[str, PositionKernel],
    deltas_by_pos: dict[str, np.ndarray],
    tr: np.ndarray,
    te: np.ndarray,
    k: int,
    basis: tuple[np.ndarray, np.ndarray],
    depths: list[int],
) -> dict[str, dict[str, float]]:
    """Held-out R² of every (fit position → eval position) pair.

    The diagonal is the within-position fit; off-diagonal entries apply position *p*'s
    kernel to position *q*'s updates with the per-depth intercepts refit on *q* (so the
    drop reflects only the kernel mismatch). Returns ``{fit_pos: {eval_pos: R²}}``.
    """
    mat: dict[str, dict[str, float]] = {}
    for fit_name, pk in kernels.items():
        mat[fit_name] = {}
        for eval_name, dlt in deltas_by_pos.items():
            mat[fit_name][eval_name] = eval_kernel_on_band(
                pk.coef_r, dlt[tr], dlt[te], depths, k, basis
            )
    return mat


def kernel_comparisons(
    kernels: dict[str, PositionKernel], k: int, r: int
) -> dict[str, dict]:
    """Pairwise Frobenius distance + output-subspace principal angles between kernels.

    Returns ``{"<a>-<b>": {frobenius, per_lag_frobenius, output_angles}}`` for every
    unordered position pair.
    """
    names = list(kernels)
    out: dict[str, dict] = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            ca, cb = kernels[a].coef_r, kernels[b].coef_r
            _, out_a = top_subspaces(ca, r)
            _, out_b = top_subspaces(cb, r)
            angles = principal_angles_deg(out_a, out_b)
            out[f"{a}-{b}"] = {
                "frobenius": relative_frobenius(ca, cb),
                "per_lag_frobenius": per_lag_frobenius(ca, cb, k),
                "output_angle_mean_deg": float(np.mean(angles)),
                "output_angle_max_deg": float(np.max(angles)),
            }
    return out


def offdiag_retention(cross_r2: dict[str, dict[str, float]]) -> dict:
    """Summarise the confusion: mean diagonal, mean off-diagonal, worst retention.

    ``retention[p→q] = R²(fit p, eval q) / R²(fit q, eval q)`` is the fraction of the
    within-position fit a foreign kernel keeps. ``worst_retention`` is the minimum over
    all ordered off-diagonal pairs — small ⇒ position-specific, ≈1 ⇒ position-invariant.
    """
    names = list(cross_r2)
    diag = [cross_r2[p][p] for p in names]
    offdiag, retentions = [], {}
    for p in names:
        for q in names:
            if p == q:
                continue
            offdiag.append(cross_r2[p][q])
            within = cross_r2[q][q]
            retentions[f"{p}->{q}"] = cross_r2[p][q] / within if within > 0 else float("nan")
    finite = [v for v in retentions.values() if np.isfinite(v)]
    return {
        "mean_diagonal_r2": float(np.mean(diag)),
        "mean_offdiagonal_r2": float(np.mean(offdiag)),
        "worst_retention": float(np.min(finite)) if finite else float("nan"),
        "mean_retention": float(np.mean(finite)) if finite else float("nan"),
        "retention": retentions,
    }


def classify_position_invariance(
    summary: dict,
    comparisons: dict[str, dict],
    *,
    retention_tol: float = 0.10,
    angle_tol_deg: float = 30.0,
) -> dict:
    """Apply the plan's Step 3 interpretation to the confusion + kernel comparisons.

    Verdict in {``position_invariant``, ``query_position_specific``, ``thin_signal``}:

    - **position_invariant** — every off-diagonal keeps ≥ (1−tol) of the within-position
      fit and all output-subspace mean angles are ≤ ``angle_tol``. The convolutive
      picture is defensible without a position qualifier.
    - **thin_signal** — the within-position fits themselves are near zero (mean diagonal
      R² ≤ 0.05): the AR signal at these positions is too weak to compare; rerun with
      more examples per position before concluding.
    - **query_position_specific** — fits are real but a foreign kernel loses a large
      fraction of the within-position R²: the dynamics narrow to context-aggregated
      recall at the query position.

    The tolerances are reporting aids for the Gate 2 memo, not pre-registered cutoffs
    (mirroring Steps 1/4); the Phase 3 architecture branch was already locked at Step 1.
    """
    max_angle = max(c["output_angle_mean_deg"] for c in comparisons.values())
    if summary["mean_diagonal_r2"] <= 0.05:
        verdict = "thin_signal"
    elif summary["worst_retention"] >= 1.0 - retention_tol and max_angle <= angle_tol_deg:
        verdict = "position_invariant"
    else:
        verdict = "query_position_specific"
    return {
        "verdict": verdict,
        "mean_diagonal_r2": summary["mean_diagonal_r2"],
        "mean_offdiagonal_r2": summary["mean_offdiagonal_r2"],
        "worst_retention": summary["worst_retention"],
        "max_output_angle_deg": float(max_angle),
        "tolerances": {"retention_tol": retention_tol, "angle_tol_deg": angle_tol_deg},
    }


def feasible_depths_for(n_delta: int, k: int) -> list[int]:
    """Target depths feasible for AR(k) — re-exported convenience for the script."""
    return feasible_depths(n_delta, k)
