"""Effective kernel depth k* of the AR-on-Δ model (Phase 2 Step 2).

How many preceding updates does Δ_ℓ meaningfully depend on for *reconstruction*?
Step 1 found the kernel is depth-indexed (H2 fails), so k* is reported **as a
function of depth**: at each target depth ℓ we fit single-depth AR(k) for the
plan's sweep (capped at k ≤ ℓ — a depth cannot look back past the embedding) and
read off the smallest k that reaches the best held-out R².

**The lag-1 tautology (see the project's cca-vdelta memory).** Δ_ℓ = v_{ℓ+1}−v_ℓ
and Δ_{ℓ-1} = v_ℓ−v_{ℓ-1} share ∓v_ℓ, so a lag-1 fit mechanically recovers much of
Δ_ℓ. The naive reconstruction R² is therefore lag-1-dominated, and k* read off it is
small *by construction*. Two reads guard the interpretation: (1) the **scrambled
null** (permute the example axis per depth) must collapse the whole curve — it
confirms the signal is real example-specific structure, not a marginal artefact;
(2) the **incremental gain beyond lag 1** (best-k R² − lag-1 R²) measures how much
lags ≥ 2 add *on top of* the mechanical predecessor. k* on the full curve answers
"reconstruction-effective depth"; it is deliberately contrasted in the Step 2 memo
with Phase 1 Step 5's Granger reach (statistical significance to lag 20), which is a
different question (incremental significance, not reconstruction value).
"""

from __future__ import annotations

import numpy as np

from layer_dynamics.diagnostics.autoregress import LAG_SWEEP, fit_ar_pooled


def feasible_k(ell: int, sweep: tuple[int, ...] = LAG_SWEEP) -> list[int]:
    """Sweep values ≤ ell (AR(k) at depth ℓ needs Δ_{ℓ-k}, i.e. ℓ ≥ k)."""
    return [k for k in sweep if k <= ell]


def kstar_from_curve(curve: dict[int, float], *, tol: float = 0.95) -> int:
    """Smallest k whose R² reaches ``tol`` of the **best** R² over the swept k.

    The plan defines k* against the asymptote at the largest k assuming monotone
    saturation; these curves are non-monotone (they peak early then decline as extra
    lags add noise at fixed rank), so the reference is the peak R² over the sweep —
    the smallest k achieving near-best reconstruction. Equivalent to the plan's rule
    on a saturating curve, and the honest choice on a declining one.
    """
    ks = sorted(curve)
    best = max(curve[k] for k in ks)
    if best <= 0:
        return ks[0]
    for k in ks:
        if curve[k] >= tol * best:
            return k
    return ks[-1]


def depth_r2_curve(
    dlt_train: np.ndarray,
    dlt_test: np.ndarray,
    ell: int,
    *,
    sweep: tuple[int, ...] = LAG_SWEEP,
    rank: int = 256,
    basis: tuple[np.ndarray, np.ndarray] | None = None,
    input_pcs: int = 256,
) -> dict[int, float]:
    """Held-out R²(k) for a single target depth ℓ over the feasible part of the sweep."""
    out: dict[int, float] = {}
    for k in feasible_k(ell, sweep):
        fit = fit_ar_pooled(
            dlt_train, dlt_test, k, ranks=(rank,), input_basis=basis,
            input_pcs=input_pcs, depths=[ell],
        )
        out[k] = fit.r2[rank]
    return out


def band_r2_curve(
    dlt_train: np.ndarray,
    dlt_test: np.ndarray,
    depths: list[int],
    *,
    sweep: tuple[int, ...] = LAG_SWEEP,
    rank: int = 256,
    basis: tuple[np.ndarray, np.ndarray] | None = None,
    input_pcs: int = 256,
) -> dict[int, float]:
    """Held-out R²(k) for a fixed band of target depths (k ≤ min(depths) to keep all)."""
    kmax = min(depths)
    out: dict[int, float] = {}
    for k in sweep:
        if k > kmax:
            continue
        fit = fit_ar_pooled(
            dlt_train, dlt_test, k, ranks=(rank,), input_basis=basis,
            input_pcs=input_pcs, depths=depths,
        )
        out[k] = fit.r2[rank]
    return out


def incremental_over_lag1(curve: dict[int, float]) -> float:
    """Best-k R² minus the lag-1 R²: how much lags ≥ 2 add beyond the predecessor."""
    if 1 not in curve:
        return float("nan")
    return float(max(curve.values()) - curve[1])
