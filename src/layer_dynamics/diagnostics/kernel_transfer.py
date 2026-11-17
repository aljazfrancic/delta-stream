"""AR-kernel transfer between two networks on the same prompts (v1.1 post-hoc check D).

T4 compares *levels*: the untrained network's persistence-stripped AR(1) ``R²`` exceeds
the trained network's on the primary model. That shows predictability of this size
needs no training. It does not show that the trained network's kernel *is* the untrained
network's. This module asks that directly, the way Phase 2 asked it of depth bands and
token positions: fit the pooled AR(k) kernel on one network and score it on the other,
with the per-depth intercepts refit on the target so that only the transition map is
compared.

Each kernel keeps its own input PCA basis (it is the recorded
:func:`~layer_dynamics.diagnostics.autoregress.fit_ar_pooled` fit, so the within-network
numbers reproduce the tracked T1 table), and the other network's updates are projected
into that basis before the kernel is applied.

A second score refits one scalar gain per target depth, which forgives a kernel that is
right up to a depth-dependent scale.
"""

from __future__ import annotations

import numpy as np

from layer_dynamics.diagnostics.autoregress import (
    ARFit,
    _depth_means,
    _lagged_design,
    _project,
    eval_kernel_on_band,
    feasible_depths,
    fit_ar_pooled,
)

TRANSFERS = 0.5      # retention at or above this: the kernel carries over
NO_TRANSFER = 0.1    # retention at or below this: it does not (the Phase-2 tolerance)


def gain_refit_r2(
    coef: np.ndarray,
    dlt_train: np.ndarray,
    dlt_test: np.ndarray,
    depths: list[int],
    k: int,
    basis: tuple[np.ndarray, np.ndarray] | None,
) -> float:
    """Held-out pooled ``R²`` of ``coef`` with a per-depth scalar gain refit on train.

    Prediction at depth ℓ is ``ȳ_ℓ + g_ℓ (x_ℓ − x̄_ℓ) coef`` with ``g_ℓ`` the
    least-squares scalar on the target network's train split. With ``g_ℓ ≡ 1`` this is
    :func:`~layer_dynamics.diagnostics.autoregress.eval_kernel_on_band`.
    """
    pred_tr = _project(dlt_train, basis)
    pred_te = _project(dlt_test, basis)
    means = _depth_means(pred_tr, dlt_train, depths, k)
    ss_res = ss_tot = 0.0
    for ell in depths:
        xbar, ybar = means[ell]
        fit_tr = (_lagged_design(pred_tr, ell, k) - xbar) @ coef
        yc_tr = dlt_train[:, ell, :] - ybar
        denom = float(np.sum(fit_tr * fit_tr))
        gain = float(np.sum(fit_tr * yc_tr)) / denom if denom > 0 else 0.0
        yte = dlt_test[:, ell, :]
        pred = ybar + gain * ((_lagged_design(pred_te, ell, k) - xbar) @ coef)
        ss_res += float(np.sum((yte - pred) ** 2))
        ss_tot += float(np.sum((yte - ybar) ** 2))
    return 1.0 - ss_res / ss_tot


def effective_map(coef: np.ndarray, basis: tuple[np.ndarray, np.ndarray] | None) -> np.ndarray:
    """The kernel as a map on raw lagged updates: ``[k·D, D]`` (input projection folded in)."""
    if basis is None:
        return coef
    comps = basis[1]                       # [q, D]
    q = comps.shape[0]
    k = coef.shape[0] // q
    return np.concatenate([comps.T @ coef[j * q:(j + 1) * q] for j in range(k)], axis=0)


def relative_frobenius(a: np.ndarray, b: np.ndarray) -> float:
    """``‖a − b‖_F / ½(‖a‖_F + ‖b‖_F)``: 0 identical, √2 orthogonal at equal norm, 2 opposite."""
    denom = 0.5 * (np.linalg.norm(a) + np.linalg.norm(b))
    return float(np.linalg.norm(a - b) / denom) if denom > 0 else float("nan")


def transfer_table(
    stores: dict[str, np.ndarray],
    tr: np.ndarray,
    te: np.ndarray,
    *,
    k: int = 1,
    ranks: tuple[int, ...] = (16, 32, 64, 128),
    input_pcs: int | None = 256,
) -> dict:
    """Within- and cross-network held-out ``R²`` for every rank, with retentions.

    ``stores`` maps a network name to its updates ``[N, n_delta, D]`` on the *same*
    examples (same ``tr`` / ``te`` rows). Returns, per rank ``r``::

        {"r2": {fit: {eval: R²}}, "r2_gain_refit": {fit: {eval: R²}},
         "retention": {"a->b": R²(a→b) / R²(b→b)}, "retention_gain_refit": {...},
         "frobenius": {"a-b": relative Frobenius of the effective maps}}

    plus ``"fits"`` with each network's own-fit provenance (ridge α, depths, the full
    rank sweep the recorded tables carry).
    """
    names = list(stores)
    n_delta = next(iter(stores.values())).shape[1]
    depths = feasible_depths(n_delta, k)
    fits: dict[str, ARFit] = {
        name: fit_ar_pooled(dlt[tr], dlt[te], k, ranks=ranks, input_pcs=input_pcs)
        for name, dlt in stores.items()
    }
    out: dict = {"k": k, "depths": depths, "ranks": list(ranks), "by_rank": {}, "fits": {
        name: {"alpha": f.alpha, "best_rank": f.best_rank, "best_r2": f.best_r2,
               "r2": {str(r): f.r2[r] for r in ranks}, "r2_full": f.r2["full"]}
        for name, f in fits.items()}}
    for r in ranks:
        coefs = {name: fits[name].coef_at_rank(r) for name in names}
        r2: dict[str, dict[str, float]] = {}
        r2_gain: dict[str, dict[str, float]] = {}
        for fit_name in names:
            basis = fits[fit_name].basis
            r2[fit_name], r2_gain[fit_name] = {}, {}
            for eval_name in names:
                dlt = stores[eval_name]
                r2[fit_name][eval_name] = eval_kernel_on_band(
                    coefs[fit_name], dlt[tr], dlt[te], depths, k, basis)
                r2_gain[fit_name][eval_name] = gain_refit_r2(
                    coefs[fit_name], dlt[tr], dlt[te], depths, k, basis)
        frob = {}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                frob[f"{a}-{b}"] = relative_frobenius(
                    effective_map(coefs[a], fits[a].basis),
                    effective_map(coefs[b], fits[b].basis))
        out["by_rank"][str(r)] = {
            "r2": r2, "r2_gain_refit": r2_gain,
            "retention": retention(r2), "retention_gain_refit": retention(r2_gain, within=r2),
            "frobenius": frob,
        }
    return out


def retention(
    cross: dict[str, dict[str, float]], within: dict[str, dict[str, float]] | None = None
) -> dict[str, float]:
    """``{"a->b": R²(fit a, eval b) / R²(fit b, eval b)}`` for every ordered pair.

    ``within`` supplies the denominators when ``cross`` is a variant score (the
    gain-refit one) that should still be read against the plain within-network fit.
    """
    ref = within if within is not None else cross
    out: dict[str, float] = {}
    for a in cross:
        for b in cross[a]:
            if a == b:
                continue
            denom = ref[b][b]
            out[f"{a}->{b}"] = cross[a][b] / denom if denom > 0 else float("nan")
    return out


def classify_transfer(ret_untrained_to_trained: float, ret_trained_to_untrained: float) -> str:
    """The pre-committed verdict row for one primary retention pair (memo, check D)."""
    a, b = ret_untrained_to_trained, ret_trained_to_untrained
    if not (np.isfinite(a) and np.isfinite(b)):
        return "undetermined"
    if a >= TRANSFERS and b >= TRANSFERS:
        return "transfers"
    if a <= NO_TRANSFER:
        return "does not transfer"
    return "partial"
