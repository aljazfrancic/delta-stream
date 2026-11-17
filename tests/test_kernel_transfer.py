"""Tests for AR-kernel transfer between two networks (v1.1 check D; synthetic)."""

from __future__ import annotations

import numpy as np
import pytest

from layer_dynamics.diagnostics.autoregress import eval_kernel_on_band, fit_ar_pooled
from layer_dynamics.diagnostics.kernel_transfer import (
    NO_TRANSFER,
    TRANSFERS,
    classify_transfer,
    effective_map,
    gain_refit_r2,
    relative_frobenius,
    retention,
    transfer_table,
)

N, NL, D = 240, 7, 10
RANKS = (4, 10)   # 10 = full rank of the D = 10 synthetic kernels


def _ar1(kernel: np.ndarray, seed: int, noise: float = 0.05) -> np.ndarray:
    """A depth trajectory Δ_ℓ = Δ_{ℓ-1} · kernel + noise, ``[N, NL, D]``."""
    rng = np.random.default_rng(seed)
    out = np.zeros((N, NL, D))
    out[:, 0] = rng.standard_normal((N, D))
    for ell in range(1, NL):
        out[:, ell] = out[:, ell - 1] @ kernel + noise * rng.standard_normal((N, D))
    return out


def _orth(seed: int) -> np.ndarray:
    q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((D, D)))
    return q


def _split() -> tuple[np.ndarray, np.ndarray]:
    idx = np.random.default_rng(0).permutation(N)
    return idx[:180], idx[180:]


def test_same_kernel_transfers():
    tr, te = _split()
    w = _orth(1)
    stores = {"untrained": _ar1(w, 10), "trained": _ar1(w, 11)}
    t = transfer_table(stores, tr, te, k=1, ranks=RANKS, input_pcs=None)
    cell = t["by_rank"]["10"]
    assert cell["r2"]["trained"]["trained"] > 0.9
    assert cell["retention"]["untrained->trained"] > 0.95
    assert cell["retention"]["trained->untrained"] > 0.95
    assert cell["frobenius"]["untrained-trained"] < 0.2
    assert classify_transfer(cell["retention"]["untrained->trained"],
                             cell["retention"]["trained->untrained"]) == "transfers"


def test_unrelated_kernels_do_not_transfer():
    tr, te = _split()
    stores = {"untrained": _ar1(_orth(1), 10), "trained": _ar1(_orth(2), 11)}
    t = transfer_table(stores, tr, te, k=1, ranks=RANKS, input_pcs=None)
    cell = t["by_rank"]["10"]
    assert cell["r2"]["trained"]["trained"] > 0.9          # each is predictable on its own
    assert cell["r2"]["untrained"]["untrained"] > 0.9
    assert cell["retention"]["untrained->trained"] < NO_TRANSFER
    assert classify_transfer(cell["retention"]["untrained->trained"],
                             cell["retention"]["trained->untrained"]) == "does not transfer"


def test_gain_refit_forgives_a_scale_and_nothing_else():
    tr, te = _split()
    w = _orth(3)
    stores = {"untrained": _ar1(0.5 * w, 10), "trained": _ar1(w, 11)}
    t = transfer_table(stores, tr, te, k=1, ranks=RANKS, input_pcs=None)
    cell = t["by_rank"]["10"]
    plain = cell["retention"]["untrained->trained"]
    refit = cell["retention_gain_refit"]["untrained->trained"]
    assert plain < 0.9 < refit                              # right direction, wrong gain
    # a rotation is not a gain: the refit cannot rescue an unrelated kernel
    stores = {"untrained": _ar1(_orth(1), 10), "trained": _ar1(_orth(2), 11)}
    cell = transfer_table(stores, tr, te, k=1, ranks=RANKS, input_pcs=None)["by_rank"]["10"]
    assert cell["retention_gain_refit"]["untrained->trained"] < 0.3


def test_gain_refit_is_the_plain_score_on_the_kernels_own_data():
    tr, te = _split()
    dlt = _ar1(_orth(4), 12)
    fit = fit_ar_pooled(dlt[tr], dlt[te], 1, ranks=RANKS, input_pcs=None)
    coef = fit.coef_at_rank(10)
    plain = eval_kernel_on_band(coef, dlt[tr], dlt[te], list(fit.depths), 1, fit.basis)
    assert gain_refit_r2(coef, dlt[tr], dlt[te], list(fit.depths), 1, fit.basis) == \
        pytest.approx(plain, abs=5e-3)


def test_within_network_fit_is_the_recorded_fit():
    """The diagonal of the table is fit_ar_pooled's own held-out R² (the gate relies on it)."""
    tr, te = _split()
    stores = {"untrained": _ar1(_orth(1), 10), "trained": _ar1(_orth(2), 11)}
    t = transfer_table(stores, tr, te, k=1, ranks=RANKS, input_pcs=6)
    for name, dlt in stores.items():
        fit = fit_ar_pooled(dlt[tr], dlt[te], 1, ranks=RANKS, input_pcs=6)
        assert t["fits"][name]["best_r2"] == fit.best_r2
        for r in RANKS:
            assert t["by_rank"][str(r)]["r2"][name][name] == pytest.approx(fit.r2[r], abs=1e-12)


def test_effective_map_folds_the_input_projection_in():
    rng = np.random.default_rng(0)
    q, k = 4, 2
    comps = np.linalg.qr(rng.standard_normal((D, q)))[0].T        # [q, D]
    coef = rng.standard_normal((k * q, D))
    eff = effective_map(coef, (np.zeros(D), comps))
    assert eff.shape == (k * D, D)
    x = rng.standard_normal((3, k * D))
    proj = np.concatenate([x[:, j * D:(j + 1) * D] @ comps.T for j in range(k)], axis=1)
    assert np.allclose(x @ eff, proj @ coef)
    assert effective_map(coef, None) is coef


def test_relative_frobenius_reference_points():
    a = np.random.default_rng(0).standard_normal((6, 5))
    assert relative_frobenius(a, a) == 0.0
    assert relative_frobenius(a, -a) == pytest.approx(2.0)
    b = np.zeros_like(a)
    assert np.isnan(relative_frobenius(b, b))


def test_retention_reads_the_target_networks_own_fit():
    cross = {"a": {"a": 0.8, "b": 0.1}, "b": {"a": -0.4, "b": 0.5}}
    r = retention(cross)
    assert r == {"a->b": pytest.approx(0.2), "b->a": pytest.approx(-0.5)}
    variant = {"a": {"a": 0.9, "b": 0.25}, "b": {"a": 0.4, "b": 0.6}}
    assert retention(variant, within=cross)["a->b"] == pytest.approx(0.5)
    assert np.isnan(retention({"a": {"a": 0.0, "b": 0.1}, "b": {"a": 0.1, "b": 0.0}})["a->b"])


def test_classify_transfer_rows():
    assert classify_transfer(TRANSFERS, TRANSFERS) == "transfers"
    assert classify_transfer(0.9, 0.3) == "partial"          # one direction only
    assert classify_transfer(0.3, 0.9) == "partial"
    assert classify_transfer(NO_TRANSFER, 0.9) == "does not transfer"
    assert classify_transfer(-1.2, -0.4) == "does not transfer"
    assert classify_transfer(float("nan"), 0.9) == "undetermined"
