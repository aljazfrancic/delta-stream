"""Tests for the A_ℓ operator-spectrum diagnostics (Phase 4 positive result)."""

from __future__ import annotations

import numpy as np
import torch

from layer_dynamics.diagnostics.operator_spectrum import (
    a_operators,
    apply_scalar_gauge,
    spectrum_by_depth,
    spectrum_cosine,
    spectrum_rel_diff,
    state_path_decay,
    state_path_tau,
)
from layer_dynamics.ssm import LowRankSSM, SSMDims


def _ssm_with_known_A(eigs):
    """A small SSM whose every A_ℓ = diag(eigs) (A2_ℓ = I so A1_ℓ A2_ℓᵀ = A1_ℓ)."""
    dz = len(eigs)
    dims = SSMDims(n_layers=4, d_model=5, d_input=6, d_state=dz, rank=dz)
    m = LowRankSSM(dims, seed=0)
    with torch.no_grad():
        for ell in range(dims.n_layers - 1):
            m.A1[ell] = torch.diag(torch.tensor(eigs, dtype=torch.float32))
            m.A2[ell] = torch.eye(dz)
    return m


def test_spectrum_by_depth_known_eigenvalues():
    m = _ssm_with_known_A([0.5, 0.3, 0.1])
    spec = spectrum_by_depth(m)
    assert spec["n_layers_trans"] == 3 and spec["d_state"] == 3
    np.testing.assert_allclose(spec["rho"], 0.5, atol=1e-5)
    np.testing.assert_allclose(spec["mean_mod"], (0.5 + 0.3 + 0.1) / 3, atol=1e-5)
    assert spec["rho_max"] <= 1.0 and spec["max_over_unit"] == 0.0  # contractive
    np.testing.assert_allclose(spec["rho_geomean"], 0.5, atol=1e-5)


def test_state_path_decay_matches_direct_frobenius():
    # The d_z-space trace identity must equal the explicit ‖C (∏A) B‖_F.
    dims = SSMDims(n_layers=4, d_model=5, d_input=6, d_state=3, rank=3)
    m = LowRankSSM(dims, seed=1)  # random factors
    decay = state_path_decay(m, max_gap=3)
    a = a_operators(m)
    c1, c2 = m.C1.detach().numpy(), m.C2.detach().numpy()
    b1, b2 = m.B1.detach().numpy(), m.B2.detach().numpy()

    def cmat(ell):
        return c1[ell] @ c2[ell].T  # [D, dz]

    def bmat(j):
        return b1[j] @ b2[j].T  # [dz, M]

    g1 = [np.linalg.norm(cmat(j + 1) @ bmat(j)) for j in range(dims.n_layers - 1)]
    np.testing.assert_allclose(decay["mean_fro"][0], np.mean(g1), rtol=1e-5)
    g2 = [np.linalg.norm(cmat(j + 2) @ a[j + 1] @ bmat(j)) for j in range(dims.n_layers - 2)]
    np.testing.assert_allclose(decay["mean_fro"][1], np.mean(g2), rtol=1e-5)


def test_state_path_decay_contractive_falls_with_gap():
    # A strongly contractive A makes the state memory kernel shrink with the gap.
    m = _ssm_with_known_A([0.2, 0.1, 0.05])
    decay = state_path_decay(m, max_gap=3)
    assert decay["ratio_to_gap1"][0] == 1.0
    assert decay["decay_gap2_over_gap1"] < 1.0  # contracts beyond one layer


def test_spectrum_cosine_bounds():
    a = np.array([0.9, 0.5, 0.3, 0.1])
    assert np.isclose(spectrum_cosine(a, a), 1.0)
    assert np.isclose(spectrum_cosine(a, np.zeros_like(a)), 0.0)
    assert 0.0 <= spectrum_cosine(a, a[::-1]) <= 1.0


# --- gauge (corrected 2026-07-09) ------------------------------------------------
# The depth-indexed SSM's symmetry is a *per-layer* T_ℓ, so ρ(A_ℓ) is not a property
# of the fit's equivalence class; the state-path kernel is. These tests pin both.

def _gauge_fixture():
    dims = SSMDims(n_layers=6, d_model=7, d_input=9, d_state=4, rank=4)
    m = LowRankSSM(dims, seed=3)
    with torch.no_grad():  # make A_ℓ contractive so the "0 non-contractive" side is real
        m.A1 *= 0.35 / max(float(np.max(np.abs(np.linalg.eigvals(a_operators(m))))), 1e-9)
    torch.manual_seed(0)
    return m, torch.randn(5, dims.n_layers, dims.d_input)


def test_scalar_gauge_preserves_the_model_output():
    m, u = _gauge_fixture()
    before = m(u).detach().clone()
    apply_scalar_gauge(m, np.exp(np.linspace(-1.5, 1.5, m.dims.n_layers)))
    torch.testing.assert_close(before, m(u).detach(), atol=1e-4, rtol=1e-4)


def test_rho_is_not_gauge_invariant_but_state_path_is():
    m, _ = _gauge_fixture()
    spec0, decay0 = spectrum_by_depth(m), state_path_decay(m, max_gap=3)
    apply_scalar_gauge(m, np.exp(np.linspace(0.0, -4.0, m.dims.n_layers)))
    spec1, decay1 = spectrum_by_depth(m), state_path_decay(m, max_gap=3)

    assert not np.allclose(spec0["rho"], spec1["rho"], atol=1e-3), (
        "ρ(A_ℓ) must move under an output-preserving per-layer gauge — if this "
        "assertion ever passes, the gauge helper is broken, not the physics"
    )
    np.testing.assert_allclose(decay0["ratio_to_gap1"], decay1["ratio_to_gap1"], atol=1e-4)


def test_spectrum_cosine_is_near_vacuous_on_random_spectra():
    """Why spectrum_rel_diff exists: sorted non-negative vectors always correlate."""
    rng = np.random.default_rng(0)
    a = np.sort(np.abs(rng.normal(size=512)))[::-1]
    b = np.sort(np.abs(rng.normal(size=512)))[::-1]
    assert spectrum_cosine(a, b) > 0.99          # two *independent* spectra
    assert spectrum_rel_diff(a, b) > 0.02        # and the honest statistic sees it


def test_state_path_tau_recovers_a_known_e_folding():
    decay = {"gaps": [1, 2, 3, 4], "ratio_to_gap1": np.exp(-np.arange(4) / 2.0)}
    np.testing.assert_allclose(state_path_tau(decay), 2.0, rtol=1e-6)
