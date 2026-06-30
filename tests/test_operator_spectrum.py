"""Tests for the A_ℓ operator-spectrum diagnostics (Phase 4 positive result)."""

from __future__ import annotations

import numpy as np
import torch

from layer_dynamics.diagnostics.operator_spectrum import (
    a_operators,
    spectrum_by_depth,
    spectrum_cosine,
    state_path_decay,
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
