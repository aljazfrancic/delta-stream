"""Tests for the residual-write CLT baseline (Phase 4 Step-1 Baseline D(b))."""

from __future__ import annotations

import torch

from layer_dynamics.clt import ResidualWriteCLT


def test_forward_shape_and_causal_pairs():
    L, D, M, r = 5, 8, 7, 2
    clt = ResidualWriteCLT(L, D, M, r, seed=0)
    # full lower-triangular reach => L(L+1)/2 pairs, all with src <= tgt.
    assert len(clt.pairs) == L * (L + 1) // 2
    assert all(lp <= ell for lp, ell in clt.pairs)
    out = clt(torch.randn(3, L, M))
    assert out.shape == (3, L, D)


def test_target0_reads_only_source0():
    # The first target layer is causal: it can only read source 0.
    clt = ResidualWriteCLT(4, 6, 5, 2, seed=1)
    tgt0_pairs = [(lp, t) for lp, t in clt.pairs if t == 0]
    assert tgt0_pairs == [(0, 0)]


def test_diagonal_reach_matches_manual_per_layer():
    # reach=0 => per-layer (diagonal-only) baseline: Δ̂_ℓ = b_ℓ + U_ℓ (V_ℓᵀ u_ℓ).
    L, D, M, r = 3, 4, 5, 2
    clt = ResidualWriteCLT(L, D, M, r, reach=0, seed=2)
    assert len(clt.pairs) == L  # one (ℓ,ℓ) pair per layer
    u = torch.randn(2, L, M)
    out = clt(u)
    manual = clt.b[None].expand(2, L, D).clone()
    for ell in range(L):
        w = clt.U[ell] @ clt.V[ell].T  # [D, M]
        manual[:, ell] = manual[:, ell] + u[:, ell] @ w.T
    assert torch.allclose(out, manual, atol=1e-5)


def test_operator_fro2_matches_materialised():
    clt = ResidualWriteCLT(4, 6, 5, 3, seed=3)
    exact = clt.operator_fro2().item()
    materialised = sum(
        ((clt.U[p] @ clt.V[p].T) ** 2).sum().item() for p in range(len(clt.pairs))
    )
    assert abs(exact - materialised) / materialised < 1e-5


def test_n_params_counts_all_factors_and_bias():
    L, D, M, r = 4, 6, 5, 3
    clt = ResidualWriteCLT(L, D, M, r, seed=0)
    p = len(clt.pairs)
    assert clt.n_params() == p * D * r + p * M * r + L * D


def test_deterministic_under_seed():
    a = ResidualWriteCLT(4, 6, 5, 2, seed=7)
    b = ResidualWriteCLT(4, 6, 5, 2, seed=7)
    u = torch.randn(2, 4, 5)
    assert torch.allclose(a(u), b(u))


def test_reach_limits_backward_span():
    # reach=1 => each target reads at most itself and one layer back.
    clt = ResidualWriteCLT(5, 4, 3, 2, reach=1, seed=0)
    for lp, ell in clt.pairs:
        assert 0 <= ell - lp <= 1
