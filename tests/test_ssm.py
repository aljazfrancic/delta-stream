"""Tests for the Branch B low-rank SSM model and loss (Phase 3 milestone 2)."""

from __future__ import annotations

import torch

from layer_dynamics.ssm import LowRankSSM, SSMDims, reconstruction_r2, ssm_loss


def _dims(L=4, D=5, M=7, dz=3, r=2):
    return SSMDims(n_layers=L, d_model=D, d_input=M, d_state=dz, rank=r)


def _ref_forward(model: LowRankSSM, u: torch.Tensor) -> torch.Tensor:
    """Reference forward built from materialised full operators (validates the factored path)."""
    L = model.dims.n_layers
    z = model._z0(u.shape[0])
    out = []
    for ell in range(L):
        C = model.C1[ell] @ model.C2[ell].T
        Dm = model.D1[ell] @ model.D2[ell].T
        out.append(model.b[ell] + z @ C.T + u[:, ell] @ Dm.T)
        if ell < L - 1:
            A = model.A1[ell] @ model.A2[ell].T
            B = model.B1[ell] @ model.B2[ell].T
            z = z @ A.T + u[:, ell] @ B.T
    return torch.stack(out, dim=1)


def test_forward_shape():
    d = _dims()
    m = LowRankSSM(d, seed=0)
    u = torch.randn(11, d.n_layers, d.d_input)
    assert m(u).shape == (11, d.n_layers, d.d_model)


def test_factored_forward_matches_full_matrix_reference():
    d = _dims()
    m = LowRankSSM(d, seed=1, init_scale=0.5)  # larger scale -> non-trivial values
    u = torch.randn(8, d.n_layers, d.d_input)
    assert torch.allclose(m(u), _ref_forward(m, u), atol=1e-5)


def test_zero_operators_predict_bias_only():
    d = _dims()
    m = LowRankSSM(d, seed=2)
    with torch.no_grad():
        for p in m.parameters():
            p.zero_()
        m.b.copy_(torch.arange(d.n_layers * d.d_model, dtype=torch.float32).reshape(
            d.n_layers, d.d_model))
    u = torch.randn(6, d.n_layers, d.d_input)
    pred = m(u)
    expected = m.b.unsqueeze(0).expand(6, -1, -1)
    assert torch.allclose(pred, expected)


def test_no_state_forward_drops_cross_layer_term():
    """use_state=False → Δ_ℓ = b_ℓ + D_ℓ u_ℓ (the Gate-3 per-layer baseline), independent
    of z₀ and of A,B,C; it must equal the full forward with C zeroed out."""
    d = _dims()
    m = LowRankSSM(d, seed=4, init_scale=0.4, use_state=False)
    u = torch.randn(7, d.n_layers, d.d_input)
    expected = torch.stack(
        [m.b[ell] + u[:, ell] @ (m.D1[ell] @ m.D2[ell].T).T for ell in range(d.n_layers)],
        dim=1,
    )
    assert torch.allclose(m(u), expected, atol=1e-5)
    # each layer's prediction depends only on that layer's input (no depth coupling)
    u2 = u.clone()
    u2[:, 0] += 5.0
    out, out2 = m(u), m(u2)
    assert torch.allclose(out[:, 1:], out2[:, 1:], atol=1e-5)
    assert not torch.allclose(out[:, 0], out2[:, 0])


def test_operator_fro2_matches_materialised():
    d = _dims()
    m = LowRankSSM(d, seed=3, init_scale=0.3)
    manual = 0.0
    for f1, f2 in ((m.A1, m.A2), (m.B1, m.B2), (m.C1, m.C2), (m.D1, m.D2)):
        for ell in range(f1.shape[0]):
            manual += (f1[ell] @ f2[ell].T).pow(2).sum().item()
    assert abs(m.operator_fro2().item() - manual) < 1e-3


def test_loss_components_and_backward():
    d = _dims()
    m = LowRankSSM(d, seed=4)
    u = torch.randn(9, d.n_layers, d.d_input, requires_grad=True)
    delta = torch.randn(9, d.n_layers, d.d_model)
    parts = ssm_loss(m, u, delta, lambda_l1=0.1, lambda_reg=1e-3)
    # recon == summed squared error; l1 == summed |u|.
    assert torch.allclose(parts["recon"], ((delta - m(u)) ** 2).sum())
    assert torch.allclose(parts["l1"], u.abs().sum())
    assert torch.allclose(parts["loss"],
                          parts["recon"] + 0.1 * parts["l1"] + 1e-3 * parts["reg"])
    parts["loss"].backward()
    assert m.A1.grad is not None and u.grad is not None


def test_learn_z0_flag():
    d = _dims()
    free = LowRankSSM(d, learn_z0=True, seed=5)
    fixed = LowRankSSM(d, learn_z0=False, seed=5)
    assert free.z0 is not None
    assert fixed.z0 is None
    # fixed z0 is zero, so a zeroed-operator fixed model predicts exactly the bias.
    with torch.no_grad():
        for p in fixed.parameters():
            p.zero_()
    u = torch.randn(4, d.n_layers, d.d_input)
    assert torch.allclose(fixed(u), torch.zeros(4, d.n_layers, d.d_model))


def test_reconstruction_r2_perfect_when_pred_matches():
    d = _dims()
    m = LowRankSSM(d, seed=6)
    u = torch.randn(20, d.n_layers, d.d_input)
    delta = m(u).detach()  # model reproduces delta exactly -> R² == 1
    assert abs(reconstruction_r2(m, u, delta) - 1.0) < 1e-5
