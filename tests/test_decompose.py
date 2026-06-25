"""Tests for the Branch B alternating-minimisation optimiser (Phase 3 milestone 3)."""

from __future__ import annotations

import numpy as np
import torch

from layer_dynamics.decompose import (
    AltMinConfig,
    alternating_minimize,
    evaluate,
    infer_inputs,
    soft_threshold,
)
from layer_dynamics.ssm import LowRankSSM, SSMDims, reconstruction_r2


def _dims(L=3, D=4, M=6, dz=3, r=2):
    return SSMDims(n_layers=L, d_model=D, d_input=M, d_state=dz, rank=r)


def _feedthrough_model(D: int, M: int, r: int, seed: int) -> tuple[LowRankSSM, torch.Tensor]:
    """An L=1 SSM with z0=0, b=0 → ``pred = u @ Φᵀ`` with ``Φ = D1 D2ᵀ``.

    Reduces Branch B to a plain (per-example) linear model, so the FISTA input update is a
    standard LASSO / least-squares problem with a closed-form / sklearn ground truth.
    """
    m = LowRankSSM(_dims(L=1, D=D, M=M, dz=3, r=r), seed=seed, init_scale=0.5)
    phi = (m.D1[0] @ m.D2[0].T).detach().clone()  # [D, M]
    return m, phi


# --- soft-thresholding (the L1 prox) ---


def test_lambda_reg_state_none_is_isotropic_backcompat():
    """lambda_reg_state=None must reproduce the isotropic λ_reg·operator_fro2 gradient exactly."""
    d = _dims()
    g = torch.Generator().manual_seed(7)
    u = torch.randn(12, d.n_layers, d.d_input, generator=g)
    delta = torch.randn(12, d.n_layers, d.d_model, generator=g)
    runs = []
    for lrs in (None, 1e-4):  # None falls back to lambda_reg=1e-4 → identical fit
        cfg = AltMinConfig(lambda_l1=0.05, lambda_reg=1e-4, lambda_reg_state=lrs,
                           outer_iters=2, operator_steps=5, input_steps=5)
        m = LowRankSSM(d, seed=0, init_scale=0.3)
        res = alternating_minimize(m, u, delta, cfg)
        runs.append(res.history[-1]["r2"])
    assert abs(runs[0] - runs[1]) < 1e-9


def test_lambda_reg_state_tilts_split_toward_feedthrough():
    """A large state-path penalty shrinks ‖A,B,C‖ relative to the isotropic baseline."""
    d = _dims(L=3, D=4, M=6, dz=3, r=2)
    g = torch.Generator().manual_seed(11)
    u = torch.randn(16, d.n_layers, d.d_input, generator=g)
    delta = torch.randn(16, d.n_layers, d.d_model, generator=g)

    def _state_norm(lrs):
        cfg = AltMinConfig(lambda_l1=0.01, lambda_reg=1e-3, lambda_reg_state=lrs,
                           outer_iters=4, operator_steps=15, input_steps=8)
        m = LowRankSSM(d, seed=0, init_scale=0.3)
        alternating_minimize(m, u, delta, cfg)
        n = m.operator_norms()
        return (n["A"] ** 2 + n["B"] ** 2 + n["C"] ** 2) ** 0.5

    assert _state_norm(1.0) < _state_norm(None)  # strong state reg ⇒ smaller state path


def test_soft_threshold_values():
    x = torch.tensor([3.0, -3.0, 0.5, -0.5, 0.0])
    out = soft_threshold(x, 1.0)
    assert torch.allclose(out, torch.tensor([2.0, -2.0, 0.0, 0.0, 0.0]))


def test_soft_threshold_zero_is_identity():
    x = torch.randn(20)
    assert torch.allclose(soft_threshold(x, 0.0), x)


# --- FISTA correctness against known optima ---


def test_fista_least_squares_matches_lstsq():
    # Overdetermined (D>M) feedthrough model → unique least-squares solution.
    D, M, r = 8, 4, 4
    m, phi = _feedthrough_model(D, M, r, seed=1)
    y = torch.randn(3, 1, D, generator=torch.Generator().manual_seed(3))  # 3 examples, L=1
    u = infer_inputs(m, torch.zeros(3, 1, M), y, lambda_l1=0.0, n_steps=6000)
    # closed-form least-squares target per example (λ=0): u* = Φ⁺ y (unique, D>M full rank)
    pinv = torch.linalg.pinv(phi)
    u_star = (y[:, 0] @ pinv.T).unsqueeze(1)
    assert torch.allclose(u, u_star, atol=1e-3, rtol=1e-2)


def test_fista_lasso_matches_sklearn():
    from sklearn.linear_model import Lasso

    D, M, r = 6, 10, 6
    m, phi = _feedthrough_model(D, M, r, seed=2)
    phi_np = phi.numpy()
    y = torch.randn(4, 1, D, generator=torch.Generator().manual_seed(7))
    alpha = 0.05
    lam = 2.0 * D * alpha  # our SSE+λ‖u‖₁ vs sklearn (1/2n)‖·‖²+α‖·‖₁ with n=D rows

    u = infer_inputs(m, torch.zeros(4, 1, M), y, lambda_l1=lam, n_steps=5000)

    def lasso_obj(w: np.ndarray, target: np.ndarray) -> float:
        return float(((target - phi_np @ w) ** 2).sum() + lam * np.abs(w).sum())

    our_obj = sk_obj = 0.0
    for i in range(4):
        yi = y[i, 0].numpy()
        w_sk = Lasso(alpha=alpha, fit_intercept=False, max_iter=100000, tol=1e-5).fit(
            phi_np, yi).coef_
        our_obj += lasso_obj(u[i, 0].numpy(), yi)
        sk_obj += lasso_obj(w_sk, yi)
    # FISTA reaches the same minimum as coordinate descent (objective is the robust check;
    # the minimiser itself can be marginally non-unique for an underdetermined design).
    assert our_obj <= sk_obj + 1e-3 * (1.0 + sk_obj)
    assert abs(our_obj - sk_obj) < 1e-2 * (1.0 + sk_obj)


# --- evaluate() ---


def test_evaluate_matches_manual_and_is_batch_invariant():
    d = _dims()
    m = LowRankSSM(d, seed=3)
    u = torch.randn(17, d.n_layers, d.d_input)
    delta = torch.randn(17, d.n_layers, d.d_model)
    full = evaluate(m, u, delta)
    batched = evaluate(m, u, delta, batch_size=5)
    assert abs(full["r2"] - batched["r2"]) < 1e-5
    assert abs(full["recon"] - batched["recon"]) < 1e-3
    # r2 agrees with the ssm reference, recon with the manual SSE.
    assert abs(full["r2"] - reconstruction_r2(m, u, delta)) < 1e-5
    assert abs(full["recon"] - ((delta - m(u)) ** 2).sum().item()) < 1e-3


def test_evaluate_l0_counts_exact_zeros():
    d = _dims()
    m = LowRankSSM(d, seed=4)
    u = torch.ones(8, d.n_layers, d.d_input)
    u[:, :, : d.d_input // 2] = 0.0  # exactly half zero
    delta = torch.randn(8, d.n_layers, d.d_model)
    assert abs(evaluate(m, u, delta)["l0"] - 0.5) < 1e-6


# --- alternating_minimize ---


def _teacher_delta(d: SSMDims, n: int, seed: int) -> torch.Tensor:
    teacher = LowRankSSM(d, seed=seed, init_scale=0.3)
    g = torch.Generator().manual_seed(seed + 1)
    u_true = torch.randn(n, d.n_layers, d.d_input, generator=g)
    with torch.no_grad():
        return teacher(u_true), u_true


def test_alternating_minimize_decreases_loss():
    d = _dims()
    delta, u_init = _teacher_delta(d, n=24, seed=10)
    student = LowRankSSM(d, seed=99)
    cfg = AltMinConfig(lambda_l1=1e-3, outer_iters=8, operator_steps=20, input_steps=15)
    res = alternating_minimize(student, u_init, delta, cfg)
    assert res.history[-1]["loss"] < res.history[0]["loss"]
    assert res.history[-1]["recon"] < res.history[0]["recon"]
    assert res.history[-1]["r2"] > res.history[0]["r2"]
    assert torch.isfinite(res.u).all()


def test_alternating_minimize_produces_real_sparsity():
    d = _dims()
    delta, u_init = _teacher_delta(d, n=20, seed=11)
    # start dense, then a large L1 must drive exact zeros (the prox guarantee).
    u_init = u_init.abs() + 0.5  # all strictly nonzero
    student = LowRankSSM(d, seed=42)
    cfg = AltMinConfig(lambda_l1=50.0, outer_iters=6, operator_steps=15, input_steps=20)
    res = alternating_minimize(student, u_init, delta, cfg)
    assert res.history[0]["l0"] == 1.0  # dense start
    assert res.history[-1]["l0"] < 0.1  # heavy L1 → almost everything thresholded to 0


def test_batch_size_invariant_for_operator_phase():
    # input_steps=0 isolates the operator (Adam) phase, which is full-batch grad
    # accumulation → identical regardless of minibatch size.
    d = _dims()
    delta, u_init = _teacher_delta(d, n=16, seed=12)
    cfg_kw = dict(lambda_l1=1e-2, outer_iters=3, operator_steps=10, input_steps=0)
    full = alternating_minimize(LowRankSSM(d, seed=7), u_init, delta,
                                AltMinConfig(batch_size=None, **cfg_kw))
    batched = alternating_minimize(LowRankSSM(d, seed=7), u_init, delta,
                                   AltMinConfig(batch_size=4, **cfg_kw))
    assert abs(full.history[-1]["recon"] - batched.history[-1]["recon"]) < 1e-2


# --- infer_inputs (held-out encoder) freezes operators ---


def test_infer_inputs_freezes_and_restores_operators():
    d = _dims()
    m = LowRankSSM(d, seed=5)
    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    flags_before = [p.requires_grad for p in m.parameters()]
    u0 = torch.randn(6, d.n_layers, d.d_input)
    delta = torch.randn(6, d.n_layers, d.d_model)
    _ = infer_inputs(m, u0, delta, lambda_l1=1e-2, n_steps=30)
    for n, p in m.named_parameters():
        assert torch.equal(p, before[n]), f"operator {n} changed during infer_inputs"
    assert [p.requires_grad for p in m.parameters()] == flags_before
