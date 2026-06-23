"""Alternating-minimisation optimiser for the Branch B decomposition (Phase 3 Step 3).

The Branch B loss ``L_B`` (see :mod:`layer_dynamics.ssm`) is bi-convex-ish: with the
per-layer inputs ``{u_ℓ}`` fixed it is a smooth (Adam-friendly) problem in the SSM
operators ``{A,B,C,D,b,z₀}``; with the operators fixed it is a single coupled LASSO
in the stacked inputs (one per example, coupled across depth because ``u_j`` feeds
``Δ_ℓ`` for every ``ℓ ≥ j`` through the state chain and the feedthrough). The plan
([phase-3](../../docs/implementation-plan/phase-3-decomposition.md) Step 1) is explicit
that these are **alternated**, never co-optimised in one Adam call:

  * **Operator update** — one Adam sweep over ``{A,B,C,D,b,z₀}`` with ``{u_ℓ}`` held
    fixed, minimising ``recon + λ_reg·‖operators‖_F²`` (the L1 term is constant here).
    Full-batch (gradients accumulated over example minibatches, then one ``step``).
  * **Input update** — one proximal-gradient (FISTA) pass over ``{u_ℓ}`` with the
    operators held fixed, minimising ``recon + λ_L1·‖u‖₁``. Proximal gradient keeps the
    L1's *exact zeros* (soft-thresholding), which a plain Adam-with-L1 would not — that
    sparsity guarantee is the whole reason the alternation is preserved.

The smooth gradient ``∇_u recon`` is obtained by autodiff through the SSM unroll, so the
coupling across depth is handled exactly with no hand-derived adjoint. The step size is
found by backtracking on a local Lipschitz estimate ``L`` (Beck & Teboulle 2009); because
the input subproblem's Hessian ``2·JᵀJ`` is identical across examples (same operators),
``L`` is shared across minibatches and warm-started across outer iterations.

This module is model-agnostic over :class:`~layer_dynamics.ssm.LowRankSSM`: it only uses
``model(u)``, ``model.parameters()``, ``model.operator_fro2()`` and ``operator_norms()``.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, field

import torch
from torch import Tensor

from layer_dynamics.ssm import LowRankSSM


def soft_threshold(x: Tensor, t: float | Tensor) -> Tensor:
    """Element-wise soft-thresholding ``sign(x)·max(|x|−t, 0)`` — the L1 prox operator."""
    return x.sign() * (x.abs() - t).clamp_min(0.0)


@dataclass
class AltMinConfig:
    """Hyperparameters for :func:`alternating_minimize`.

    ``lambda_l1`` is the sparsity knob swept for the Pareto curve; ``lambda_reg`` is the
    conservative Frobenius floor on the operators (plan default ``1e-4``). The per-phase
    step counts are inner iterations of one outer alternation.
    """

    lambda_l1: float
    lambda_reg: float = 1e-4
    outer_iters: int = 15
    operator_steps: int = 40  # Adam steps per outer iteration (inputs fixed)
    operator_lr: float = 1e-2
    input_steps: int = 25  # FISTA steps per outer iteration (operators fixed)
    batch_size: int | None = None  # examples per minibatch (None → full batch)


@dataclass
class AltMinResult:
    """Output of :func:`alternating_minimize`: refined inputs + per-iteration history."""

    u: Tensor  # [N, L, M] refined inputs on the model device
    history: list[dict[str, float]] = field(default_factory=list)
    lipschitz: float = 1.0  # final backtracking Lipschitz estimate (reuse for held-out)


def _batches(n: int, batch_size: int | None) -> Iterator[slice]:
    """Contiguous example slices. Order-independent: ops here are sums or per-example."""
    if batch_size is None or batch_size >= n:
        yield slice(0, n)
        return
    for start in range(0, n, batch_size):
        yield slice(start, min(start + batch_size, n))


def _set_requires_grad(model: LowRankSSM, flag: bool) -> None:
    for p in model.parameters():
        p.requires_grad_(flag)


def _fista(
    model: LowRankSSM,
    u0: Tensor,
    delta: Tensor,
    *,
    lambda_l1: float,
    n_steps: int,
    lipschitz: float,
    eta_inc: float = 2.0,
) -> tuple[Tensor, float]:
    """FISTA with backtracking for ``min_u ‖delta − model(u)‖² + λ‖u‖₁`` (operators fixed).

    Caller must have frozen the operators (``requires_grad=False``) so the only graph is
    ``u → recon``. ``lipschitz`` is the warm-started step-size estimate; it is returned so
    the next call (next minibatch / outer iteration) can reuse it. Pure least-squares
    (``lambda_l1=0``) reduces to accelerated gradient descent.
    """
    u = u0.detach().clone()
    y = u.clone()
    t = 1.0
    lip = lipschitz
    for _ in range(n_steps):
        yv = y.detach().requires_grad_(True)
        f_y = ((delta - model(yv)) ** 2).sum()
        (grad,) = torch.autograd.grad(f_y, yv)
        f_y = f_y.detach()
        grad = grad.detach()
        while True:  # backtrack until the quadratic upper bound holds
            u_next = soft_threshold(y - grad / lip, lambda_l1 / lip)
            with torch.no_grad():
                f_next = ((delta - model(u_next)) ** 2).sum()
            diff = u_next - y
            q = f_y + (grad * diff).sum() + 0.5 * lip * (diff * diff).sum()
            if f_next <= q + 1e-6:
                break
            lip *= eta_inc
        t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
        y = u_next + ((t - 1.0) / t_next) * (u_next - u)
        u, t = u_next, t_next
    return u, lip


def infer_inputs(
    model: LowRankSSM,
    u0: Tensor,
    delta: Tensor,
    *,
    lambda_l1: float,
    n_steps: int = 100,
    lipschitz: float = 1.0,
    batch_size: int | None = None,
) -> Tensor:
    """Solve the LASSO input update with the operators **frozen** — the held-out encoder.

    Operators are parameters that should generalise; the per-example inputs ``u`` are a
    code inferred per example. The honest generalisation test (and the Gate-3 protocol)
    therefore fits operators on train, then runs *only* this input update on test examples
    with operators frozen. Restores the operators' prior ``requires_grad`` state on exit.
    """
    device = next(model.parameters()).device
    u0 = u0.to(device)
    delta = delta.to(device)
    prior = [p.requires_grad for p in model.parameters()]
    _set_requires_grad(model, False)
    try:
        out = u0.clone()
        lip = lipschitz
        for sl in _batches(u0.shape[0], batch_size):
            out[sl], lip = _fista(model, u0[sl], delta[sl], lambda_l1=lambda_l1,
                                  n_steps=n_steps, lipschitz=lip)
        return out
    finally:
        for p, flag in zip(model.parameters(), prior, strict=True):
            p.requires_grad_(flag)


def evaluate(
    model: LowRankSSM,
    u: Tensor,
    delta: Tensor,
    *,
    lambda_l1: float = 0.0,
    lambda_reg: float = 0.0,
    batch_size: int | None = None,
) -> dict[str, float]:
    """Minibatched, no-grad metrics: ``r2``, ``recon`` (SSE), ``l1``, ``reg``, ``loss``,
    ``l0`` (fraction of nonzero inputs), ``b_norm`` and per-family operator norms.

    ``r2`` is vs the per-depth mean of ``delta`` (the intercept-only baseline, matching
    :func:`layer_dynamics.ssm.reconstruction_r2` and the Phase 1/2 convention). Minibatched
    so the full ``[N, L, M]`` input tensor is never doubled on the GPU.
    """
    device = next(model.parameters()).device
    u = u.to(device)
    delta = delta.to(device)
    mean = delta.mean(dim=0, keepdim=True)
    ss_res = ss_tot = l1 = 0.0
    nnz = total = 0
    with torch.no_grad():
        for sl in _batches(u.shape[0], batch_size):
            ub, db = u[sl], delta[sl]
            ss_res += ((db - model(ub)) ** 2).sum().item()
            ss_tot += ((db - mean) ** 2).sum().item()
            l1 += ub.abs().sum().item()
            nnz += int((ub != 0).sum().item())
            total += ub.numel()
    reg = model.operator_fro2().item()
    loss = ss_res + lambda_l1 * l1 + lambda_reg * reg
    norms = model.operator_norms()
    return {
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0,
        "recon": ss_res, "l1": l1, "reg": reg, "loss": loss,
        "l0": nnz / total if total else 0.0,
        "b_norm": float(model.b.norm()),
        "op_A": norms["A"], "op_B": norms["B"], "op_C": norms["C"], "op_D": norms["D"],
    }


def alternating_minimize(
    model: LowRankSSM,
    u_init: Tensor,
    delta: Tensor,
    cfg: AltMinConfig,
    *,
    log=None,
) -> AltMinResult:
    """Fit Branch B by alternating Adam-over-operators / FISTA-over-inputs.

    ``u_init`` are the SAE-initialised inputs (warm start); they are refined in place across
    outer iterations and returned. The operator update is full-batch (gradients summed over
    minibatches before one Adam ``step``); the input update is an exact per-example LASSO.
    Both phases minibatch over examples for memory only — the result is independent of
    ``batch_size`` for the operator phase and per-example-independent for the input phase.
    """
    device = next(model.parameters()).device
    u = u_init.to(device).clone()
    delta = delta.to(device)
    n = u.shape[0]
    opt = torch.optim.Adam(model.parameters(), lr=cfg.operator_lr)
    lip = 1.0
    history: list[dict[str, float]] = []

    rec0 = evaluate(model, u, delta, lambda_l1=cfg.lambda_l1, lambda_reg=cfg.lambda_reg,
                    batch_size=cfg.batch_size)
    rec0["iter"] = 0
    history.append(rec0)
    if log:
        log(f"  init      loss={rec0['loss']:.3e} recon={rec0['recon']:.3e} "
            f"R²={rec0['r2']:+.3f} L0={rec0['l0']:.3f}")

    for it in range(1, cfg.outer_iters + 1):
        # --- operator update: Adam on {A,B,C,D,b,z0}, inputs fixed ---
        _set_requires_grad(model, True)
        for _ in range(cfg.operator_steps):
            opt.zero_grad(set_to_none=True)
            for sl in _batches(n, cfg.batch_size):
                ((delta[sl] - model(u[sl])) ** 2).sum().backward()
            (cfg.lambda_reg * model.operator_fro2()).backward()  # reg added once
            opt.step()

        # --- input update: FISTA on {u_ℓ}, operators fixed ---
        _set_requires_grad(model, False)
        for sl in _batches(n, cfg.batch_size):
            u[sl], lip = _fista(model, u[sl], delta[sl], lambda_l1=cfg.lambda_l1,
                                n_steps=cfg.input_steps, lipschitz=lip)
        _set_requires_grad(model, True)

        rec = evaluate(model, u, delta, lambda_l1=cfg.lambda_l1, lambda_reg=cfg.lambda_reg,
                       batch_size=cfg.batch_size)
        rec["iter"] = it
        history.append(rec)
        if log:
            log(f"  iter {it:>3} loss={rec['loss']:.3e} recon={rec['recon']:.3e} "
                f"R²={rec['r2']:+.3f} L0={rec['l0']:.3f} |b|={rec['b_norm']:.2f} "
                f"|D|={rec['op_D']:.2f}")

    return AltMinResult(u=u, history=history, lipschitz=lip)
