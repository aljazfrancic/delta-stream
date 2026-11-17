"""Depth-causal codes for the state probe (v1.1 post-hoc check A).

The recorded state probe (``scripts/phase4_state_probe.py``) reads ``u_infer_full``:
LASSO codes inferred for each example against its **whole** update trajectory. The
reconstruction of ``Δ_ℓ`` reaches the codes below layer ℓ only through the state
``z_ℓ``, so whatever those codes pick up about ``Δ_ℓ`` and later updates is written along
the directions the state map reads. The probe's state arm can therefore see the write it
is supposed to precede (paper §6.7).

This module infers the codes a *depth-causal* state would be built from. For a prefix
length ``p`` the same LASSO is solved with the same frozen operators, the same ``λ``,
the same initial codes and the same number of FISTA steps, but against ``Δ_0 … Δ_{p-1}``
only. The state entering observation ``p`` that those codes produce is a function of the
first ``p`` updates and of nothing later. Because ``Δ̂_ℓ`` depends on ``u_0 … u_ℓ`` alone,
truncating the unroll is exact, and at ``p = L`` the procedure is the recorded held-out
encoder (:func:`layer_dynamics.decompose.infer_inputs`) — which is the built-in check
that the truncation changes nothing else.

Pure functions; orchestration lives in ``scripts/v11_causal_codes.py``.
"""

from __future__ import annotations

import torch
from torch import Tensor

from layer_dynamics.decompose import _batches, _fista, _set_requires_grad
from layer_dynamics.ssm import LowRankSSM


class PrefixSSM:
    """The first ``p`` observations of a :class:`LowRankSSM`: ``u[:, :p] → Δ̂[:, :p]``.

    Mirrors :meth:`LowRankSSM.forward` operation for operation, so at ``p = L`` the two
    agree exactly. A callable, not a module: it only borrows the model's parameters.
    """

    def __init__(self, model: LowRankSSM, p: int):
        if not 1 <= p <= model.dims.n_layers:
            raise ValueError(f"prefix length {p} outside 1..{model.dims.n_layers}")
        self.model = model
        self.p = p

    def __call__(self, u: Tensor) -> Tensor:
        m = self.model
        if u.shape[1] != self.p:
            raise ValueError(f"expected {self.p} input layers, got {u.shape[1]}")
        if not m.use_state:
            return torch.stack(
                [m.b[ell] + (u[:, ell] @ m.D2[ell]) @ m.D1[ell].T for ell in range(self.p)],
                dim=1,
            )
        z = m._z0(u.shape[0])
        out = []
        for ell in range(self.p):
            delta = (
                m.b[ell]
                + (z @ m.C2[ell]) @ m.C1[ell].T
                + (u[:, ell] @ m.D2[ell]) @ m.D1[ell].T
            )
            out.append(delta)
            if ell < self.p - 1:
                z = (z @ m.A2[ell]) @ m.A1[ell].T + (
                    u[:, ell] @ m.B2[ell]) @ m.B1[ell].T
        return torch.stack(out, dim=1)


def infer_prefix_inputs(
    model: LowRankSSM,
    u0: Tensor,
    delta: Tensor,
    p: int,
    *,
    lambda_l1: float,
    n_steps: int = 100,
    lipschitz: float = 1.0,
    batch_size: int | None = None,
) -> Tensor:
    """LASSO codes for the first ``p`` layers, inferred against ``Δ_0 … Δ_{p-1}`` only.

    Same solver, warm start, step count and Lipschitz hand-over as
    :func:`layer_dynamics.decompose.infer_inputs`; the one difference is the truncated
    target. ``u0`` and ``delta`` may be full-length (``[N, L, ·]``) — only their first
    ``p`` layers are read. Batches are moved to the model's device one at a time and the
    result is returned on the CPU, so the full code tensor never sits on the GPU.
    """
    device = next(model.parameters()).device
    fwd = PrefixSSM(model, p)
    prior = [prm.requires_grad for prm in model.parameters()]
    _set_requires_grad(model, False)
    try:
        out = u0[:, :p].detach().cpu().clone()
        lip = lipschitz
        for sl in _batches(u0.shape[0], batch_size):
            ub, lip = _fista(fwd, u0[sl, :p].to(device), delta[sl, :p].to(device),
                             lambda_l1=lambda_l1, n_steps=n_steps, lipschitz=lip)
            out[sl] = ub.detach().cpu()
        return out
    finally:
        for prm, flag in zip(model.parameters(), prior, strict=True):
            prm.requires_grad_(flag)


@torch.no_grad()
def prefix_state(model: LowRankSSM, u_prefix: Tensor) -> Tensor:
    """The state entering observation ``p`` given codes for layers ``0 … p-1``.

    Equals ``model.states(u)[:, p]`` for any ``u`` whose first ``p`` layers are
    ``u_prefix``. Requires ``1 ≤ p ≤ L-1`` (there is no transition out of the last layer).
    """
    if not model.use_state:
        raise ValueError("prefix_state() is undefined for the use_state=False baseline")
    p = u_prefix.shape[1]
    if not 1 <= p <= model.dims.n_layers - 1:
        raise ValueError(f"prefix length {p} outside 1..{model.dims.n_layers - 1}")
    z = model._z0(u_prefix.shape[0])
    for ell in range(p):
        z = (z @ model.A2[ell]) @ model.A1[ell].T + (
            u_prefix[:, ell] @ model.B2[ell]) @ model.B1[ell].T
    return z


@torch.no_grad()
def feedthrough_image(model: LowRankSSM, u_layer: Tensor, layer: int) -> Tensor:
    """The feedthrough's rank-``r`` image of one layer's code: ``u_ℓ D2_ℓ`` → ``[N, r]``.

    ``D_ℓ u_ℓ = D1_ℓ (D2_ℓᵀ u_ℓ)``, so this is the ``r``-dimensional bottleneck through
    which the per-layer path writes ``Δ_ℓ``: a per-layer, reconstruction-trained linear
    image of the code, the matched comparator for the ``d_z``-dimensional state.
    """
    return u_layer @ model.D2[layer]
