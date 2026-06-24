"""Branch B decomposition: a first-order, fully depth-indexed low-rank SSM (Phase 3).

Gate 2 locked Branch B (`runs/gate2/phase3_architecture.md`): a first-order
state-space model over *depth* that generates the residual updates from per-layer
sparse inputs ``u_ℓ``. For ``ℓ = 0 … L-2``

    z_{ℓ+1} = A_ℓ z_ℓ + B_ℓ u_ℓ                         (state transition)

and for ``ℓ = 0 … L-1``

    Δ_ℓ ≈ b_ℓ + C_ℓ z_ℓ + D_ℓ u_ℓ                       (observation / decode)

with the initial state ``z_0`` a free parameter or fixed at zero. Every operator
is **free per layer** (the depth-dependence is pervasive — Gate-2 Step 1) and
**low-rank** via factorisation ``A_ℓ = A_ℓ^{(1)} (A_ℓ^{(2)})^⊤`` (analogously
``B, C, D``); the rank ``r`` is swept. ``d_z`` is the latent-state dimension, ``M``
the input (SAE dictionary) width, ``D`` the residual dimension. The feedthrough
``D_ℓ u_ℓ`` is the Branch-A ``W_0`` analogue (keeps Branch B from being a
strictly-weaker, one-lag-shifted predictor); ``b_ℓ`` is the unpenalised per-layer
bias absorbing ``\bar Δ_ℓ``.

All matvecs route through the rank-``r`` bottleneck (``u @ B2`` then ``@ B1ᵀ``), so
the ``D×M`` / ``d_z×M`` operators are never materialised — essential at ``M`` ~ 32k.

This module is the **model and loss only**; the alternating optimiser (Adam over
operators / proximal-gradient LASSO over inputs) is the next step. The consolidated
loss matches the plan's ``L_B``: reconstruction MSE + ``λ_L1`` on the inputs +
``λ_reg`` Frobenius on the operators (``b_ℓ``, ``z_0`` unpenalised).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


def _factor(n: int, rows: int, cols: int, r: int, scale: float, gen: torch.Generator) -> Tensor:
    """A stacked low-rank factor ``[n, rows, r]`` (or ``[n, cols, r]``) init ~N(0, scale²)."""
    return torch.randn(n, rows, r, generator=gen) * scale


@dataclass
class SSMDims:
    """Shape bundle for :class:`LowRankSSM`."""

    n_layers: int  # L (number of Δ targets / observations)
    d_model: int  # D
    d_input: int  # M
    d_state: int  # d_z
    rank: int  # r


class LowRankSSM(nn.Module):
    """First-order depth-indexed SSM with per-layer low-rank operators (Branch B).

    Parameters are stored as stacked rank-``r`` factor tensors so a forward pass is a
    plain depth loop. ``A,B`` exist for ``ℓ = 0 … L-2`` (they produce ``z_1 … z_{L-1}``);
    ``C,D,b`` exist for ``ℓ = 0 … L-1``. ``z_0`` is learned unless ``learn_z0=False``.
    """

    def __init__(self, dims: SSMDims, *, learn_z0: bool = True, init_scale: float = 0.02,
                 seed: int = 0, use_state: bool = True):
        super().__init__()
        self.dims = dims
        # use_state=False ablates the cross-layer state chain (A,B,C drop out of the
        # forward), leaving Δ_ℓ ≈ b_ℓ + D_ℓ u_ℓ — the per-layer sparse-dictionary
        # baseline for the Gate-3 Pareto (a per-layer SAE on Δ_ℓ, same optimiser/L1
        # sweep). A,B,C remain as (inert) parameters; they never enter the graph so the
        # recon gradient on them is zero and only the λ_reg floor touches them.
        self.use_state = use_state
        L, D, M, dz, r = (dims.n_layers, dims.d_model, dims.d_input, dims.d_state, dims.rank)
        g = torch.Generator().manual_seed(seed)
        s = init_scale
        # Transition (ℓ=0..L-2): A_ℓ = A1 A2ᵀ [dz×dz], B_ℓ = B1 B2ᵀ [dz×M].
        self.A1 = nn.Parameter(_factor(L - 1, dz, dz, r, s, g))
        self.A2 = nn.Parameter(_factor(L - 1, dz, dz, r, s, g))
        self.B1 = nn.Parameter(_factor(L - 1, dz, M, r, s, g))
        self.B2 = nn.Parameter(_factor(L - 1, M, dz, r, s, g))
        # Observation (ℓ=0..L-1): C_ℓ = C1 C2ᵀ [D×dz], D_ℓ = D1 D2ᵀ [D×M].
        self.C1 = nn.Parameter(_factor(L, D, dz, r, s, g))
        self.C2 = nn.Parameter(_factor(L, dz, D, r, s, g))
        self.D1 = nn.Parameter(_factor(L, D, M, r, s, g))
        self.D2 = nn.Parameter(_factor(L, M, D, r, s, g))
        self.b = nn.Parameter(torch.zeros(L, D))
        z0 = torch.zeros(dz)
        self.z0 = nn.Parameter(z0) if learn_z0 else None
        self.register_buffer("_z0_fixed", z0 if not learn_z0 else torch.empty(0))

    def _z0(self, batch: int) -> Tensor:
        z0 = self.z0 if self.z0 is not None else self._z0_fixed
        return z0.unsqueeze(0).expand(batch, -1)

    def forward(self, u: Tensor) -> Tensor:
        """Inputs ``u[batch, L, M]`` → predicted updates ``Δ̂[batch, L, D]``.

        Unrolls the state recursion through depth; every operator application goes
        through the rank-``r`` factors, so no ``D×M`` matrix is formed.
        """
        L = self.dims.n_layers
        if not self.use_state:
            # No-state baseline: Δ_ℓ = b_ℓ + (u D2)D1ᵀ per layer, no state chain.
            return torch.stack(
                [self.b[ell] + (u[:, ell] @ self.D2[ell]) @ self.D1[ell].T for ell in range(L)],
                dim=1,
            )
        z = self._z0(u.shape[0])  # z_0  [batch, dz]
        out = []
        for ell in range(L):
            # Δ_ℓ = b_ℓ + (z C2)C1ᵀ + (u D2)D1ᵀ
            delta = (
                self.b[ell]
                + (z @ self.C2[ell]) @ self.C1[ell].T
                + (u[:, ell] @ self.D2[ell]) @ self.D1[ell].T
            )
            out.append(delta)
            if ell < L - 1:
                # z_{ℓ+1} = (z A2)A1ᵀ + (u B2)B1ᵀ
                z = (z @ self.A2[ell]) @ self.A1[ell].T + (
                    u[:, ell] @ self.B2[ell]) @ self.B1[ell].T
        return torch.stack(out, dim=1)  # [batch, L, D]

    def operator_fro2(self) -> Tensor:
        """Σ ‖A_ℓ‖_F² + ‖B_ℓ‖_F² + ‖C_ℓ‖_F² + ‖D_ℓ‖_F² of the *products* (low-rank-exact).

        ‖F1 F2ᵀ‖_F² = ⟨F1ᵀF1, F2ᵀF2⟩ — computed in the ``r×r`` space, never forming the
        product. This is the plan's Frobenius penalty on the operators (not the factors).
        """
        total = torch.zeros((), device=self.b.device)
        for f1, f2 in ((self.A1, self.A2), (self.B1, self.B2), (self.C1, self.C2),
                       (self.D1, self.D2)):
            g1 = f1.transpose(-1, -2) @ f1  # [n, r, r]
            g2 = f2.transpose(-1, -2) @ f2  # [n, r, r]
            total = total + (g1 * g2).sum()
        return total

    def operator_norms(self) -> dict[str, float]:
        """Per-family stacked Frobenius norms ``√(Σ_ℓ ‖·_ℓ‖_F²)`` for ``A,B,C,D``.

        Logging-only (no graph): the Gate-3 collapse check (Phase 3 Step 4 #1) watches
        these alongside ``‖b‖`` — a large bias paired with vanishing operator norms is
        the tell for the degenerate "biases absorb the per-layer mean" solution.
        """
        out: dict[str, float] = {}
        with torch.no_grad():
            for name, (f1, f2) in {"A": (self.A1, self.A2), "B": (self.B1, self.B2),
                                   "C": (self.C1, self.C2), "D": (self.D1, self.D2)}.items():
                g1 = f1.transpose(-1, -2) @ f1
                g2 = f2.transpose(-1, -2) @ f2
                out[name] = float((g1 * g2).sum().clamp_min(0.0).sqrt())
        return out


def ssm_loss(
    model: LowRankSSM,
    u: Tensor,
    delta: Tensor,
    *,
    lambda_l1: float,
    lambda_reg: float,
) -> dict[str, Tensor]:
    """Consolidated Branch B loss ``L_B`` and its components.

    Returns ``{loss, recon, l1, reg}`` (all scalars). ``recon`` is the summed squared
    error over examples × depths (matching the plan's ``Σ_i Σ_ℓ ‖·‖²``); ``l1`` is the
    summed ``‖u_ℓ‖_1``; ``reg`` is the operator Frobenius². ``b_ℓ``/``z_0`` are not
    penalised.
    """
    pred = model(u)
    recon = ((delta - pred) ** 2).sum()
    l1 = u.abs().sum()
    reg = model.operator_fro2()
    loss = recon + lambda_l1 * l1 + lambda_reg * reg
    return {"loss": loss, "recon": recon, "l1": l1, "reg": reg}


def reconstruction_r2(model: LowRankSSM, u: Tensor, delta: Tensor) -> float:
    """Held-out R² of the SSM reconstruction vs the per-depth mean of ``delta``."""
    with torch.no_grad():
        pred = model(u)
        mean = delta.mean(dim=0, keepdim=True)  # per-depth mean (intercept-only baseline)
        ss_res = ((delta - pred) ** 2).sum().item()
        ss_tot = ((delta - mean) ** 2).sum().item()
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
