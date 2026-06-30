"""Residual-write Cross-Layer Transcoder baseline (Phase 4 Step-1 Baseline D(b)).

The *primary* conceptual comparison of [proposal §4.2]: CLT places a **free** decoder
matrix at every (source, target) layer pair ``ℓ' ≤ ℓ`` — the unconstrained ``O(L²)``
parameterisation our Branch-B state-space model is meant to be a parsimonious
reparameterisation of. Here we fit the **residual-write variant** — decoder
contributions summed into ``Δ_ℓ`` rather than into MLP outputs — on the *same cached
per-layer sparse sources* the SSM uses, so the only thing that differs is the decoder
parameterisation (free per-pair vs propagated low-rank state):

    Δ̂_ℓ = b_ℓ + Σ_{ℓ' = max(0, ℓ-reach)}^{ℓ}  U^{ℓ'→ℓ} (V^{ℓ'→ℓ})ᵀ u_{ℓ'}

with each ``W^{ℓ'→ℓ} = U^{ℓ'→ℓ} (V^{ℓ'→ℓ})ᵀ`` low-rank (rank ``r``, matched to the SSM).
``reach = L-1`` gives the full lower-triangular CLT; ``reach = 0`` gives the per-layer
(diagonal-only) baseline, which is the same family with no cross-layer term.

This lives entirely on cached sources (no transformer forward), so it fits the 3 GB
dev box: it is a linear low-rank ridge in ``u`` solved by gradient descent.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ResidualWriteCLT(nn.Module):
    """Unconstrained ``O(L²)`` residual-write CLT over fixed per-layer sources.

    Parameters are stacked rank-``r`` factor tensors ``U[P, D, r]`` and ``V[P, M, r]``
    over the ``P`` causal (source ≤ target) pairs, plus a per-layer bias ``b[L, D]``
    that absorbs the per-depth mean ``Δ̄_ℓ`` (same role as the SSM's ``b_ℓ``).
    """

    def __init__(self, n_layers: int, d_model: int, n_sources: int, rank: int,
                 *, reach: int | None = None, init_scale: float = 0.01, seed: int = 0):
        super().__init__()
        L = n_layers
        self.L, self.D, self.M, self.r = L, d_model, n_sources, rank
        self.reach = L - 1 if reach is None else reach
        pairs = [(lp, t) for t in range(L) for lp in range(max(0, t - self.reach), t + 1)]
        self.pairs = pairs
        p = len(pairs)
        g = torch.Generator().manual_seed(seed)
        self.U = nn.Parameter(torch.randn(p, d_model, rank, generator=g) * init_scale)
        self.V = nn.Parameter(torch.randn(p, n_sources, rank, generator=g) * init_scale)
        self.b = nn.Parameter(torch.zeros(L, d_model))
        # Pairs are generated target-major, so each target's sources are a contiguous
        # block of the pair axis: store (target, slice) and the source index per pair.
        self.register_buffer("src_index", torch.tensor([lp for lp, _ in pairs], dtype=torch.long))
        self._tgt_slices: list[tuple[int, slice]] = []
        i = 0
        for ell in range(L):
            n = sum(1 for _, t in pairs if t == ell)
            self._tgt_slices.append((ell, slice(i, i + n)))
            i += n

    def forward(self, u: Tensor) -> Tensor:
        """``u[B, L, M]`` → ``Δ̂[B, L, D]``."""
        out = self.b[None].expand(u.shape[0], self.L, self.D).clone()
        for ell, sl in self._tgt_slices:
            srcs = self.src_index[sl]                                   # [S]
            us = u[:, srcs, :]                                          # [B, S, M]
            proj = torch.einsum("bsm,smr->bsr", us, self.V[sl])         # [B, S, r]
            out[:, ell] = out[:, ell] + torch.einsum("bsr,sdr->bd", proj, self.U[sl])
        return out

    def operator_fro2(self) -> Tensor:
        """Exact ``Σ ‖U Vᵀ‖_F² = Σ ⟨UᵀU, VᵀV⟩`` — never materialises a ``D×M`` block."""
        utu = torch.einsum("pdr,pds->prs", self.U, self.U)             # [P, r, r]
        vtv = torch.einsum("pmr,pms->prs", self.V, self.V)
        return (utu * vtv).sum()

    def n_params(self) -> int:
        return self.U.numel() + self.V.numel() + self.b.numel()
