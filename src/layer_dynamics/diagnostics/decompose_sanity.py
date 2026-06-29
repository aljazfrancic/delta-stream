"""Sanity checks for the Branch B decomposition (Phase 3 Step 4).

The plan's Step 4 (``docs/implementation-plan/phase-3-decomposition.md``) lists five
checks that apply to Branch B (item 6, the MA-stationarity recheck, is Branch A only):

  1. **Trivial-solution / collapse** — already watched inline by the optimiser
     (large ``‖b‖`` paired with vanishing operator norms); :func:`collapse_flags`
     re-derives the verdict from a stats dict so the Step-4 report is self-contained.
  2. **Reconstruction eyeball** — :func:`reconstruction_pairs` returns predicted vs
     actual ``Δ_ℓ`` for a handful of examples to plot.
  3. **Feature structure** — :func:`top_features` ranks input coordinates by total
     ``|u|`` mass with their home layer; :func:`feature_top_examples` finds the
     examples that most activate a given feature (for the subject/relation theme read).
  4. **Seed stability** — :func:`decoder_directions` reads each feature's immediate
     write into ``Δ`` (the feedthrough column ``D_ℓ[:, j]``, Branch B's decoder
     direction); :func:`direction_cosines` / :func:`match_rate` score how stable those
     directions are across re-fits.
  5. **Dataset stability** — :func:`feature_overlap` (Jaccard of top-feature sets) plus
     the same decoder-direction cosines on the shared features.

**On feature matching across re-fits.** The plan phrases seed stability as "max cosine
similarity of decoder directions", a formulation inherited from SAEs trained *from
scratch* where dictionary atoms permute between seeds. Here every fit shares the **same
frozen pretrained SAE** dictionary, so a feature coordinate ``j`` is literally the same
atom in every fit — the honest stability question is whether its *learned* readout
direction ``D_ℓ[:, j]`` stays put. We therefore report the identity-matched cosine (the
fixed-dictionary specialisation of the plan's max-cosine metric); there is no permutation
to search over.

All functions are pure (no run-dir / plotting side effects) so they unit-test cheaply;
the orchestration (fitting the extra seeds / halves) lives in ``scripts/step4_sanity.py``.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from layer_dynamics.ssm import LowRankSSM


def feature_mass(u: Tensor) -> Tensor:
    """Per-(layer, feature) activation mass ``Σ_i |u_{i,ℓ,j}|`` → ``[L, M]`` (on CPU)."""
    return u.detach().abs().sum(dim=0).cpu()


def top_features(u: Tensor, top_n: int = 20) -> list[dict[str, float]]:
    """Top input features by total ``|u|`` mass across examples and layers.

    Ranks coordinates ``j`` by ``Σ_{i,ℓ} |u_{i,ℓ,j}|`` (the plan's "largest summed
    ``|u_ℓ|`` coordinate-wise across examples and layers") and tags each with its
    **home layer** ``ℓ*`` = the layer carrying the most of that feature's mass.
    Returns a list of ``{feature, home_layer, mass, home_mass}`` dicts, mass-descending.
    """
    mass = feature_mass(u)  # [L, M]
    per_feature = mass.sum(dim=0)  # [M]
    top_n = min(top_n, per_feature.numel())
    order = torch.argsort(per_feature, descending=True)[:top_n]
    home = mass.argmax(dim=0)  # [M]
    return [
        {
            "feature": int(j),
            "home_layer": int(home[j]),
            "mass": float(per_feature[j]),
            "home_mass": float(mass[home[j], j]),
        }
        for j in order.tolist()
    ]


def feature_top_examples(u: Tensor, feature: int, layer: int, top_e: int = 10) -> list[int]:
    """Example indices with the largest ``|u_{i, layer, feature}|`` (strongest activations)."""
    col = u.detach()[:, layer, feature].abs().cpu()
    top_e = min(top_e, col.numel())
    return torch.argsort(col, descending=True)[:top_e].tolist()


def decoder_directions(model: LowRankSSM, features: list[int], layers: list[int]) -> np.ndarray:
    """Branch B decoder directions ``D_ℓ[:, j]`` for paired ``(feature j, layer ℓ)`` → ``[k, D]``.

    ``D_ℓ = D1_ℓ D2_ℓᵀ`` (low-rank), so column ``j`` is ``D1_ℓ @ D2_ℓ[j]`` — formed
    without materialising the ``D×M`` feedthrough. This is the immediate write of a unit
    of feature ``j`` into ``Δ_ℓ``, the natural Branch B analogue of an SAE/W₀ decoder
    direction (see module docstring).
    """
    if len(features) != len(layers):
        raise ValueError("features and layers must be the same length")
    with torch.no_grad():
        out = []
        for j, ell in zip(features, layers, strict=True):
            out.append((model.D1[ell] @ model.D2[ell][j]).cpu().numpy())
    return np.stack(out, axis=0) if out else np.zeros((0, model.dims.d_model), dtype=np.float32)


def direction_cosines(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise cosine similarity between two ``[k, D]`` direction stacks."""
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + eps
    return num / den


def match_rate(cosines: np.ndarray, threshold: float = 0.7) -> float:
    """Fraction of features whose direction cosine meets ``threshold`` (Gate-3 ≥ 0.7)."""
    return float((np.asarray(cosines) >= threshold).mean()) if len(cosines) else 0.0


def feature_impulse_responses(
    model: LowRankSSM, features: list[int], layers: list[int], *, batch_size: int = 64
) -> np.ndarray:
    """Gauge-invariant per-feature Δ-contribution: the SSM impulse response ``g_j ∈ R^{L×D}``.

    The Branch B SSM is **linear** in the inputs ``u``, so the data-averaged contribution of
    feature ``(home layer ℓ*, coordinate j)`` to ``Δ`` is — up to the (cosine-irrelevant)
    scalar mean activation — its *impulse response*: feed a unit activation at ``(ℓ*, j)`` and
    read the resulting ``Δ̂`` at every layer, with the input-free baseline (``b`` + the
    ``z₀``-driven ``Cz`` term) differenced out. Unlike the raw decoder column ``D_{ℓ*}[:, j]``
    (:func:`decoder_directions`) — only the *home-layer* feedthrough write, which shifts under
    the SSM's state↔feedthrough gauge freedom — the full ``g_j`` (feedthrough at ``ℓ*`` **plus**
    the downstream state propagation ``C_ℓ A…A B_{ℓ*}`` at ``ℓ > ℓ*``) is a property of the
    input→output map and so is gauge-invariant. Its home-layer slice ``g_j[ℓ*]`` equals
    ``D_{ℓ*}[:, j]`` exactly (a consistency check).

    Returns ``[k, L, D]`` (k = len(features)); flatten over (L, D) for a direction cosine.
    """
    if len(features) != len(layers):
        raise ValueError("features and layers must be the same length")
    k = len(features)
    L, D, M = model.dims.n_layers, model.dims.d_model, model.dims.d_input
    device = next(model.parameters()).device
    out = np.zeros((k, L, D), dtype=np.float32)
    if k == 0:
        return out
    was_training = model.training
    model.eval()
    with torch.no_grad():
        base = model(torch.zeros(1, L, M, device=device))  # [1, L, D] input-free baseline
        for start in range(0, k, batch_size):
            sel = list(range(start, min(start + batch_size, k)))
            u = torch.zeros(len(sel), L, M, device=device)
            for row, t in enumerate(sel):
                u[row, layers[t], features[t]] = 1.0
            out[start : start + len(sel)] = (model(u) - base).cpu().numpy()
    if was_training:
        model.train()
    return out


def _orthonormal_basis(a: np.ndarray, rtol: float = 1e-8) -> np.ndarray:
    """Orthonormal basis ``[D, rank]`` for the span of the rows of ``a`` (a ``[k, D]`` stack)."""
    if a.size == 0:
        return np.zeros((a.shape[1] if a.ndim == 2 else 0, 0))
    u, s, _ = np.linalg.svd(a.T, full_matrices=False)  # a.T is [D, k]; left vectors span the rows
    tol = s.max() * rtol if len(s) else 0.0
    return u[:, s > tol]


def subspace_overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Grassmann overlap of the subspaces spanned by the rows of two ``[k, D]`` stacks.

    Orthonormalises each row set and returns the **mean cosine of the principal angles** (mean
    singular value of ``Qaᵀ Qb``) ∈ [0, 1]: 1 = identical span, 0 = orthogonal. Unlike the
    per-direction cosine (:func:`direction_cosines`), this is invariant to how the directions
    are rotated *within* their joint span, so it scores whether two seeds find the same feature
    *subspace* even when individual directions are gauge-shuffled across that subspace.
    """
    qa = _orthonormal_basis(a)
    qb = _orthonormal_basis(b)
    if qa.shape[1] == 0 or qb.shape[1] == 0:
        return 0.0
    s = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return float(np.clip(s, 0.0, 1.0).mean())


def state_eigenvalue_moduli(model: LowRankSSM) -> np.ndarray:
    """Descending moduli of the eigenvalues of each transition ``A_ℓ`` → ``[L-1, d_z]``.

    ``A_ℓ = A1_ℓ A2_ℓᵀ``. Eigenvalues are invariant under the SSM's state similarity gauge
    ``A → T A T⁻¹`` (Symmetry A), so the modulus spectrum is a gauge-invariant summary of the
    cross-layer state dynamics. Per-layer moduli are sorted descending for cross-seed matching
    (compare two seeds' spectra row-wise with :func:`direction_cosines`).
    """
    with torch.no_grad():
        out = []
        for ell in range(model.dims.n_layers - 1):
            mat = (model.A1[ell] @ model.A2[ell].transpose(-1, -2)).cpu().numpy()
            out.append(np.sort(np.abs(np.linalg.eigvals(mat)))[::-1])
    return np.stack(out, axis=0) if out else np.zeros((0, model.dims.d_state))


def feature_overlap(features_a: list[int], features_b: list[int]) -> dict[str, float]:
    """Jaccard + intersection size of two top-feature index sets (dataset stability)."""
    sa, sb = set(features_a), set(features_b)
    union = sa | sb
    inter = sa & sb
    return {
        "jaccard": len(inter) / len(union) if union else 0.0,
        "intersection": len(inter),
        "n_a": len(sa),
        "n_b": len(sb),
    }


def reconstruction_pairs(
    model: LowRankSSM, u: Tensor, delta: Tensor, idx: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Predicted and actual ``Δ_ℓ`` for the chosen examples → two ``[len(idx), L, D]`` arrays."""
    device = next(model.parameters()).device
    sel = torch.as_tensor(idx)
    with torch.no_grad():
        pred = model(u[sel].to(device)).cpu().numpy()
    return pred, delta[sel].cpu().numpy()


def collapse_flags(stats: dict, *, op_tol: float = 1e-3, l0_tol: float = 1e-4) -> dict:
    """Re-derive the collapse verdict (Step 4 #1) from a train-metrics dict.

    The degenerate solution is ``{u}=0`` with the operators near-zero and ``b`` absorbing
    the per-layer mean. We flag it when the inputs are essentially empty (``l0 < l0_tol``)
    **and** the stacked operator norm is negligible relative to ``‖b‖``. Returns the
    component norms alongside the boolean so the report can show *why*.
    """
    op_total = float(
        (stats["op_A"] ** 2 + stats["op_B"] ** 2 + stats["op_C"] ** 2 + stats["op_D"] ** 2)
        ** 0.5
    )
    b_norm = float(stats["b_norm"])
    collapsed = bool(stats["l0"] < l0_tol and op_total < op_tol * b_norm)
    return {
        "operator_total_fro": op_total,
        "b_norm": b_norm,
        "l0": float(stats["l0"]),
        "collapsed": collapsed,
    }
