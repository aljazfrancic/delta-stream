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
