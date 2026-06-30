"""Characterise the learned Branch-B state-transition operator A_ℓ (Phase 4).

The reconstruction win (Gate 3 / Step 1) and the non-stationarity result (H2) say the
cross-layer *state* is load-bearing but depth-varying; this module describes the
operator that carries it. ``A_ℓ = A1_ℓ A2_ℓᵀ`` maps ``z_ℓ → z_{ℓ+1}``. Its
**eigenvalues are gauge-invariant** — Symmetry A (``A → T A T⁻¹``) leaves the spectrum
fixed — which is why S1 found the modulus spectrum seed-stable, so the spectrum is the
robust object to report. Two readouts:

  * :func:`spectrum_by_depth` — per-layer eigenvalues and summaries (spectral radius
    ``ρ``, mean ``|λ|``, fraction complex). ``ρ < 1`` everywhere ⇒ a contractive
    (stable) state that forgets with depth.
  * :func:`state_path_decay` — the operator-level memory kernel
    ``‖C_ℓ (∏A) B_j‖_F`` by depth gap ``g = ℓ − j``. A fast decay is the SSM-native
    explanation of the reconstruction ``k*≈1``: a contractive ``A`` means a source
    injected ≥2 layers back barely reaches ``Δ_ℓ`` through the state.

Everything is computed in the tiny ``d_z×d_z`` latent space (the ``D×M`` operators are
never formed): ``‖C P B‖_F² = tr( (CᵀC) P (BBᵀ) Pᵀ )`` collapses to a 32×32 trace.
"""

from __future__ import annotations

import numpy as np


def a_operators(model) -> np.ndarray:
    """Stacked state-transition matrices ``A_ℓ = A1_ℓ A2_ℓᵀ`` → ``[L-1, d_z, d_z]``."""
    a1 = model.A1.detach().cpu().numpy()  # [L-1, dz, r]
    a2 = model.A2.detach().cpu().numpy()  # [L-1, dz, r]
    return np.einsum("lir,ljr->lij", a1, a2)  # A1 @ A2ᵀ per layer


def spectrum_by_depth(model) -> dict:
    """Per-layer eigenvalues of ``A_ℓ`` and gauge-invariant summaries.

    Returns ``rho`` (spectral radius per layer), ``mean_mod`` (mean ``|λ|``),
    ``frac_complex`` (share of eigenvalues with non-negligible imaginary part),
    ``eig_real``/``eig_imag`` (``[L-1, d_z]`` for the complex-plane plot) and
    ``moduli_sorted`` (all ``|λ|`` pooled and sorted desc, for cross-seed cosine).
    """
    a = a_operators(model)
    n, dz, _ = a.shape
    eig = np.array([np.linalg.eigvals(a[ell]) for ell in range(n)])  # [n, dz] complex
    mod = np.abs(eig)
    rho = mod.max(axis=1)
    mean_mod = mod.mean(axis=1)
    frac_complex = (np.abs(eig.imag) > 1e-8 * np.maximum(np.abs(eig.real), 1.0)).mean(axis=1)
    return {
        "n_layers_trans": int(n), "d_state": int(dz),
        "rho": rho, "mean_mod": mean_mod, "frac_complex": frac_complex,
        "eig_real": eig.real, "eig_imag": eig.imag,
        "moduli_sorted": np.sort(mod.reshape(-1))[::-1],
        "rho_geomean": float(np.exp(np.log(np.clip(rho, 1e-12, None)).mean())),
        "rho_median": float(np.median(rho)), "rho_max": float(rho.max()),
        "max_over_unit": float((rho >= 1.0).mean()),  # fraction of layers non-contractive
    }


def state_path_decay(model, *, max_gap: int = 8) -> dict:
    """Operator memory kernel: mean ``‖C_ℓ (∏_{i} A_i) B_j‖_F`` by depth gap ``g=ℓ−j``.

    ``u_j`` reaches ``Δ_ℓ`` through the state as ``C_ℓ (A_{ℓ-1}…A_{j+1}) B_j`` (gap
    ``g=ℓ−j≥1``; ``g=1`` is ``C_{j+1} B_j`` with an empty ``A`` product). The Frobenius
    norm is evaluated as ``tr( (C_ℓᵀC_ℓ) P (B_jB_jᵀ) Pᵀ )`` entirely in ``d_z`` space.
    Returns per-gap mean/median Frobenius norm and the ratio to gap 1 — a fast decay is
    the operator-level signature of the reconstruction ``k*≈1``.
    """
    a = a_operators(model)  # [L-1, dz, dz]
    n_trans, dz, _ = a.shape
    # CᵀC_ℓ for ℓ=0..L-1 (C1[D,r], C2[dz,r] → CᵀC = C2 (C1ᵀC1) C2ᵀ, in dz space).
    c1 = model.C1.detach().cpu().numpy()  # [L, D, r]
    c2 = model.C2.detach().cpu().numpy()  # [L, dz, r]
    ctc = np.einsum("lir,lrs,ljs->lij", c2, c1.transpose(0, 2, 1) @ c1, c2)  # [L, dz, dz]
    # BBᵀ_j for j=0..L-2 (B1[dz,r], B2[M,r] → BBᵀ = B1 (B2ᵀB2) B1ᵀ, in dz space).
    b1 = model.B1.detach().cpu().numpy()  # [L-1, dz, r]
    b2 = model.B2.detach().cpu().numpy()  # [L-1, M, r]
    bbt = np.einsum("lir,lrs,ljs->lij", b1, b2.transpose(0, 2, 1) @ b2, b1)  # [L-1, dz, dz]

    max_gap = min(max_gap, n_trans)
    by_gap: dict[int, list[float]] = {g: [] for g in range(1, max_gap + 1)}
    for j in range(n_trans):  # source layer j (has a B_j)
        p = np.eye(dz)
        for g in range(1, max_gap + 1):
            ell = j + g
            if ell > n_trans:  # need C_ℓ with ℓ ≤ L-1 = n_trans
                break
            # P = A_{ℓ-1} … A_{j+1}; for g=1 it's I, then left-multiply A_{j+g-1} each step.
            if g >= 2:
                p = a[j + g - 1] @ p
            fro2 = float(np.trace(ctc[ell] @ p @ bbt[j] @ p.T))
            by_gap[g].append(max(fro2, 0.0) ** 0.5)

    gaps = [g for g in range(1, max_gap + 1) if by_gap[g]]
    mean_fro = np.array([float(np.mean(by_gap[g])) for g in gaps])
    median_fro = np.array([float(np.median(by_gap[g])) for g in gaps])
    ratio = mean_fro / mean_fro[0] if mean_fro.size else mean_fro
    return {"gaps": gaps, "mean_fro": mean_fro, "median_fro": median_fro,
            "ratio_to_gap1": ratio,
            "decay_gap2_over_gap1": float(ratio[1]) if ratio.size > 1 else float("nan")}


def spectrum_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two equal-length sorted-modulus spectra (cross-seed metric)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0
