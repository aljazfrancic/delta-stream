"""Per-layer SAE sources for the Branch B decomposition (Phase 3 Step 2).

The Branch B inputs ``u_{ℓ'}`` initialise from per-layer sparse-autoencoder latents
of the residual state ``v_{ℓ'}`` (Phase 3 Step 2). For the **70m-deduped debug
lane** these SAEs are public — ``EleutherAI/sae-pythia-70m-deduped-32k`` — and
stored as plain safetensors (``encoder.weight/bias``, ``W_dec``, ``b_dec``) plus a
``cfg.json`` recording the TopK width ``k``. This module loads them directly and
implements the TopK encode/decode, so the heavy ``sparsify`` runtime is not a
dependency (it also has no verified Pascal/GTX-1050 path).

**Hookpoint ↔ residual-state indexing.** The suite stores one SAE per block output:
``layers.N`` hooks ``blocks.N.hook_resid_post`` = our ``v_{N+1}`` (``v_0`` is
``blocks.0.hook_resid_pre``; see :mod:`layer_dynamics.activations`). So the SAE that
matches source state ``v_{ℓ'}`` is ``layers.{ℓ'−1}`` for ``ℓ' ≥ 1``; ``v_0`` (the
embedding) has **no** matched SAE in the suite — :func:`source_hookpoint` returns a
flagged proxy for it (debug-lane only). There is also a spare ``layers.{L−1}``
(``v_L``), which is never a source (``v_L`` has no downstream update to write into).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_SAE_REPO = "EleutherAI/sae-pythia-70m-deduped-32k"


@dataclass
class TopKSAE:
    """A loaded TopK SAE: encoder/decoder weights and the TopK width ``k``.

    Shapes: ``W_enc[M, D]``, ``b_enc[M]``, ``W_dec[M, D]``, ``b_dec[D]`` where ``M``
    is the dictionary width (num latents) and ``D`` the residual dimension.
    """

    hookpoint: str
    W_enc: np.ndarray
    b_enc: np.ndarray
    W_dec: np.ndarray
    b_dec: np.ndarray
    k: int

    @property
    def d_in(self) -> int:
        return int(self.b_dec.shape[0])

    @property
    def n_latents(self) -> int:
        return int(self.b_enc.shape[0])


def load_sae(repo: str, hookpoint: str, *, cache_dir: str | Path | None = None) -> TopKSAE:
    """Download (cached) and load one ``hookpoint`` SAE from ``repo``."""
    from huggingface_hub import hf_hub_download
    from safetensors.numpy import load_file

    kw = {"cache_dir": str(cache_dir)} if cache_dir is not None else {}
    cfg = json.loads(Path(hf_hub_download(repo, f"{hookpoint}/cfg.json", **kw)).read_text())
    t = load_file(hf_hub_download(repo, f"{hookpoint}/sae.safetensors", **kw))
    return TopKSAE(
        hookpoint=hookpoint,
        W_enc=t["encoder.weight"].astype(np.float32),
        b_enc=t["encoder.bias"].astype(np.float32),
        W_dec=t["W_dec"].astype(np.float32),
        b_dec=t["b_dec"].astype(np.float32),
        k=int(cfg["k"]),
    )


def encode(sae: TopKSAE, x: np.ndarray, *, relu: bool = True) -> np.ndarray:
    """TopK-sparse latents of ``x[..., D]`` → ``z[..., M]`` (exactly ``k`` nonzeros/row).

    Pre-activations are ``(x − b_dec) W_encᵀ + b_enc`` (the standard pre-encoder
    decoder-bias subtraction); ``relu`` clamps negatives before the TopK selection
    (``signed=False`` in the suite's config). Only the top-``k`` coordinates per row
    are kept, matching the TopK SAE forward.
    """
    pre = (x - sae.b_dec) @ sae.W_enc.T + sae.b_enc
    if relu:
        pre = np.maximum(pre, 0.0)
    flat = pre.reshape(-1, sae.n_latents)
    z = np.zeros_like(flat)
    idx = np.argpartition(flat, -sae.k, axis=1)[:, -sae.k :]
    rows = np.arange(flat.shape[0])[:, None]
    z[rows, idx] = flat[rows, idx]
    return z.reshape(*x.shape[:-1], sae.n_latents)


def decode(sae: TopKSAE, z: np.ndarray) -> np.ndarray:
    """Reconstruct ``z[..., M] W_dec + b_dec`` → ``x̂[..., D]``."""
    return z @ sae.W_dec + sae.b_dec


def reconstruction_r2(sae: TopKSAE, x: np.ndarray) -> float:
    """Held-in reconstruction R² of ``x`` under encode→decode (sanity for the loader)."""
    xhat = decode(sae, encode(sae, x))
    ss_res = float(np.sum((x - xhat) ** 2))
    ss_tot = float(np.sum((x - x.mean(axis=0)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def source_hookpoint(source_layer: int, n_layers: int) -> tuple[str, bool]:
    """SAE hookpoint matching source state ``v_{ℓ'}`` → ``(hookpoint, matched)``.

    ``v_{ℓ'}`` for ``ℓ' ≥ 1`` matches ``layers.{ℓ'−1}`` (``matched=True``). ``v_0``
    has no SAE in the suite; we return ``layers.0`` as a **proxy** (``matched=False``)
    so the debug pipeline has a source there — flag it and do not read interpretability
    into the ``v_0`` source. ``ℓ'`` must be in ``[0, n_layers−1]`` (no source at ``v_L``).
    """
    if not 0 <= source_layer < n_layers:
        raise ValueError(f"source layer {source_layer} out of range [0, {n_layers - 1}]")
    if source_layer == 0:
        return "layers.0", False
    return f"layers.{source_layer - 1}", True
