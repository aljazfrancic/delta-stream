"""Per-layer residual SAE sources for **GPT-2-small** (cross-family Phase 3 lane).

The Pythia lanes use TopK MLSAE checkpoints (:mod:`layer_dynamics.saes_mlsae`,
:mod:`layer_dynamics.saes`). The GPT-2-small cross-family replication uses Joseph
Bloom's **res-jb** suite (``jbloom/GPT2-Small-SAEs-Reformatted``): one classic
L1/ReLU SAE per block's **resid_pre** hook, ``d_in = 768``, ``d_sae = 24576``
(×32 expansion). These are *not* TopK and are *not* input-standardized.

**Convention (verified from the res-jb cfg.json + SAELens classic-SAE forward).**
Per token, with ``apply_b_dec_to_input = True`` and ``activation_fn = relu``:

1. ``sae_in = x − b_dec``                      (subtract the decoder bias pre-encode)
2. ``acts = relu(sae_in · W_enc + b_enc)``     (no TopK — sparsity is from L1 training)
3. ``x̂ = acts · W_dec + b_dec``

``W_enc`` is ``[D, M]``, ``W_dec`` is ``[M, D]``, ``b_enc`` is ``[M]``, ``b_dec`` is
``[D]`` (SAELens layout). res-jb has no input normalization and no ``scaling_factor``.

**Hookpoint ↔ residual-state indexing (differs from the mlsae suite).** ``resid_pre``
of block ``N`` is the residual stream *entering* block ``N`` = our ``v_N``. So source
state ``v_{ℓ'}`` is encoded by ``blocks.{ℓ'}.hook_resid_pre`` **directly** — and unlike
the mlsae (resid_post) suite, **``v_0`` has an SAE** (``blocks.0.hook_resid_pre``), so
*no* layer is zero-initialised. The sources are ``v_0 … v_{L-1}`` (``blocks.0 …
blocks.{L-1}``); ``v_L`` (final resid_post) has no resid_pre SAE and is never a source.

**Alive compression.** As in the mlsae lane, each layer is pruned to its *alive*
latents (active on ≥1 example) and padded to ``M* = max_ℓ |alive_ℓ|`` — Branch B
permits a per-layer width ``M_ℓ``. res-jb encodes to a dense ``[N, M]`` acts matrix
(``M = 24576`` → ~0.1 GB per layer at ``N ≈ 1000``, a fine transient), so the alive set
and per-example values come straight from the nonzero columns of ``acts``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from layer_dynamics.saes_mlsae import CompressedSources  # shared output structure

REPO = "jbloom/GPT2-Small-SAEs-Reformatted"
SUBDIR_TEMPLATE = "blocks.{layer}.hook_resid_pre"


@dataclass
class ResJBSAE:
    """A loaded res-jb classic L1 SAE (GPT-2-small resid_pre suite).

    ``W_enc[D, M]``, ``W_dec[M, D]``, ``b_enc[M]``, ``b_dec[D]`` (SAELens layout).
    """

    layer: int
    W_enc: np.ndarray
    W_dec: np.ndarray
    b_enc: np.ndarray
    b_dec: np.ndarray

    @property
    def d_in(self) -> int:
        return int(self.W_enc.shape[0])

    @property
    def n_latents(self) -> int:
        return int(self.W_enc.shape[1])


def resjb_subdir(layer: int) -> str:
    """res-jb subdirectory for the SAE on ``blocks.{layer}.hook_resid_pre`` (= ``v_layer``)."""
    return SUBDIR_TEMPLATE.format(layer=layer)


def load_resjb(layer: int, *, repo: str = REPO, cache_dir: str | Path | None = None) -> ResJBSAE:
    """Download (cached) and load the res-jb SAE for ``blocks.{layer}.hook_resid_pre``."""
    from huggingface_hub import hf_hub_download
    from safetensors.numpy import load_file

    sub = resjb_subdir(layer)
    kw = {"cache_dir": str(cache_dir)} if cache_dir is not None else {}
    t = load_file(hf_hub_download(repo, f"{sub}/sae_weights.safetensors", **kw))
    return ResJBSAE(
        layer=layer,
        W_enc=t["W_enc"].astype(np.float32),  # [D, M]
        W_dec=t["W_dec"].astype(np.float32),  # [M, D]
        b_enc=t["b_enc"].astype(np.float32),  # [M]
        b_dec=t["b_dec"].astype(np.float32),  # [D]
    )


def encode_dense(sae: ResJBSAE, x: np.ndarray) -> np.ndarray:
    """ReLU encode ``x[N, D]`` → acts ``[N, M]`` (apply_b_dec_to_input=True, no TopK)."""
    if x.ndim != 2:
        raise ValueError(f"encode_dense expects [N, D], got {x.shape}")
    sae_in = x - sae.b_dec
    return np.maximum(sae_in @ sae.W_enc + sae.b_enc, 0.0)


def reconstruction_r2(sae: ResJBSAE, x: np.ndarray) -> float:
    """Held-in reconstruction R² of ``x[N, D]`` (loader sanity)."""
    acts = encode_dense(sae, x)
    rec = acts @ sae.W_dec + sae.b_dec
    ss_res = float(np.sum((x - rec) ** 2))
    ss_tot = float(np.sum((x - x.mean(axis=0)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def build_compressed_sources_resjb(
    rs: np.ndarray,
    *,
    repo: str = REPO,
    cache_dir: str | Path | None = None,
    log=None,
) -> CompressedSources:
    """SAE-initialised, per-layer-alive-pruned Branch B inputs from ``rs[N, L+1, D]``.

    Each source ``v_{ℓ'}`` (``ℓ' = 0 … L-1``) is encoded by ``blocks.{ℓ'}.hook_resid_pre``
    (every source has an SAE — no zero-init layer). Returns the same
    :class:`~layer_dynamics.saes_mlsae.CompressedSources` structure as the mlsae lane.
    """
    n, n_states, _ = rs.shape
    n_layers = n_states - 1
    acts_alive: list[np.ndarray] = []   # [N, |alive_ℓ|] per layer
    alive_lists: list[np.ndarray] = []
    recon_r2 = np.full(n_layers, np.nan, dtype=np.float64)
    n_latents_full = 0
    for ellp in range(n_layers):
        sae = load_resjb(ellp, repo=repo, cache_dir=cache_dir)
        n_latents_full = sae.n_latents
        x = rs[:, ellp, :].astype(np.float32)
        acts = encode_dense(sae, x)                 # [N, M]
        alive = np.where(np.any(acts != 0.0, axis=0))[0]
        alive_lists.append(alive)
        acts_alive.append(acts[:, alive].astype(np.float32))
        rec = acts @ sae.W_dec + sae.b_dec
        ss_res = float(np.sum((x - rec) ** 2))
        ss_tot = float(np.sum((x - x.mean(axis=0)) ** 2))
        recon_r2[ellp] = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        if log:
            log(f"  v_{ellp:<2d} <- blocks.{ellp}.hook_resid_pre  alive={len(alive):5d}"
                f"/{sae.n_latents}  recon_R²={recon_r2[ellp]:+.3f}")

    m_star = max((len(a) for a in alive_lists), default=0)
    if m_star == 0:
        raise ValueError("no res-jb SAE produced any alive latents")
    u = np.zeros((n, n_layers, m_star), dtype=np.float32)
    feat_index = np.full((n_layers, m_star), -1, dtype=np.int64)
    alive_count = np.zeros(n_layers, dtype=np.int64)
    for ellp in range(n_layers):
        alive = alive_lists[ellp]
        alive_count[ellp] = len(alive)
        u[:, ellp, : len(alive)] = acts_alive[ellp]
        feat_index[ellp, : len(alive)] = alive
    return CompressedSources(u=u, feat_index=feat_index, alive_count=alive_count,
                             n_latents_full=n_latents_full, recon_r2=recon_r2)
