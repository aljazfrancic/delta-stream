"""Per-layer residual SAE sources for **Qwen2-0.5B** (third-family Phase 3 lane).

The cross-family generality test so far covers Pythia (mlsae TopK) and GPT-2-small
(res-jb L1). Qwen2-0.5B (``D = 896``, ``L = 24``) adds a genuinely different family —
RoPE / GQA / SwiGLU architecture, a different tokenizer, and different training data —
*and* matches the 410m primary's depth (24 layers), so the SSM-vs-baseline gap is
comparable at equal depth. Public per-layer **residual** SAEs exist: chanind's
``chanind/sae-qwen2-0.5b-res`` (one classic L1/ReLU SAE per block's **resid_post**
hook, ``d_in = 896``, ``d_sae = 7168`` = ×8 expansion).

It also disentangles the Fix-6 seed-stability confound: res-jb (L1) was seed-stable
(0.95) and mlsae (TopK) was not (0.38), but L1-vs-TopK was entangled with
GPT-2-vs-Pythia. Qwen's SAE is **L1** on a *third* family — if it is seed-stable, the
instability tracks the TopK dictionary type rather than deep non-GPT-2 geometry.

**Convention (SAELens "standard" architecture; verified against real activations).**
Per token, with ``apply_b_dec_to_input = True`` and a ReLU activation:

1. ``sae_in = x − b_dec``                       (subtract decoder bias pre-encode)
2. ``acts = relu(sae_in · W_enc + b_enc)``      (no TopK — sparsity is from L1 training)
3. ``x̂ = acts · W_dec + b_dec``

``W_enc`` is ``[D, M]``, ``W_dec`` is ``[M, D]``, ``b_enc`` is ``[M]``, ``b_dec`` is
``[D]`` (SAELens layout, **identical** to :mod:`layer_dynamics.saes_resjb`). The cfg
records ``normalize_activations = "expected_average_only_in"``, but chanind **folded the
scaling factor into the saved weights** (``fold_activation_norm_scaling_factor``), so the
SAE reconstructs directly in the original activation space with **no runtime scaling** —
verified on real Qwen ``blocks.0.hook_resid_post`` activations (raw encode→decode gives
R² ≈ 0.91 at L0 ≈ 7e-4; applying any input scaling collapses it to R² ≪ 0).

**Layer coverage.** The public chanind suite covers only ``blocks.0..11`` (the lower
half of Qwen2-0.5B's 24 layers). :func:`build_compressed_sources_qwen` therefore builds
sources for as many layers as ``rs`` provides — pass the **12-layer-prefix** residual
stream (``rs[:, :13]``) for the Phase-3 decomposition; H1/H2 diagnostics use the full
24-layer stream separately (they need no SAE). A depth-12 prefix is a clean equal-depth
comparison point against GPT-2-small (res-jb L1) and Pythia-160m (mlsae TopK).

**Hookpoint ↔ residual-state indexing (same as the mlsae suite, *not* res-jb).**
``resid_post`` of block ``N`` is our ``v_{N+1}``. So source state ``v_{ℓ'}`` for
``ℓ' ≥ 1`` is encoded by ``blocks.{ℓ'−1}.hook_resid_post``; ``v_0`` (embedding) has
**no** resid_post SAE and is zero-initialised (a free LASSO variable downstream);
``blocks.{L-1}.hook_resid_post`` (``v_L``) is the spare (never a source).

**Alive compression.** As in the other lanes, each layer is pruned to its *alive*
latents (active on ≥1 example) and padded to ``M* = max_ℓ |alive_ℓ|`` — Branch B
permits a per-layer width ``M_ℓ``. ``d_sae = 7168`` encodes to a small dense ``[N, M]``
acts matrix, so the alive set and per-example values come from its nonzero columns.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from layer_dynamics.saes_mlsae import CompressedSources  # shared output structure

REPO = "chanind/sae-qwen2-0.5b-res"
SUBDIR_TEMPLATE = "blocks.{layer}.hook_resid_post"


@dataclass
class QwenSAE:
    """A loaded chanind Qwen2-0.5B residual SAE (classic L1, SAELens layout).

    ``layer`` is the **block index** whose ``hook_resid_post`` (= ``v_{layer+1}``) this
    SAE encodes. ``W_enc[D, M]``, ``W_dec[M, D]``, ``b_enc[M]``, ``b_dec[D]``.
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


def qwen_subdir(layer: int) -> str:
    """chanind subdir for the SAE on ``blocks.{layer}.hook_resid_post`` (= ``v_{layer+1}``)."""
    return SUBDIR_TEMPLATE.format(layer=layer)


def load_qwen(layer: int, *, repo: str = REPO, cache_dir: str | Path | None = None) -> QwenSAE:
    """Download (cached) and load the chanind SAE for ``blocks.{layer}.hook_resid_post``."""
    from huggingface_hub import hf_hub_download
    from safetensors.numpy import load_file

    sub = qwen_subdir(layer)
    kw = {"cache_dir": str(cache_dir)} if cache_dir is not None else {}
    t = load_file(hf_hub_download(repo, f"{sub}/sae_weights.safetensors", **kw))
    return QwenSAE(
        layer=layer,
        W_enc=t["W_enc"].astype(np.float32),  # [D, M]
        W_dec=t["W_dec"].astype(np.float32),  # [M, D]
        b_enc=t["b_enc"].astype(np.float32),  # [M]
        b_dec=t["b_dec"].astype(np.float32),  # [D]
    )


def encode_dense(sae: QwenSAE, x: np.ndarray) -> np.ndarray:
    """ReLU encode ``x[N, D]`` → acts ``[N, M]`` (apply_b_dec_to_input, no TopK, no scaling)."""
    if x.ndim != 2:
        raise ValueError(f"encode_dense expects [N, D], got {x.shape}")
    sae_in = x - sae.b_dec
    return np.maximum(sae_in @ sae.W_enc + sae.b_enc, 0.0)


def reconstruct(sae: QwenSAE, x: np.ndarray) -> np.ndarray:
    """Encode→decode ``x[N, D]`` → ``x̂[N, D]`` (original activation space)."""
    return encode_dense(sae, x) @ sae.W_dec + sae.b_dec


def reconstruction_r2(sae: QwenSAE, x: np.ndarray) -> float:
    """Held-in reconstruction R² of ``x[N, D]`` (loader sanity)."""
    rec = reconstruct(sae, x)
    ss_res = float(np.sum((x - rec) ** 2))
    ss_tot = float(np.sum((x - x.mean(axis=0)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def build_compressed_sources_qwen(
    rs: np.ndarray,
    *,
    repo: str = REPO,
    cache_dir: str | Path | None = None,
    log=None,
) -> CompressedSources:
    """SAE-initialised, per-layer-alive-pruned Branch B inputs from ``rs[N, L+1, D]``.

    ``v_0`` (embedding) has no resid_post SAE → stays zero. ``v_{ℓ'≥1}`` is encoded by
    ``blocks.{ℓ'−1}.hook_resid_post``. Returns the same
    :class:`~layer_dynamics.saes_mlsae.CompressedSources` structure as the other lanes.
    """
    n, n_states, _ = rs.shape
    n_layers = n_states - 1
    acts_alive: list[np.ndarray] = []   # [N, |alive_ℓ|] per layer
    alive_lists: list[np.ndarray] = []
    recon_r2 = np.full(n_layers, np.nan, dtype=np.float64)
    n_latents_full = 0
    for ellp in range(n_layers):
        if ellp == 0:                       # v_0: no resid_post SAE → stays zero
            alive_lists.append(np.empty(0, dtype=np.int64))
            acts_alive.append(np.empty((n, 0), dtype=np.float32))
            continue
        sae = load_qwen(ellp - 1, repo=repo, cache_dir=cache_dir)
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
            log(f"  v_{ellp:<2d} <- blocks.{ellp - 1}.hook_resid_post  alive={len(alive):5d}"
                f"/{sae.n_latents}  recon_R²={recon_r2[ellp]:+.3f}")

    m_star = max((len(a) for a in alive_lists), default=0)
    if m_star == 0:
        raise ValueError("no Qwen SAE produced any alive latents")
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
