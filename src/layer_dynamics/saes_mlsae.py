"""Per-layer residual SAE sources for the **410m primary lane** (Phase 3 Step 2).

The 70m debug lane uses the EleutherAI ``sae-pythia-70m-deduped-32k`` suite
(:mod:`layer_dynamics.saes`). The 410m-deduped primary model has **no** EleutherAI
residual SAE (``sae-pythia-410m-65k`` is non-deduped + MLP-only), but a complete
per-layer **residual** suite exists from the MLSAE work (Lawson et al. 2024, the
paper already cited in the proposal): ``tim-lawson/sae-pythia-410m-deduped-x64-k32-layers-{N}``
— one HF repo per block ``N = 0 … 23``, ``M = 65536`` (×64 expansion), TopK ``k = 32``.

**Convention (differs from the EleutherAI suite — verified against the mlsae source).**
The TopK SAE forward is, per token:

1. **standardize** ``x`` over the ``D`` dimension (LayerNorm-style, no affine):
   ``mu = mean(x); std = std(x, unbiased); x̃ = (x − mu) / (std + 1e-5)`` and keep
   ``(mu, std)`` to un-standardize the reconstruction;
2. encode ``latents = (x̃ − pre_encoder_bias) · W_encᵀ`` (**no** encoder bias);
3. select the top-``k`` latents **by raw value**, then **ReLU** the kept values
   (ReLU is applied *after* the TopK selection, not before);
4. decode ``x̂̃ = z · W_decᵀ + pre_encoder_bias`` then un-standardize
   ``x̂ = x̂̃ · std + mu``.

``W_enc`` is ``[M, D]``, ``W_dec`` is ``[D, M]`` (unit-norm columns), and
``pre_encoder_bias`` is a single ``[D]`` tied bias (subtracted pre-encode, added
post-decode). Source code: https://github.com/tim-lawson/mlsae.

**Hookpoint ↔ residual-state indexing (same as the 70m suite).** mlsae appends one
hidden state *after each block* and excludes the embedding, so ``layers-N`` is the
output of block ``N`` = our ``v_{N+1}``. The SAE matching source state ``v_{ℓ'}`` is
therefore ``layers-{ℓ'−1}`` for ``ℓ' ≥ 1``; ``v_0`` (embedding) has **no** SAE and is
zero-initialised; ``layers-{L−1}`` (``v_L``) is the spare (never a source). This is the
identical mapping :func:`layer_dynamics.saes.source_hookpoint` implements, so it is
reused directly.

**Memory / per-layer width.** A dense ``u[N, L, M]`` at ``M = 65536`` is ~10 GB — it
does not fit. But the query-position factual-recall distribution is narrow: only a few
hundred of the 65536 latents ever fire per layer across the dataset. Branch B permits a
**per-layer dictionary width** ``M_ℓ`` (its operators are free per layer), so
:func:`build_compressed_sources` prunes each layer to its *alive* latents (those active
on ≥1 example) and pads to ``M* = max_ℓ |alive_ℓ|`` for the uniform-width SSM. Pruning
provably-dead columns is exact for reconstruction (a never-active latent contributes
nothing); the only restriction is that the held-out LASSO re-infers within the alive
column space, which is the standard dead-latent removal, recorded in the memo.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_TEMPLATE = "tim-lawson/sae-pythia-410m-deduped-x64-k32-layers-{layer}"


@dataclass
class MLSAE:
    """A loaded mlsae TopK SAE (410m residual suite).

    ``W_enc[M, D]``, ``W_dec[D, M]`` (unit-norm columns), ``pre_bias[D]`` (tied bias),
    TopK width ``k``, and the per-token ``standardize`` flag.
    """

    repo: str
    W_enc: np.ndarray
    W_dec: np.ndarray
    pre_bias: np.ndarray
    k: int
    standardize: bool = True

    @property
    def d_in(self) -> int:
        return int(self.pre_bias.shape[0])

    @property
    def n_latents(self) -> int:
        return int(self.W_enc.shape[0])


def mlsae_repo(layer: int, template: str = REPO_TEMPLATE) -> str:
    """HF repo id for the SAE trained on block ``layer`` output (= ``v_{layer+1}``)."""
    return template.format(layer=layer)


def load_mlsae(layer: int, *, template: str = REPO_TEMPLATE,
               cache_dir: str | Path | None = None) -> MLSAE:
    """Download (cached) and load the mlsae SAE for block ``layer``."""
    from huggingface_hub import hf_hub_download
    from safetensors.numpy import load_file

    repo = mlsae_repo(layer, template)
    kw = {"cache_dir": str(cache_dir)} if cache_dir is not None else {}
    cfg = json.loads(Path(hf_hub_download(repo, "config.json", **kw)).read_text())
    t = load_file(hf_hub_download(repo, "model.safetensors", **kw))
    return MLSAE(
        repo=repo,
        W_enc=t["encoder.weight"].astype(np.float32),    # [M, D]
        W_dec=t["decoder.weight"].astype(np.float32),    # [D, M]
        pre_bias=t["pre_encoder_bias"].astype(np.float32),
        k=int(cfg["k"]),
        standardize=bool(cfg.get("standardize", True)),
    )


def _standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-token standardize over the last dim → ``(x̃, mu, std)`` (mlsae convention)."""
    mu = x.mean(axis=-1, keepdims=True)
    xc = x - mu
    std = xc.std(axis=-1, keepdims=True, ddof=1)  # torch .std() default is unbiased
    return xc / (std + 1e-5), mu, std


def encode_sparse(
    sae: MLSAE, x: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """TopK encode ``x[N, D]`` → ``(idx[N, k], vals[N, k], mu[N, 1], std[N, 1])``.

    ``idx``/``vals`` are the top-``k``-by-raw-value latents with ReLU applied to the kept
    values (so some may be 0). ``mu``/``std`` are the standardization stats (for decode).
    Sparse form so the dense ``[N, M]`` pre-activation is the only large transient.
    """
    if x.ndim != 2:
        raise ValueError(f"encode_sparse expects [N, D], got {x.shape}")
    xs, mu, std = (_standardize(x) if sae.standardize else (x, np.zeros(1), np.ones(1)))
    pre = (xs - sae.pre_bias) @ sae.W_enc.T          # [N, M]
    idx = np.argpartition(pre, -sae.k, axis=1)[:, -sae.k:]
    rows = np.arange(pre.shape[0])[:, None]
    vals = np.maximum(pre[rows, idx], 0.0)           # ReLU after TopK selection
    return idx.astype(np.int64), vals.astype(np.float32), mu, std


def encode(sae: MLSAE, x: np.ndarray) -> np.ndarray:
    """Dense TopK latents ``z[N, M]`` (convenience; materialises ``M`` — small N only)."""
    idx, vals, _, _ = encode_sparse(sae, x)
    z = np.zeros((x.shape[0], sae.n_latents), dtype=np.float32)
    rows = np.arange(x.shape[0])[:, None]
    z[rows, idx] = vals
    return z


def reconstruct(sae: MLSAE, x: np.ndarray) -> np.ndarray:
    """TopK encode then decode ``x[N, D]`` → ``x̂[N, D]`` (un-standardized).

    The SAE's own reconstruction of the residual state. Used by the loader sanity check
    (:func:`reconstruction_r2`) and by the Phase 4 Baseline B implied-Δ comparison, which
    differences two reconstructions ``v̂_{ℓ+1} − v̂_ℓ`` to predict ``Δ_ℓ``.
    """
    idx, vals, mu, std = encode_sparse(sae, x)
    # decode: gather the k decoder columns per row → [N, D], add tied bias, un-standardize
    cols = sae.W_dec[:, idx]                          # [D, N, k]
    rec = np.einsum("dnk,nk->nd", cols, vals) + sae.pre_bias
    if sae.standardize:
        rec = rec * std + mu
    return rec


def reconstruction_r2(sae: MLSAE, x: np.ndarray) -> float:
    """Held-in reconstruction R² of ``x[N, D]`` (loader sanity; un-standardizes)."""
    rec = reconstruct(sae, x)
    ss_res = float(np.sum((x - rec) ** 2))
    ss_tot = float(np.sum((x - x.mean(axis=0)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


@dataclass
class CompressedSources:
    """Branch B inputs pruned to per-layer alive latents and padded to a uniform width.

    ``u[N, L, M*]`` are the SAE-initialised inputs (``M* = max_ℓ |alive_ℓ|``); slot ``j``
    at layer ``ℓ`` maps to **global** SAE latent ``feat_index[ℓ, j]`` (``-1`` for padding
    and for the SAE-less ``v_0`` layer). ``alive_count[ℓ]`` is the unpadded width. The
    per-layer maps differ because each layer uses a different SAE — slot ``j`` is only
    comparable to itself at the **same** layer (as in the 70m pipeline).
    """

    u: np.ndarray             # [N, L, M*]  float32
    feat_index: np.ndarray    # [L, M*]     int64  (global latent id per slot, -1 = pad)
    alive_count: np.ndarray   # [L]         int64
    n_latents_full: int       # M before pruning (65536)
    recon_r2: np.ndarray      # [L]  per-layer SAE reconstruction R² (NaN for v_0, no SAE)


def load_cached_u(path: str | Path) -> np.ndarray:
    """Load the cached compressed source tensor ``u[N, L, M*]`` (float32) from an npz.

    The 410m fit scripts pass ``--sources-npz`` pointing at the file written by
    ``scripts/build_410m_sources.py``; only ``u`` is needed to fit (``feat_index`` is
    for interpreting features at Step 4). Kept here so the load convention lives with
    the builder.
    """
    return np.load(path)["u"].astype(np.float32)


def build_compressed_sources(
    rs: np.ndarray,
    *,
    template: str = REPO_TEMPLATE,
    cache_dir: str | Path | None = None,
    log=None,
) -> CompressedSources:
    """SAE-initialised, per-layer-alive-pruned Branch B inputs from ``rs[N, L+1, D]``.

    For each source state ``v_{ℓ'}`` (``ℓ' = 0 … L-1``): ``v_0`` is left zero (no SAE);
    ``v_{ℓ'≥1}`` is encoded by SAE ``layers-{ℓ'−1}``, the set of latents active on any
    example is collected (the *alive* set), and those activations are written into a
    compact per-layer slot space. All layers are padded to ``M* = max_ℓ |alive_ℓ|``.
    """
    from layer_dynamics.saes import source_hookpoint  # same v_{ℓ'} ↔ layers-{ℓ'-1} map

    n, n_states, _ = rs.shape
    n_layers = n_states - 1
    sparse: list[tuple[np.ndarray, np.ndarray, np.ndarray] | None] = [None] * n_layers
    alive_lists: list[np.ndarray] = []
    recon_r2 = np.full(n_layers, np.nan, dtype=np.float64)
    n_latents_full = 0
    for ellp in range(n_layers):
        _, matched = source_hookpoint(ellp, n_layers)
        if not matched:                       # v_0: no SAE → stays zero
            alive_lists.append(np.empty(0, dtype=np.int64))
            continue
        sae = load_mlsae(ellp - 1, template=template, cache_dir=cache_dir)
        n_latents_full = sae.n_latents
        x = rs[:, ellp, :].astype(np.float32)
        idx, vals, mu, std = encode_sparse(sae, x)
        alive = np.unique(idx[vals > 0])
        alive_lists.append(alive)
        sparse[ellp] = (idx, vals, alive)
        # reconstruction R² (loader sanity, un-standardized) from the same encode
        rec = np.einsum("dnk,nk->nd", sae.W_dec[:, idx], vals) + sae.pre_bias
        rec = rec * std + mu if sae.standardize else rec
        ss_res = float(np.sum((x - rec) ** 2))
        ss_tot = float(np.sum((x - x.mean(axis=0)) ** 2))
        recon_r2[ellp] = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        if log:
            log(f"  v_{ellp:<2d} <- layers-{ellp - 1:<2d}  alive={len(alive):5d}/{sae.n_latents}"
                f"  recon_R²={recon_r2[ellp]:+.3f}")

    m_star = max((len(a) for a in alive_lists), default=0)
    if m_star == 0:
        raise ValueError("no matched SAE produced any alive latents")
    u = np.zeros((n, n_layers, m_star), dtype=np.float32)
    feat_index = np.full((n_layers, m_star), -1, dtype=np.int64)
    alive_count = np.zeros(n_layers, dtype=np.int64)
    for ellp in range(n_layers):
        alive = alive_lists[ellp]
        alive_count[ellp] = len(alive)
        if sparse[ellp] is None:
            continue
        feat_index[ellp, : len(alive)] = alive
        # map each *active* global latent id → its compact slot, then scatter the values.
        # Only vals>0 entries are alive; TopK can return ReLU'd-to-zero indices that fire
        # on no example (never in `alive`), so mask to active before the slot lookup.
        slot_of = {int(g): s for s, g in enumerate(alive)}
        idx, vals, _ = sparse[ellp]
        mask = vals > 0
        glob = idx[mask]
        rows = np.broadcast_to(np.arange(n)[:, None], idx.shape)[mask]
        local = np.fromiter((slot_of[int(g)] for g in glob), dtype=np.int64, count=glob.size)
        u[rows, ellp, local] = vals[mask]               # TopK ids unique per row → no clash
    return CompressedSources(u=u, feat_index=feat_index, alive_count=alive_count,
                             n_latents_full=n_latents_full, recon_r2=recon_r2)
