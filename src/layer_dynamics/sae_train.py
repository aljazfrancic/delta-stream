"""Train matched per-layer residual SAEs on the cached 410m activations (Phase 4 E2).

The 410m primary lane sources its features from the public **mlsae TopK** suite
(:mod:`layer_dynamics.saes_mlsae`). The Gate-3 recorded limitation — decoder-direction
seed-stability ``0.45 < 0.7`` — was diagnosed across six models as a property of the
**TopK dictionary**, not the SSM (L1 suites are seed-stable on two families, TopK
unstable on two; see ``progress.md`` Fix 6 / S1). Every L1 datapoint, though, is
*off-Pythia* and every TopK datapoint *on-Pythia*, so the within-Pythia-410m L1-vs-TopK
control is missing. This module closes it: it trains **two SAEs that differ only in the
sparsity mechanism** (TopK vs L1) on the *same* cached query-position residuals, with the
same width / standardization / decoder normalisation, so the contrast is uncontaminated.

The trained-TopK variant additionally isolates "trained-here vs pretrained-mlsae" — its
weights map exactly onto the :class:`~layer_dynamics.saes_mlsae.MLSAE` dataclass (per-token
standardize, tied ``pre_encoder_bias``, ``W_enc[M, D]`` / ``W_dec[D, M]`` unit-norm
columns, ReLU-after-TopK), so the existing tested encoders build its sources.

**Indexing matches the mlsae lane exactly** (:func:`layer_dynamics.saes.source_hookpoint`):
source state ``v_{ℓ'}`` trains on ``rs[:, ℓ', :]``; ``v_0`` (embedding) has **no** SAE and
is zero-initialised, identical to the mlsae build, so the only thing that changes vs the
recorded Gate-3 run is the dictionary. The output is a drop-in
:class:`~layer_dynamics.saes_mlsae.CompressedSources` → ``sources_trained_{mode}.npz``
consumed by ``gate3_pareto`` / ``step4_sanity`` / ``phase4_checkpoint`` via ``--sources-npz``.

Training is cheap: the query-position distribution is one residual vector per fact (N≈1758),
so each layer's SAE trains full-batch on a ~7 MB tensor in seconds on the dev GPU. The cost
of E2 is the *downstream* re-fits (Gate-3 λ-sweep × seeds, the causal re-run), not this.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from layer_dynamics.saes import source_hookpoint
from layer_dynamics.saes_mlsae import CompressedSources


def _standardize_t(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-token standardize over the last dim → ``(x̃, mu, std)`` (mlsae convention)."""
    mu = x.mean(dim=-1, keepdim=True)
    xc = x - mu
    std = xc.std(dim=-1, keepdim=True, unbiased=True)
    return xc / (std + 1e-5), mu, std


class TrainableSAE(torch.nn.Module):
    """A residual SAE trainable in **topk** or **l1** mode (shared architecture).

    Architecture is the mlsae one for both modes: per-token standardize (optional), tied
    ``pre_bias[D]`` (subtracted pre-encode, added post-decode), ``W_enc[M, D]``,
    ``W_dec[D, M]`` with unit-norm columns. The *only* difference between the two trained
    variants is the activation: ``topk`` keeps the top-``k`` pre-activations then ReLUs
    them; ``l1`` ReLUs all and relies on an L1 penalty (added by the trainer) for sparsity.
    """

    def __init__(self, d_in: int, n_latents: int, *, mode: str = "topk", k: int = 32,
                 standardize: bool = True, seed: int = 0):
        super().__init__()
        if mode not in ("topk", "l1"):
            raise ValueError(f"mode must be 'topk' or 'l1', got {mode!r}")
        self.mode = mode
        self.k = int(k)
        self.standardize = bool(standardize)
        g = torch.Generator().manual_seed(seed)
        w = torch.randn(d_in, n_latents, generator=g)
        w /= w.norm(dim=0, keepdim=True).clamp_min(1e-8)
        self.W_dec = torch.nn.Parameter(w)                 # [D, M] unit-norm cols
        self.W_enc = torch.nn.Parameter(w.t().clone())     # [M, D] tied init
        self.pre_bias = torch.nn.Parameter(torch.zeros(d_in))

    @property
    def d_in(self) -> int:
        return int(self.W_dec.shape[0])

    @property
    def n_latents(self) -> int:
        return int(self.W_dec.shape[1])

    def _preact(self, x: torch.Tensor):
        xs, mu, std = (_standardize_t(x) if self.standardize
                       else (x, torch.zeros_like(x[..., :1]), torch.ones_like(x[..., :1])))
        pre = (xs - self.pre_bias) @ self.W_enc.t()        # [N, M]
        return pre, xs, mu, std

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                               torch.Tensor]:
        """Encode ``x[N, D]`` → ``(z[N, M], xs, mu, std)`` (standardized target kept)."""
        pre, xs, mu, std = self._preact(x)
        if self.mode == "topk":
            topv, topi = pre.topk(self.k, dim=-1)
            z = torch.zeros_like(pre).scatter(-1, topi, torch.relu(topv))
        else:
            z = torch.relu(pre)
        return z, xs, mu, std

    def decode_std(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latents to **standardized** residual space (``z @ W_decᵀ + pre_bias``)."""
        return z @ self.W_dec.t() + self.pre_bias

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z, xs, mu, std = self.encode(x)
        rec_std = self.decode_std(z)
        rec = rec_std * std + mu if self.standardize else rec_std
        return rec, z

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        """Renormalize ``W_dec`` columns to unit norm (post-step; standard SAE practice)."""
        self.W_dec.div_(self.W_dec.norm(dim=0, keepdim=True).clamp_min(1e-8))

    @torch.no_grad()
    def encode_acts(self, x: torch.Tensor) -> torch.Tensor:
        """Latent activations ``z[N, M]`` only (for source building / alive pruning)."""
        return self.encode(x)[0]


@dataclass
class TrainResult:
    """A trained SAE plus the curve numbers used for the loader sanity / log."""

    sae: TrainableSAE
    recon_r2: float          # held-in reconstruction R² in original (un-standardized) space
    l0: float                # mean active latents per token
    final_loss: float


def _r2_original(sae: TrainableSAE, x: torch.Tensor) -> float:
    rec, _ = sae(x)
    ss_res = float(((x - rec) ** 2).sum())
    ss_tot = float(((x - x.mean(dim=0)) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def train_sae(
    x: np.ndarray,
    *,
    mode: str = "topk",
    n_latents: int = 8192,
    k: int = 32,
    l1_coeff: float = 1e-2,
    standardize: bool = True,
    epochs: int = 2000,
    lr: float = 1e-3,
    seed: int = 0,
    device: str = "cpu",
    log=None,
) -> TrainResult:
    """Full-batch train one residual SAE on ``x[N, D]`` (recon MSE in standardized space).

    ``topk`` ignores ``l1_coeff``; ``l1`` ignores ``k``. Loss is the per-token MSE against
    the standardized target plus (l1 mode) ``l1_coeff · mean ‖z‖₁`` — the decoder columns
    are renormalised to unit norm after every step so the L1 penalty is on a fixed scale.
    """
    xt = torch.as_tensor(np.asarray(x, dtype=np.float32), device=device)
    sae = TrainableSAE(xt.shape[1], n_latents, mode=mode, k=k,
                       standardize=standardize, seed=seed).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    final_loss = float("nan")
    for ep in range(epochs):
        opt.zero_grad()
        z, xs, _, _ = sae.encode(xt)
        rec_std = sae.decode_std(z)
        mse = ((rec_std - xs) ** 2).sum(dim=-1).mean()
        loss = mse + (l1_coeff * z.abs().sum(dim=-1).mean() if mode == "l1" else 0.0)
        loss.backward()
        opt.step()
        sae.normalize_decoder()
        final_loss = float(loss)
        if log and (ep % max(1, epochs // 5) == 0 or ep == epochs - 1):
            with torch.no_grad():
                l0 = float((z != 0).float().sum(dim=-1).mean())
            log(f"    ep {ep:5d}  loss={final_loss:.4e}  mse={float(mse):.4e}  L0={l0:.1f}")
    with torch.no_grad():
        z = sae.encode_acts(xt)
        l0 = float((z != 0).float().sum(dim=-1).mean())
        r2 = _r2_original(sae, xt)
    return TrainResult(sae=sae, recon_r2=r2, l0=l0, final_loss=final_loss)


def build_compressed_sources_trained(
    rs: np.ndarray,
    *,
    mode: str = "topk",
    n_latents: int = 8192,
    k: int = 32,
    l1_coeff: float = 1e-2,
    standardize: bool = True,
    epochs: int = 2000,
    lr: float = 1e-3,
    seed: int = 0,
    device: str = "cpu",
    log=None,
) -> tuple[CompressedSources, list[dict]]:
    """Train one SAE per source layer and return drop-in Branch-B sources + state dicts.

    Mirrors :func:`layer_dynamics.saes_mlsae.build_compressed_sources`: source ``v_{ℓ'}`` is
    encoded by an SAE trained on ``rs[:, ℓ', :]``; ``v_0`` has no SAE (zero-init, identical
    to the mlsae build); each layer is pruned to its alive latents and padded to
    ``M* = max_ℓ |alive_ℓ|``. The returned ``state_dicts`` (one per source layer, ``None``
    for ``v_0``) let the caller persist the trained dictionaries for reuse/interpretation.
    """
    n, n_states, _ = rs.shape
    n_layers = n_states - 1
    acts_alive: list[np.ndarray | None] = [None] * n_layers
    alive_lists: list[np.ndarray] = []
    state_dicts: list[dict | None] = [None] * n_layers
    recon_r2 = np.full(n_layers, np.nan, dtype=np.float64)
    n_latents_full = n_latents
    for ellp in range(n_layers):
        _, matched = source_hookpoint(ellp, n_layers)
        if not matched:                       # v_0: no SAE → stays zero
            alive_lists.append(np.empty(0, dtype=np.int64))
            continue
        x = rs[:, ellp, :].astype(np.float32)
        res = train_sae(x, mode=mode, n_latents=n_latents, k=k, l1_coeff=l1_coeff,
                        standardize=standardize, epochs=epochs, lr=lr, seed=seed,
                        device=device, log=None)
        with torch.no_grad():
            acts = res.sae.encode_acts(
                torch.as_tensor(x, device=device)).cpu().numpy().astype(np.float32)
        alive = np.where(np.any(acts != 0.0, axis=0))[0]
        alive_lists.append(alive)
        acts_alive[ellp] = acts[:, alive]
        state_dicts[ellp] = {kk: v.detach().cpu().numpy() for kk, v in
                             res.sae.state_dict().items()}
        recon_r2[ellp] = res.recon_r2
        if log:
            log(f"  v_{ellp:<2d} <- SAE[{mode}] trained  alive={len(alive):5d}/{n_latents}"
                f"  L0={res.l0:.1f}  recon_R²={res.recon_r2:+.3f}")

    m_star = max((len(a) for a in alive_lists), default=0)
    if m_star == 0:
        raise ValueError("no trained SAE produced any alive latents")
    u = np.zeros((n, n_layers, m_star), dtype=np.float32)
    feat_index = np.full((n_layers, m_star), -1, dtype=np.int64)
    alive_count = np.zeros(n_layers, dtype=np.int64)
    for ellp in range(n_layers):
        alive = alive_lists[ellp]
        alive_count[ellp] = len(alive)
        if acts_alive[ellp] is None:
            continue
        feat_index[ellp, : len(alive)] = alive
        u[:, ellp, : len(alive)] = acts_alive[ellp]
    cs = CompressedSources(u=u, feat_index=feat_index, alive_count=alive_count,
                           n_latents_full=n_latents_full, recon_r2=recon_r2)
    return cs, state_dicts
