"""Tests for the 410m mlsae loader/encoder + compressed source builder (synthetic)."""

from __future__ import annotations

import numpy as np

from layer_dynamics import saes_mlsae
from layer_dynamics.saes_mlsae import (
    MLSAE,
    _standardize,
    build_compressed_sources,
    encode,
    encode_sparse,
    mlsae_repo,
    reconstruction_r2,
)


def _toy(d=8, m=20, k=3, seed=0, standardize=True):
    rng = np.random.default_rng(seed)
    W_dec = rng.standard_normal((d, m)).astype(np.float32)
    W_dec /= np.linalg.norm(W_dec, axis=0, keepdims=True)  # unit-norm columns
    return MLSAE(
        repo="toy",
        W_enc=rng.standard_normal((m, d)).astype(np.float32),
        W_dec=W_dec,
        pre_bias=rng.standard_normal(d).astype(np.float32),
        k=k,
        standardize=standardize,
    )


def test_standardize_zero_mean_unit_std():
    x = np.random.default_rng(0).standard_normal((10, 64)).astype(np.float32)
    xs, mu, std = _standardize(x)
    assert np.allclose(xs.mean(axis=-1), 0.0, atol=1e-5)
    assert np.allclose(xs.std(axis=-1, ddof=1), 1.0, atol=1e-3)
    assert mu.shape == (10, 1) and std.shape == (10, 1)


def test_encode_sparse_is_k_sparse_and_nonneg():
    sae = _toy()
    x = np.random.default_rng(1).standard_normal((50, sae.d_in)).astype(np.float32)
    idx, vals, mu, std = encode_sparse(sae, x)
    assert idx.shape == (50, sae.k) and vals.shape == (50, sae.k)
    assert np.all(vals >= 0.0)                       # ReLU after TopK
    assert (np.unique(idx[0]).size == sae.k)         # distinct latents per row


def test_encode_dense_matches_sparse():
    sae = _toy()
    x = np.random.default_rng(2).standard_normal((7, sae.d_in)).astype(np.float32)
    z = encode(sae, x)
    idx, vals, _, _ = encode_sparse(sae, x)
    assert z.shape == (7, sae.n_latents)
    rows = np.arange(7)[:, None]
    manual = np.zeros_like(z)
    manual[rows, idx] = vals
    assert np.allclose(z, manual)


def test_reconstruction_r2_perfect_when_identity_no_standardize():
    # identity decoder, no standardization, all-positive x, k=d → exact reconstruction.
    d = 4
    sae = MLSAE(repo="toy", W_enc=np.eye(d, dtype=np.float32),
                W_dec=np.eye(d, dtype=np.float32), pre_bias=np.zeros(d, np.float32),
                k=d, standardize=False)
    x = np.abs(np.random.default_rng(0).standard_normal((30, d))).astype(np.float32)
    assert np.isclose(reconstruction_r2(sae, x), 1.0, atol=1e-5)


def test_reconstruction_r2_matches_formula_with_standardize():
    sae = _toy(d=6, m=12, k=2, seed=7, standardize=True)
    x = np.random.default_rng(8).standard_normal((200, sae.d_in)).astype(np.float32)
    idx, vals, mu, std = encode_sparse(sae, x)
    cols = sae.W_dec[:, idx]
    rec = np.einsum("dnk,nk->nd", cols, vals) + sae.pre_bias
    rec = rec * std + mu
    expected = 1.0 - np.sum((x - rec) ** 2) / np.sum((x - x.mean(0)) ** 2)
    assert np.isclose(reconstruction_r2(sae, x), expected, atol=1e-5)


def test_mlsae_repo_template():
    assert mlsae_repo(5).endswith("layers-5")
    assert mlsae_repo(0).endswith("layers-0")


def test_build_compressed_sources(monkeypatch):
    # 3 source layers (v_0..v_2) over L+1=4 states; v_0 has no SAE (zero), v_1<-layers-0,
    # v_2<-layers-1. Patch load_mlsae to deterministic toys keyed by block index.
    saes = {0: _toy(d=5, m=30, k=2, seed=10), 1: _toy(d=5, m=30, k=4, seed=11)}
    monkeypatch.setattr(saes_mlsae, "load_mlsae", lambda layer, **kw: saes[layer])

    rng = np.random.default_rng(3)
    rs = rng.standard_normal((12, 4, 5)).astype(np.float32)  # [N, L+1, D], L=3
    cs = build_compressed_sources(rs)

    assert cs.u.shape[0] == 12 and cs.u.shape[1] == 3       # [N, L, M*]
    assert cs.feat_index.shape == (3, cs.u.shape[2])
    # v_0: no SAE → all zero, no alive features, feat_index all padding
    assert cs.alive_count[0] == 0
    assert np.all(cs.u[:, 0, :] == 0.0)
    assert np.all(cs.feat_index[0] == -1)
    # padded width = max alive over layers 1,2
    assert cs.u.shape[2] == max(cs.alive_count[1], cs.alive_count[2])
    # every nonzero slot maps to a real global latent; padding stays -1
    for ell in (1, 2):
        n_alive = int(cs.alive_count[ell])
        assert np.all(cs.feat_index[ell, :n_alive] >= 0)
        assert np.all(cs.feat_index[ell, n_alive:] == -1)

    # reconstruct one layer's source densely and compare to a direct encode
    sae1 = saes[0]  # v_1 <- layers-0
    idx, vals, _, _ = encode_sparse(sae1, rs[:, 1, :])
    dense_direct = np.zeros((12, sae1.n_latents), np.float32)
    rows = np.arange(12)[:, None]
    dense_direct[rows, idx] = vals
    # map compressed slots back to global ids
    dense_from_cs = np.zeros((12, sae1.n_latents), np.float32)
    fi = cs.feat_index[1]
    for slot, g in enumerate(fi):
        if g >= 0:
            dense_from_cs[:, g] = cs.u[:, 1, slot]
    assert np.allclose(dense_direct, dense_from_cs)


def test_build_compressed_sources_prunes_dead_latents(monkeypatch):
    # A wide SAE on few examples must leave most latents dead → M* ≪ M.
    sae = _toy(d=4, m=500, k=2, seed=5)
    monkeypatch.setattr(saes_mlsae, "load_mlsae", lambda layer, **kw: sae)
    rs = np.random.default_rng(0).standard_normal((8, 3, 4)).astype(np.float32)
    cs = build_compressed_sources(rs)
    assert cs.u.shape[2] < 50          # at most 8*2 distinct per layer ≪ 500
    assert cs.n_latents_full == 500
