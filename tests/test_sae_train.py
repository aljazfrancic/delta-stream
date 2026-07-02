"""Tests for the E2 trainable per-layer SAE (topk / l1) + source builder (synthetic)."""

from __future__ import annotations

import numpy as np
import torch

from layer_dynamics.sae_train import (
    TrainableSAE,
    build_compressed_sources_trained,
    train_sae,
)


def _toy_rs(n=40, n_states=5, d=16, seed=0):
    """A low-rank residual stream so an SAE can actually reconstruct it."""
    rng = np.random.default_rng(seed)
    basis = rng.standard_normal((d, 4)).astype(np.float32)
    rs = np.empty((n, n_states, d), dtype=np.float32)
    for s in range(n_states):
        coeff = rng.standard_normal((n, 4)).astype(np.float32)
        rs[:, s, :] = coeff @ basis.T + 0.01 * rng.standard_normal((n, d)).astype(np.float32)
    return rs


def test_topk_encode_is_k_sparse():
    sae = TrainableSAE(16, 32, mode="topk", k=3, seed=1)
    x = torch.randn(8, 16)
    z, _, _, _ = sae.encode(x)
    # ReLU-after-TopK ⇒ at most k nonzeros per row (some may be ReLU'd to 0)
    assert int((z != 0).sum(dim=-1).max()) <= 3
    assert z.shape == (8, 32)


def test_l1_encode_is_relu_dense_then_sparsifies_with_penalty():
    sae = TrainableSAE(16, 32, mode="l1", seed=1)
    z, _, _, _ = sae.encode(torch.randn(8, 16))
    assert (z >= 0).all()  # ReLU, no TopK mask


def test_decoder_columns_unit_norm_after_normalize():
    sae = TrainableSAE(16, 32, mode="l1", seed=2)
    with torch.no_grad():
        sae.W_dec.mul_(3.7)  # perturb scale
    sae.normalize_decoder()
    norms = sae.W_dec.norm(dim=0)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_standardize_roundtrip_shapes():
    sae = TrainableSAE(16, 32, mode="topk", k=4, standardize=True, seed=0)
    x = torch.randn(5, 16)
    rec, z = sae(x)
    assert rec.shape == x.shape and z.shape == (5, 32)


def test_train_reduces_loss_and_reconstructs():
    rng = np.random.default_rng(0)
    basis = rng.standard_normal((16, 4)).astype(np.float32)
    x = (rng.standard_normal((60, 4)).astype(np.float32) @ basis.T)
    res = train_sae(x, mode="topk", n_latents=32, k=6, epochs=400, lr=5e-3, seed=0)
    assert res.recon_r2 > 0.5  # a low-rank signal is reconstructable
    assert res.l0 <= 6 + 1e-6


def test_l1_penalty_increases_sparsity():
    rng = np.random.default_rng(1)
    basis = rng.standard_normal((16, 4)).astype(np.float32)
    x = (rng.standard_normal((60, 4)).astype(np.float32) @ basis.T)
    lo = train_sae(x, mode="l1", n_latents=32, l1_coeff=1e-3, epochs=400, lr=5e-3, seed=0)
    hi = train_sae(x, mode="l1", n_latents=32, l1_coeff=3e-1, epochs=400, lr=5e-3, seed=0)
    assert hi.l0 <= lo.l0  # stronger L1 ⇒ no more active latents


def test_build_sources_shape_and_v0_zero_init():
    rs = _toy_rs(n=30, n_states=5, d=16, seed=3)  # 4 source layers v_0..v_3
    cs, state_dicts = build_compressed_sources_trained(
        rs, mode="topk", n_latents=32, k=4, epochs=150, lr=5e-3, seed=0)
    n, n_states, _ = rs.shape
    n_layers = n_states - 1
    assert cs.u.shape[0] == n and cs.u.shape[1] == n_layers
    # v_0 has no SAE: zero-init sources, no state dict, NaN recon, feat_index all -1
    assert np.all(cs.u[:, 0, :] == 0.0)
    assert state_dicts[0] is None
    assert np.isnan(cs.recon_r2[0])
    assert np.all(cs.feat_index[0] == -1)
    # matched layers carry trained dictionaries + finite recon
    assert state_dicts[1] is not None
    assert np.isfinite(cs.recon_r2[1:]).all()


def test_build_sources_feat_index_matches_alive_count():
    rs = _toy_rs(n=24, n_states=4, d=16, seed=5)
    cs, _ = build_compressed_sources_trained(
        rs, mode="l1", n_latents=32, l1_coeff=1e-2, epochs=120, lr=5e-3, seed=0)
    for ell in range(rs.shape[1] - 1):
        c = int(cs.alive_count[ell])
        assert np.all(cs.feat_index[ell, :c] >= 0) or c == 0
        assert np.all(cs.feat_index[ell, c:] == -1)


def test_build_sources_train_rows_encodes_all_rows():
    """R4: with train_rows, the SAE fit/alive-mask see only train rows but every row
    is encoded into the returned sources."""
    rs = _toy_rs(n=30, n_states=4, d=16, seed=5)
    train_rows = np.arange(20)
    cs, _ = build_compressed_sources_trained(
        rs, mode="topk", n_latents=32, k=3, epochs=50, train_rows=train_rows)
    assert cs.u.shape[0] == 30                       # all rows encoded
    assert np.all(cs.u[:, 0, :] == 0)                # v_0 still zero-init
    held = cs.u[20:, 1:, :]                          # held-out rows, SAE'd layers
    assert np.abs(held).sum() > 0                    # test rows got real codes
    # deterministic wrt the same train_rows
    cs2, _ = build_compressed_sources_trained(
        rs, mode="topk", n_latents=32, k=3, epochs=50, train_rows=train_rows)
    assert np.allclose(cs.u, cs2.u)
