"""Tests for the Qwen2-0.5B chanind loader/encoder + compressed source builder (synthetic)."""

from __future__ import annotations

import numpy as np

from layer_dynamics import saes_qwen
from layer_dynamics.saes_qwen import (
    QwenSAE,
    build_compressed_sources_qwen,
    encode_dense,
    qwen_subdir,
    reconstruction_r2,
)


def _toy(d=8, m=20, seed=0) -> QwenSAE:
    rng = np.random.default_rng(seed)
    return QwenSAE(
        layer=0,
        W_enc=rng.standard_normal((d, m)).astype(np.float32),
        W_dec=rng.standard_normal((m, d)).astype(np.float32),
        b_enc=rng.standard_normal(m).astype(np.float32),
        b_dec=rng.standard_normal(d).astype(np.float32),
    )


def test_subdir_template():
    assert qwen_subdir(0) == "blocks.0.hook_resid_post"
    assert qwen_subdir(23) == "blocks.23.hook_resid_post"


def test_encode_dense_relu_nonneg_and_shape():
    sae = _toy()
    x = np.random.default_rng(1).standard_normal((30, sae.d_in)).astype(np.float32)
    acts = encode_dense(sae, x)
    assert acts.shape == (30, sae.n_latents)
    assert np.all(acts >= 0.0)  # ReLU


def test_encode_dense_matches_formula():
    sae = _toy(d=6, m=15, seed=4)
    x = np.random.default_rng(5).standard_normal((11, sae.d_in)).astype(np.float32)
    expected = np.maximum((x - sae.b_dec) @ sae.W_enc + sae.b_enc, 0.0)
    assert np.allclose(encode_dense(sae, x), expected)


def test_reconstruction_r2_perfect_when_identity():
    # identity decoder/encoder, zero biases, all-positive x → exact reconstruction (no scaling).
    d = 4
    sae = QwenSAE(layer=0, W_enc=np.eye(d, dtype=np.float32), W_dec=np.eye(d, dtype=np.float32),
                  b_enc=np.zeros(d, np.float32), b_dec=np.zeros(d, np.float32))
    x = np.abs(np.random.default_rng(0).standard_normal((25, d))).astype(np.float32) + 0.1
    assert np.isclose(reconstruction_r2(sae, x), 1.0, atol=1e-4)


def test_build_compressed_sources_qwen_v0_zero_init(monkeypatch):
    # L=3 sources v_0..v_2; v_0 has NO SAE (zero-init), v_{ℓ'≥1} <- blocks.{ℓ'-1}.resid_post
    saes = {ell: _toy(d=5, m=40, seed=10 + ell) for ell in range(2)}  # blocks 0,1 for v_1,v_2
    monkeypatch.setattr(saes_qwen, "load_qwen", lambda layer, **kw: saes[layer])

    rng = np.random.default_rng(3)
    rs = rng.standard_normal((14, 4, 5)).astype(np.float32)  # [N, L+1, D], L=3
    cs = build_compressed_sources_qwen(rs)

    assert cs.u.shape[0] == 14 and cs.u.shape[1] == 3
    assert cs.feat_index.shape == (3, cs.u.shape[2])
    # v_0 has no SAE → zero-initialised (no alive features)
    assert cs.alive_count[0] == 0
    assert np.all(cs.u[:, 0, :] == 0.0)
    assert np.isnan(cs.recon_r2[0])
    # v_1, v_2 have real SAEs → alive features
    assert cs.alive_count[1] > 0 and cs.alive_count[2] > 0
    for ell in range(3):
        n_alive = int(cs.alive_count[ell])
        assert np.all(cs.feat_index[ell, :n_alive] >= 0)
        assert np.all(cs.feat_index[ell, n_alive:] == -1)

    # reconstruct v_1's compressed source back to dense and compare to a direct encode
    acts_direct = encode_dense(saes[0], rs[:, 1, :])  # v_1 <- blocks.0
    dense_from_cs = np.zeros_like(acts_direct)
    for slot, g in enumerate(cs.feat_index[1]):
        if g >= 0:
            dense_from_cs[:, g] = cs.u[:, 1, slot]
    assert np.allclose(acts_direct, dense_from_cs)


def test_build_compressed_sources_qwen_prunes_dead_latents(monkeypatch):
    sae = _toy(d=4, m=600, seed=5)
    monkeypatch.setattr(saes_qwen, "load_qwen", lambda layer, **kw: sae)
    rs = np.random.default_rng(0).standard_normal((6, 3, 4)).astype(np.float32)
    cs = build_compressed_sources_qwen(rs)
    assert cs.n_latents_full == 600
    assert cs.u.shape[2] <= 600
