"""Tests for the GPT-2-small res-jb loader/encoder + compressed source builder (synthetic)."""

from __future__ import annotations

import numpy as np

from layer_dynamics import saes_resjb
from layer_dynamics.saes_resjb import (
    ResJBSAE,
    build_compressed_sources_resjb,
    encode_dense,
    reconstruction_r2,
    resjb_subdir,
)


def _toy(d=8, m=20, seed=0) -> ResJBSAE:
    rng = np.random.default_rng(seed)
    return ResJBSAE(
        layer=0,
        W_enc=rng.standard_normal((d, m)).astype(np.float32),
        W_dec=rng.standard_normal((m, d)).astype(np.float32),
        b_enc=rng.standard_normal(m).astype(np.float32),
        b_dec=rng.standard_normal(d).astype(np.float32),
    )


def test_subdir_template():
    assert resjb_subdir(0) == "blocks.0.hook_resid_pre"
    assert resjb_subdir(11) == "blocks.11.hook_resid_pre"


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
    # identity decoder/encoder, zero biases, all-positive x → exact reconstruction.
    d = 4
    sae = ResJBSAE(layer=0, W_enc=np.eye(d, dtype=np.float32), W_dec=np.eye(d, dtype=np.float32),
                   b_enc=np.zeros(d, np.float32), b_dec=np.zeros(d, np.float32))
    x = np.abs(np.random.default_rng(0).standard_normal((25, d))).astype(np.float32)
    assert np.isclose(reconstruction_r2(sae, x), 1.0, atol=1e-5)


def test_reconstruction_r2_matches_formula():
    sae = _toy(d=5, m=12, seed=7)
    x = np.random.default_rng(8).standard_normal((200, sae.d_in)).astype(np.float32)
    acts = encode_dense(sae, x)
    rec = acts @ sae.W_dec + sae.b_dec
    expected = 1.0 - np.sum((x - rec) ** 2) / np.sum((x - x.mean(0)) ** 2)
    assert np.isclose(reconstruction_r2(sae, x), expected, atol=1e-5)


def test_build_compressed_sources_resjb_indexing_and_v0_has_sae(monkeypatch):
    # L=3 sources v_0..v_2 over L+1=4 states; every v_{ℓ'} <- blocks.{ℓ'} (v_0 INCLUDED).
    saes = {ell: _toy(d=5, m=40, seed=10 + ell) for ell in range(3)}
    monkeypatch.setattr(saes_resjb, "load_resjb", lambda layer, **kw: saes[layer])

    rng = np.random.default_rng(3)
    rs = rng.standard_normal((14, 4, 5)).astype(np.float32)  # [N, L+1, D], L=3
    cs = build_compressed_sources_resjb(rs)

    assert cs.u.shape[0] == 14 and cs.u.shape[1] == 3
    assert cs.feat_index.shape == (3, cs.u.shape[2])
    # unlike mlsae, v_0 has a real SAE → it should have alive features (not all zero)
    assert cs.alive_count[0] > 0
    assert np.any(cs.u[:, 0, :] != 0.0)
    # padded width = max alive across all three layers
    assert cs.u.shape[2] == max(int(c) for c in cs.alive_count)
    # every nonzero slot maps to a real latent; padding stays -1
    for ell in range(3):
        n_alive = int(cs.alive_count[ell])
        assert np.all(cs.feat_index[ell, :n_alive] >= 0)
        assert np.all(cs.feat_index[ell, n_alive:] == -1)

    # reconstruct v_1's compressed source back to dense and compare to a direct encode
    acts_direct = encode_dense(saes[1], rs[:, 1, :])
    dense_from_cs = np.zeros_like(acts_direct)
    for slot, g in enumerate(cs.feat_index[1]):
        if g >= 0:
            dense_from_cs[:, g] = cs.u[:, 1, slot]
    assert np.allclose(acts_direct, dense_from_cs)


def test_build_compressed_sources_resjb_prunes_dead_latents(monkeypatch):
    sae = _toy(d=4, m=600, seed=5)
    monkeypatch.setattr(saes_resjb, "load_resjb", lambda layer, **kw: sae)
    rs = np.random.default_rng(0).standard_normal((6, 3, 4)).astype(np.float32)
    cs = build_compressed_sources_resjb(rs)
    assert cs.n_latents_full == 600
    assert cs.u.shape[2] <= 600
