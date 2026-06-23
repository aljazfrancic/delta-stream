"""Tests for the Phase 3 per-layer SAE source loader/encoder (synthetic, no download)."""

from __future__ import annotations

import numpy as np
import pytest

from layer_dynamics.saes import (
    TopKSAE,
    decode,
    encode,
    reconstruction_r2,
    source_hookpoint,
)


def _toy_sae(d=8, m=20, k=3, seed=0):
    rng = np.random.default_rng(seed)
    return TopKSAE(
        hookpoint="layers.0",
        W_enc=rng.standard_normal((m, d)).astype(np.float32),
        b_enc=rng.standard_normal(m).astype(np.float32),
        W_dec=rng.standard_normal((m, d)).astype(np.float32),
        b_dec=rng.standard_normal(d).astype(np.float32),
        k=k,
    )


def test_encode_is_exactly_k_sparse():
    sae = _toy_sae()
    x = np.random.default_rng(1).standard_normal((50, sae.d_in)).astype(np.float32)
    z = encode(sae, x)
    assert z.shape == (50, sae.n_latents)
    assert np.all((z != 0).sum(axis=1) <= sae.k)        # at most k (ties/zeros aside)
    assert (z != 0).sum(axis=1).max() == sae.k          # generically exactly k
    assert np.all(z >= 0)                                # relu before topk


def test_encode_preserves_batch_shape():
    sae = _toy_sae()
    x = np.random.default_rng(2).standard_normal((4, 6, sae.d_in)).astype(np.float32)
    z = encode(sae, x)
    assert z.shape == (4, 6, sae.n_latents)
    assert decode(sae, z).shape == (4, 6, sae.d_in)


def test_decode_matches_manual():
    sae = _toy_sae()
    x = np.random.default_rng(3).standard_normal((5, sae.d_in)).astype(np.float32)
    z = encode(sae, x)
    assert np.allclose(decode(sae, z), z @ sae.W_dec + sae.b_dec)


def test_reconstruction_r2_matches_formula():
    # The metric should equal 1 - SS_res/SS_tot computed from the same encode/decode.
    # (Untrained random weights give a poor/negative R² — that's expected; real
    # trained SAEs are validated for quality in scripts/verify_saes.py.)
    sae = _toy_sae(d=6, m=12, k=2, seed=7)
    x = np.random.default_rng(8).standard_normal((200, sae.d_in)).astype(np.float32)
    xhat = decode(sae, encode(sae, x))
    expected = 1.0 - np.sum((x - xhat) ** 2) / np.sum((x - x.mean(axis=0)) ** 2)
    assert np.isclose(reconstruction_r2(sae, x), expected)


def test_reconstruction_r2_is_one_when_perfect():
    # An identity-decoder SAE that reproduces x exactly -> R² == 1.
    d = 4
    sae = TopKSAE(
        hookpoint="layers.0",
        W_enc=np.eye(d, dtype=np.float32),
        b_enc=np.zeros(d, dtype=np.float32),
        W_dec=np.eye(d, dtype=np.float32),
        b_dec=np.zeros(d, dtype=np.float32),
        k=d,
    )
    x = np.abs(np.random.default_rng(0).standard_normal((30, d))).astype(np.float32)
    assert np.isclose(reconstruction_r2(sae, x), 1.0)  # relu+topk keep all of nonneg x


def test_source_hookpoint_indexing():
    # v_{l'} matches layers.{l'-1} for l'>=1; v_0 is a flagged proxy.
    assert source_hookpoint(1, 6) == ("layers.0", True)
    assert source_hookpoint(5, 6) == ("layers.4", True)
    assert source_hookpoint(0, 6) == ("layers.0", False)
    with pytest.raises(ValueError):
        source_hookpoint(6, 6)        # no source at v_L
    with pytest.raises(ValueError):
        source_hookpoint(-1, 6)
