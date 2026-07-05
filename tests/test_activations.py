"""Tests for the residual-stream collection helpers (Phase 1 Step 2 + R8)."""

from __future__ import annotations

import numpy as np

from layer_dynamics.activations import (
    build_pseudo_stream,
    resid_hook_names,
    stream_additivity_stats,
    stream_write_hook_names,
)
from layer_dynamics.diagnostics.dataset import deltas


def test_resid_hook_names_shape():
    names = resid_hook_names(24)
    assert len(names) == 25  # v_0 .. v_24
    assert names[0] == "blocks.0.hook_resid_pre"          # v_0 = embedding output
    assert names[1] == "blocks.0.hook_resid_post"         # v_1
    assert names[-1] == "blocks.23.hook_resid_post"       # v_24 = final state
    assert len(set(names)) == len(names)                  # no duplicates


def test_stream_write_hook_names():
    names = stream_write_hook_names(24)
    assert len(names["attn"]) == len(names["mlp"]) == 24  # one write per layer
    assert names["attn"][0] == "blocks.0.hook_attn_out"
    assert names["mlp"][-1] == "blocks.23.hook_mlp_out"
    assert not set(names["attn"]) & set(names["mlp"])


def test_build_pseudo_stream_roundtrip():
    rng = np.random.default_rng(0)
    v0 = rng.normal(size=(5, 8)).astype(np.float32)
    writes = rng.normal(size=(5, 6, 8)).astype(np.float32)
    w = build_pseudo_stream(v0, writes)
    assert w.shape == (5, 7, 8) and w.dtype == np.float32
    np.testing.assert_array_equal(w[:, 0], v0)
    # deltas(w) must recover the writes to fp32 rounding — the property every
    # downstream --h5 diagnostic relies on.
    np.testing.assert_allclose(deltas(w), writes, atol=1e-5)


def test_stream_additivity_stats_exact_and_violated():
    rng = np.random.default_rng(1)
    attn = rng.normal(size=(4, 3, 8)).astype(np.float32)
    mlp = rng.normal(size=(4, 3, 8)).astype(np.float32)
    stats = stream_additivity_stats(attn, mlp, attn + mlp)
    assert stats["max_abs_err"] == 0.0 and stats["rel_frobenius_err"] == 0.0
    assert len(stats["attn_energy_share"]) == 3
    assert all(0.0 <= s <= 1.0 for s in stats["attn_energy_share"])
    bad = stream_additivity_stats(attn, mlp, attn + mlp + 1.0)
    assert bad["max_abs_err"] > 0.5 and bad["rel_frobenius_err"] > 0.01
