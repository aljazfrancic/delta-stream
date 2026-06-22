"""Tests for the residual-stream collection helpers (Phase 1 Step 2)."""

from __future__ import annotations

from layer_dynamics.activations import resid_hook_names


def test_resid_hook_names_shape():
    names = resid_hook_names(24)
    assert len(names) == 25  # v_0 .. v_24
    assert names[0] == "blocks.0.hook_resid_pre"          # v_0 = embedding output
    assert names[1] == "blocks.0.hook_resid_post"         # v_1
    assert names[-1] == "blocks.23.hook_resid_post"       # v_24 = final state
    assert len(set(names)) == len(names)                  # no duplicates
