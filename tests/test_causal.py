"""SSM-side causal machinery (Phase 4 Step 3): feature contribution δ_ℓ."""

from __future__ import annotations

import numpy as np
import torch

from layer_dynamics.causal import (
    feature_delta_contribution,
    group_delta_contribution,
    group_random_delta_contribution,
    magnitude_matched_features,
    top_and_control_examples,
)
from layer_dynamics.diagnostics.decompose_sanity import decoder_directions
from layer_dynamics.ssm import LowRankSSM, SSMDims


def _toy(seed: int = 0) -> LowRankSSM:
    dims = SSMDims(n_layers=5, d_model=7, d_input=6, d_state=3, rank=2)
    return LowRankSSM(dims, seed=seed)


def test_contribution_is_zero_before_home_layer():
    model = _toy()
    u = torch.randn(4, 5, 6)
    ell_star, m = 2, 3
    delta = feature_delta_contribution(model, u, m, ell_star, batch_size=2)
    # the input only enters at its home layer and afterwards → no effect before it
    assert torch.allclose(delta[:, :ell_star], torch.zeros_like(delta[:, :ell_star]), atol=1e-6)
    # ...and a nonzero effect from the home layer onward (generic operators)
    assert delta[:, ell_star:].abs().max() > 1e-6


def test_home_layer_contribution_equals_feedthrough_column():
    # δ_{ℓ*} = u[:, ℓ*, m] · D_{ℓ*}[:, m] exactly (the SSM is linear in u; at the home layer
    # only the feedthrough sees the coordinate — the state z_{ℓ*} predates this input).
    model = _toy(seed=1)
    u = torch.randn(8, 5, 6)
    ell_star, m = 1, 4
    delta = feature_delta_contribution(model, u, m, ell_star)
    d_col = torch.from_numpy(decoder_directions(model, [m], [ell_star])[0])  # [D]
    expected = u[:, ell_star, m][:, None] * d_col[None, :]
    torch.testing.assert_close(delta[:, ell_star], expected, atol=1e-5, rtol=1e-4)


def test_ablating_silent_coordinate_is_a_noop():
    # mechanism-sanity check (item 4's control): zeroing an already-zero coordinate → δ = 0.
    model = _toy()
    u = torch.randn(5, 5, 6)
    ell_star, m = 3, 2
    u[:, ell_star, m] = 0.0
    delta = feature_delta_contribution(model, u, m, ell_star)
    assert torch.allclose(delta, torch.zeros_like(delta), atol=1e-6)


def test_batched_matches_unbatched():
    model = _toy()
    u = torch.randn(10, 5, 6)
    a = feature_delta_contribution(model, u, 0, 0, batch_size=3)
    b = feature_delta_contribution(model, u, 0, 0, batch_size=10)
    torch.testing.assert_close(a, b)


def test_top_and_control_exact_zero():
    a = np.array([0.0, 5.0, 0.0, 3.0, 0.0, 0.0, 1.0])
    top, ctrl, rule = top_and_control_examples(a, n=2)
    assert rule == "exact-zero"
    assert set(top.tolist()) == {1, 3}            # the two largest |a|
    assert np.all(a[ctrl] == 0.0)                 # controls are silent
    assert len(ctrl) == 2


def test_top_and_control_fallback_when_no_zeros():
    a = np.linspace(0.1, 1.0, 10)                  # no exact zeros
    top, ctrl, rule = top_and_control_examples(a, n=3)
    assert rule == "5th-percentile-fallback"
    assert np.abs(a[ctrl]).max() < np.abs(a[top]).min()  # controls are the weakest


def test_magnitude_matched_features_respects_tolerance():
    rng = np.random.default_rng(0)
    n, L, M = 40, 3, 8
    u = rng.normal(size=(n, L, M)).astype(np.float32)
    cand_l, cand_m = 1, 2
    top_idx = np.arange(10)
    u[top_idx, cand_l, cand_m] = 4.0                # candidate mean |act| = 4 on top
    # plant matched features at the same home layer
    u[top_idx, cand_l, 5] = 4.2
    u[top_idx, cand_l, 6] = 3.9
    feat_layers = np.full(M, cand_l)                # all slots home at cand_l (test only)
    feats, tol = magnitude_matched_features(
        u, cand_l, cand_m, top_idx, n_rand=2, feat_layers=feat_layers, tol=0.2, rng=rng
    )
    target = 4.0
    for ell, m in feats:
        assert (ell, m) != (cand_l, cand_m)         # candidate excluded
        mm = np.abs(u[top_idx, ell, m]).mean()
        assert target * 0.8 <= mm <= target * 1.2   # within ±20%


def test_group_contribution_grows_with_k_and_matches_full_at_capacity():
    model = _toy(seed=3)
    u = torch.randn(6, 5, 6)
    d1 = group_delta_contribution(model, u, 1)
    d_all = group_delta_contribution(model, u, u[0].numel())  # zero every coord
    # zeroing all coords == removing the full feedthrough+state contribution (down to bias)
    full = model(u) - model(torch.zeros_like(u))
    torch.testing.assert_close(d_all, full, atol=1e-5, rtol=1e-4)
    # more coords ablated → at least as much total contribution removed
    assert d_all.abs().sum() >= d1.abs().sum()


def test_group_random_count_matches_k_and_skips_inactive():
    # count-match removes exactly min(k, n_active) coords, all from the active set.
    model = _toy(seed=4)
    u = torch.zeros(5, 5, 6)
    u.reshape(5, -1)[:, [0, 7, 13, 20]] = torch.tensor([1.0, -2.0, 3.0, 0.5])  # 4 active/example
    _, n_abl, _ = group_random_delta_contribution(model, u, 2, seed=0, match="count")
    assert torch.all(n_abl == 2)  # exactly k, never an inactive (zero) coordinate


def test_group_random_count_caps_at_active_set_and_equals_full():
    # k beyond the active set → all (and only) active coords ablated == the full contribution.
    model = _toy(seed=5)
    u = torch.zeros(3, 5, 6)
    u.reshape(3, -1)[:, [2, 9]] = torch.tensor([1.0, 4.0])  # 2 active/example
    d, n_abl, _ = group_random_delta_contribution(model, u, k=10, seed=1, match="count")
    assert torch.all(n_abl == 2)  # capped at the active set, never selects a zero coord
    full = model(u) - model(torch.zeros_like(u))
    torch.testing.assert_close(d, full, atol=1e-5, rtol=1e-4)


def test_group_random_mass_match_reaches_topk_mass():
    # mass-match removes random active coords until their |u| sum reaches the top-k mass.
    model = _toy(seed=6)
    u = torch.zeros(4, 5, 6)
    u.reshape(4, -1)[:, [0, 7, 13, 20]] = torch.tensor([1.0, -2.0, 3.0, 0.5])
    target = 3.0 + 2.0  # top-2 |u|
    _, n_abl, mass = group_random_delta_contribution(model, u, 2, seed=0, match="mass")
    assert torch.all(mass >= target - 1e-5)  # matched (or just over) the top-k mass
    assert torch.all(n_abl >= 2)             # needs at least the 2 largest-equivalent mass


def test_group_random_is_seed_deterministic():
    model = _toy(seed=7)
    u = torch.randn(6, 5, 6)  # dense (all active) → selection is a free random choice
    a, _, _ = group_random_delta_contribution(model, u, 3, seed=0)
    b, _, _ = group_random_delta_contribution(model, u, 3, seed=0)
    c, _, _ = group_random_delta_contribution(model, u, 3, seed=1)
    torch.testing.assert_close(a, b)            # same seed → identical selection
    assert (a - c).abs().max() > 0              # different seed → different selection


def test_group_random_matches_unbatched():
    model = _toy(seed=8)
    u = torch.randn(10, 5, 6)
    a, na, ma = group_random_delta_contribution(model, u, 4, seed=2, batch_size=3)
    b, nb, mb = group_random_delta_contribution(model, u, 4, seed=2, batch_size=10)
    torch.testing.assert_close(a, b)            # batch-size-independent
    torch.testing.assert_close(na, nb)
    torch.testing.assert_close(ma, mb)


def test_contribution_is_linear_in_activation():
    # doubling the coordinate doubles δ (linearity is the basis of the magnitude-matching null).
    model = _toy(seed=2)
    u = torch.randn(6, 5, 6)
    ell_star, m = 0, 1
    d1 = feature_delta_contribution(model, u, m, ell_star)
    u2 = u.clone()
    u2[:, ell_star, m] *= 2.0
    d2 = feature_delta_contribution(model, u2, m, ell_star)
    torch.testing.assert_close(d2, 2.0 * d1, atol=1e-5, rtol=1e-4)
    assert np.isfinite(d1.numpy()).all()
