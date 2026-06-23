"""Tests for the Phase 3 Step 4 sanity-check diagnostics (Branch B)."""

from __future__ import annotations

import numpy as np
import torch

from layer_dynamics.diagnostics.decompose_sanity import (
    collapse_flags,
    decoder_directions,
    direction_cosines,
    feature_mass,
    feature_overlap,
    feature_top_examples,
    match_rate,
    reconstruction_pairs,
    top_features,
)
from layer_dynamics.ssm import LowRankSSM, SSMDims


def _dims(L=3, D=4, M=6, dz=3, r=2):
    return SSMDims(n_layers=L, d_model=D, d_input=M, d_state=dz, rank=r)


def _u_with_planted_feature():
    """An input tensor where feature 2 carries the most mass, peaking at layer 1."""
    u = torch.zeros(5, 3, 6)
    u[:, 1, 2] = torch.tensor([10.0, 9.0, 8.0, 1.0, 0.5])  # home layer 1, biggest mass
    u[:, 0, 2] = 1.0  # some mass at layer 0 too, but less
    u[:, 0, 4] = 3.0  # a second-ranked feature
    return u


# --- feature ranking ---


def test_feature_mass_shape_and_values():
    u = _u_with_planted_feature()
    mass = feature_mass(u)
    assert mass.shape == (3, 6)
    # feature 2's mass sits mostly at layer 1
    assert mass[1, 2] > mass[0, 2]


def test_top_features_ranks_and_home_layer():
    u = _u_with_planted_feature()
    feats = top_features(u, top_n=2)
    assert feats[0]["feature"] == 2
    assert feats[0]["home_layer"] == 1
    # mass is descending
    assert feats[0]["mass"] >= feats[1]["mass"]


def test_top_features_caps_at_m():
    u = torch.randn(4, 3, 6).abs()
    assert len(top_features(u, top_n=100)) == 6


def test_feature_top_examples_orders_by_activation():
    u = _u_with_planted_feature()
    ex = feature_top_examples(u, feature=2, layer=1, top_e=3)
    assert ex == [0, 1, 2]  # 10, 9, 8 are the strongest


# --- decoder directions + matching ---


def test_decoder_directions_match_explicit_column():
    m = LowRankSSM(_dims(), seed=1, init_scale=0.5)
    dirs = decoder_directions(m, [2, 0], [0, 1])
    expected0 = (m.D1[0] @ m.D2[0][2]).detach().numpy()
    expected1 = (m.D1[1] @ m.D2[1][0]).detach().numpy()
    assert np.allclose(dirs[0], expected0, atol=1e-5)
    assert np.allclose(dirs[1], expected1, atol=1e-5)
    assert dirs.shape == (2, 4)


def test_decoder_directions_length_mismatch_raises():
    m = LowRankSSM(_dims(), seed=0)
    try:
        decoder_directions(m, [0, 1], [0])
    except ValueError:
        return
    raise AssertionError("expected ValueError on mismatched lengths")


def test_direction_cosines_identical_is_one():
    a = np.random.default_rng(0).standard_normal((5, 4))
    assert np.allclose(direction_cosines(a, a), 1.0, atol=1e-6)


def test_direction_cosines_orthogonal_is_zero():
    a = np.array([[1.0, 0.0], [0.0, 1.0]])
    b = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert np.allclose(direction_cosines(a, b), 0.0, atol=1e-6)


def test_match_rate_threshold():
    cos = np.array([0.9, 0.8, 0.6, 0.71, 0.69])
    assert match_rate(cos, 0.7) == 0.6  # 0.9, 0.8, 0.71
    assert match_rate(np.array([]), 0.7) == 0.0


# --- dataset overlap ---


def test_feature_overlap_jaccard():
    ov = feature_overlap([1, 2, 3, 4], [3, 4, 5, 6])
    assert ov["intersection"] == 2
    assert ov["jaccard"] == 2 / 6


def test_feature_overlap_disjoint_and_identical():
    assert feature_overlap([1, 2], [3, 4])["jaccard"] == 0.0
    assert feature_overlap([1, 2], [1, 2])["jaccard"] == 1.0


# --- reconstruction pairs ---


def test_reconstruction_pairs_shapes_and_actual():
    m = LowRankSSM(_dims(L=3, D=4, M=6), seed=0)
    u = torch.randn(8, 3, 6)
    delta = torch.randn(8, 3, 4)
    pred, actual = reconstruction_pairs(m, u, delta, [0, 3, 5])
    assert pred.shape == (3, 3, 4)
    assert actual.shape == (3, 3, 4)
    assert np.allclose(actual, delta[[0, 3, 5]].numpy())


# --- collapse flagging ---


def test_collapse_flags_detects_degenerate():
    stats = {"op_A": 1e-6, "op_B": 1e-6, "op_C": 1e-6, "op_D": 1e-6,
             "b_norm": 5.0, "l0": 1e-9}
    out = collapse_flags(stats)
    assert out["collapsed"] is True


def test_collapse_flags_healthy_fit():
    stats = {"op_A": 2.0, "op_B": 10.0, "op_C": 14.0, "op_D": 170.0,
             "b_norm": 4.7, "l0": 1e-3}
    out = collapse_flags(stats)
    assert out["collapsed"] is False
    assert out["operator_total_fro"] > 100.0
