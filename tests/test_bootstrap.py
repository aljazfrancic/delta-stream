"""Tests for diagnostics.bootstrap (R0 paired bootstrap) + the group-disjoint split (R5)."""

from __future__ import annotations

import numpy as np

from layer_dynamics.diagnostics.bootstrap import paired_gap_ci, per_example_ss, pooled_r2
from layer_dynamics.diagnostics.dataset import group_disjoint_split_idx


def _toy(n=64, ell=3, d=5, seed=0):
    rng = np.random.default_rng(seed)
    target = rng.normal(size=(n, ell, d)).astype(np.float32)
    mean = target.mean(axis=0, keepdims=True)
    return target, mean, rng


def test_per_example_ss_matches_pooled_convention():
    target, mean, rng = _toy()
    pred = target + 0.1 * rng.normal(size=target.shape)
    ss_res, ss_tot = per_example_ss(pred, target, mean)
    assert ss_res.shape == (64,)
    r2 = pooled_r2(ss_res, ss_tot)
    # matches the direct pooled computation
    direct = 1.0 - ((target - pred) ** 2).sum() / ((target - mean) ** 2).sum()
    assert abs(r2 - direct) < 1e-6


def test_perfect_prediction_r2_one():
    target, mean, _ = _toy()
    ss_res, ss_tot = per_example_ss(target, target, mean)
    assert pooled_r2(ss_res, ss_tot) == 1.0


def test_paired_gap_ci_detects_clear_separation():
    target, mean, rng = _toy(n=200)
    pred_a = target + 0.1 * rng.normal(size=target.shape)   # good model
    pred_b = target + 0.8 * rng.normal(size=target.shape)   # clearly worse
    ss_a, ss_tot = per_example_ss(pred_a, target, mean)
    ss_b, _ = per_example_ss(pred_b, target, mean)
    ci = paired_gap_ci(ss_a, ss_b, ss_tot, n_boot=2000, seed=0)
    assert ci["gap"] > 0
    assert ci["ci_lo"] > 0 and ci["significant"]
    assert ci["ci_lo"] < ci["gap"] < ci["ci_hi"]


def test_paired_gap_ci_tie_is_not_significant():
    target, mean, rng = _toy(n=200)
    noise = rng.normal(size=target.shape)
    pred_a = target + 0.3 * noise
    pred_b = target + 0.3 * rng.permutation(noise, axis=2)  # same error scale
    ss_a, ss_tot = per_example_ss(pred_a, target, mean)
    ss_b, _ = per_example_ss(pred_b, target, mean)
    ci = paired_gap_ci(ss_a, ss_b, ss_tot, n_boot=2000, seed=0)
    assert ci["ci_lo"] < 0 < ci["ci_hi"]
    assert not ci["significant"]


def test_paired_gap_ci_deterministic_and_validates():
    target, mean, rng = _toy()
    pred = target + 0.2 * rng.normal(size=target.shape)
    ss_a, ss_tot = per_example_ss(pred, target, mean)
    c1 = paired_gap_ci(ss_a, ss_a * 1.1, ss_tot, n_boot=500, seed=7)
    c2 = paired_gap_ci(ss_a, ss_a * 1.1, ss_tot, n_boot=500, seed=7)
    assert c1 == c2
    try:
        paired_gap_ci(ss_a[:-1], ss_a, ss_tot, n_boot=10)
        raise AssertionError("length mismatch should raise")
    except ValueError:
        pass


def test_group_disjoint_split_no_group_straddles():
    rng = np.random.default_rng(0)
    groups = rng.choice([f"subj{i}" for i in range(40)], size=300)
    tr, te = group_disjoint_split_idx(groups, test_frac=0.2, seed=1)
    assert len(tr) + len(te) == 300
    assert set(tr).isdisjoint(te)
    assert set(groups[tr]).isdisjoint(set(groups[te]))
    # test fraction approximately honoured (whole groups → some overshoot allowed)
    assert 0.15 <= len(te) / 300 <= 0.35


def test_group_disjoint_split_deterministic():
    groups = np.array(["a", "a", "b", "c", "c", "c", "d", "e"] * 10)
    s1 = group_disjoint_split_idx(groups, seed=3)
    s2 = group_disjoint_split_idx(groups, seed=3)
    assert np.array_equal(s1[0], s2[0]) and np.array_equal(s1[1], s2[1])
