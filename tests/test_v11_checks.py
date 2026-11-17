"""Tests for the v1.1 on-box checks' verdict logic (pure functions)."""

from __future__ import annotations

import copy

import pytest

from layer_dynamics.diagnostics.v11_checks import (
    PREFIX_LATE_GAIN,
    PROBE_PRIMARY,
    ci_verdict,
    depth_verdict,
    learned_share_verdict,
    paired_gap_verdict,
    prefix_verdict,
    probe_overall,
    probe_reproduction,
    probe_verdict,
    seeds_verdict,
)


def _ci(lo, hi):
    return {"diff": (lo + hi) / 2, "ci_lo": lo, "ci_hi": hi, "significant": lo > 0 or hi < 0}


def test_ci_verdict_rows():
    assert ci_verdict(_ci(0.01, 0.09)) == "survives"
    assert ci_verdict(_ci(-0.09, -0.01)) == "reversed"
    assert ci_verdict(_ci(-0.02, 0.05)) == "not distinguishable"
    assert ci_verdict(_ci(0.0, 0.05)) == "not distinguishable"     # touching zero is not above it
    assert probe_verdict(_ci(0.01, 0.09)) == "survives"


def test_probe_primary_is_the_causal_state_against_its_own_codes_at_mid_depth():
    assert PROBE_PRIMARY == "object/state_causal−prefix_pca_causal/mid"


def test_probe_overall_rows():
    assert probe_overall({"mlsae": "survives", "l1": "survives"}) == "survives on every dictionary"
    assert probe_overall({"mlsae": "survives", "l1": "not distinguishable"}) == \
        "dictionary-dependent"
    assert probe_overall({"mlsae": "reversed", "l1": "not distinguishable"}) == "does not survive"
    assert probe_overall({}) == "undetermined"


def _probe_table():
    arms = ["state", "code_pca"]
    return {
        "object_top1_acc": {a: [0.1, 0.2, 0.3] for a in arms},
        "relation_acc": {a: [0.9, 0.95, 1.0] for a in arms},
        "conf_logit_r2": {a: [0.0, 0.1, 0.2] for a in arms},
        "band_contrasts": {"object/state−code_pca/mid": _ci(0.02, 0.08)},
    }


def test_probe_reproduction_passes_on_identical_tables_and_ignores_new_arms():
    rec = _probe_table()
    new = copy.deepcopy(rec)
    new["object_top1_acc"]["state_causal"] = [0.0, 0.0, 0.0]         # an arm R7 never had
    new["band_contrasts"]["object/state_causal−prefix_pca_causal/mid"] = _ci(-1, 1)
    rep = probe_reproduction(new, rec, ["state", "code_pca"], [0, 1, 2])
    assert rep["ok"] and rep["n_band_contrasts_compared"] == 1
    assert rep["max_abs_diff_object_top1"] == 0.0


@pytest.mark.parametrize("key", ["object_top1_acc", "relation_acc", "conf_logit_r2"])
def test_probe_reproduction_fails_on_a_moved_number(key):
    rec = _probe_table()
    new = copy.deepcopy(rec)
    new[key]["state"][1] += 1 / 311
    assert not probe_reproduction(new, rec, ["state", "code_pca"], [0, 1, 2])["ok"]
    # ... unless the moved layer is not one of the layers compared (a smoke subset)
    assert probe_reproduction(new, rec, ["state", "code_pca"], [0, 2])["ok"]


def test_probe_reproduction_fails_on_a_moved_contrast_or_missing_layer():
    rec = _probe_table()
    new = copy.deepcopy(rec)
    new["band_contrasts"]["object/state−code_pca/mid"]["ci_lo"] += 0.01
    assert not probe_reproduction(new, rec, ["state", "code_pca"], [0, 1, 2])["ok"]
    new = copy.deepcopy(rec)
    new["object_top1_acc"]["state"][0] = None
    assert not probe_reproduction(new, rec, ["state", "code_pca"], [0, 1, 2])["ok"]


def _rows(vals):
    return [{"cutoff": c, "train_r2": v, "test_r2": v} for c, v in vals.items()]


RECORDED = {0: -0.44, 3: -0.30, 7: -0.15, 11: 0.20, 15: 0.45, 19: 0.58, 23: 0.77}


def test_prefix_verdict_when_the_floor_was_an_unfit_bias():
    new = {0: -0.004, 3: 0.08, 7: 0.17, 11: 0.31, 15: 0.46, 19: 0.585, 23: 0.772}
    v = prefix_verdict(_rows(new), _rows(RECORDED))
    assert v["embed_only_at_mean_floor"] and v["embed_only_recorded_r2"] == -0.44
    assert v["late_gain"] == pytest.approx(0.187) and v["late_gain_from_cutoff"] == 19
    assert v["late_sources_load_bearing"] and v["monotone"]
    assert v["upper_half_stable"] and v["upper_half_max_move"] == pytest.approx(0.01)
    assert v["lower_half_max_move"] == pytest.approx(0.436)


def test_prefix_verdict_when_nothing_changes_or_the_claim_fails():
    v = prefix_verdict(_rows(RECORDED), _rows(RECORDED))
    assert not v["embed_only_at_mean_floor"] and v["upper_half_max_move"] == 0.0
    flat = {**RECORDED, 23: RECORDED[19] + PREFIX_LATE_GAIN - 0.01}
    v = prefix_verdict(_rows(flat), _rows(RECORDED))
    assert not v["late_sources_load_bearing"] and not v["upper_half_stable"]
    dip = {**RECORDED, 7: -0.5}
    assert not prefix_verdict(_rows(dip), _rows(RECORDED))["monotone"]


def test_prefix_verdict_with_a_single_cutoff_is_the_gate_shape():
    v = prefix_verdict(_rows({0: -0.44}), _rows(RECORDED))
    assert v["late_gain"] == 0.0 and v["upper_half_stable"] is None


def test_seeds_verdict():
    v = seeds_verdict([0.0077, 0.0081, 0.0069])
    assert v["robust"] and v["verdict"] == "robust to refit"
    assert v["mean"] == pytest.approx(0.00757, abs=1e-5)
    assert not seeds_verdict([0.0077, -0.0002, 0.0069])["robust"]        # a negative seed
    assert not seeds_verdict([0.0077, 0.0010, 0.0140])["robust"]         # positive but noisy
    assert not seeds_verdict([0.0077])["robust"]                         # one seed is no spread


def test_paired_gap_verdict():
    assert paired_gap_verdict(_ci(0.005, 0.010)) == "holds"
    assert paired_gap_verdict(_ci(-0.002, 0.004)) == "not established"
    assert paired_gap_verdict(_ci(-0.010, -0.005)) == "reversed"


def test_learned_share_verdict():
    assert learned_share_verdict(0.008, 0.007, 0.0014)["verdict"] == "largely architectural"
    assert learned_share_verdict(0.008, 0.004, 0.0014)["verdict"] == "about half"
    assert learned_share_verdict(0.008, 0.001, 0.0014)["verdict"] == "largely learned"
    assert learned_share_verdict(0.008, -0.004, 0.0014)["verdict"] == "largely learned"
    v = learned_share_verdict(0.003, 0.002, 0.0014)          # trained gap within 3 sd of zero
    assert v["verdict"] == "not resolvable" and v["untrained_share"] is None
    assert learned_share_verdict(-0.005, 0.002, 0.0014)["verdict"] == "not resolvable"
    assert learned_share_verdict(0.008, 0.004, float("nan"))["verdict"] == "not resolvable"


def test_depth_verdict():
    deep = [0.0077, 0.0081, 0.0069]
    assert depth_verdict([0.001, 0.0005, 0.0015], deep)["verdict"] == "larger at the deeper model"
    assert depth_verdict([0.007, 0.008, 0.0075], deep)["verdict"] == \
        "no depth trend at matched parameters"
    assert depth_verdict([0.020, 0.021, 0.019], deep)["verdict"] == \
        "larger at the shallower model"
    assert depth_verdict([0.001], deep)["verdict"] == "undetermined"
