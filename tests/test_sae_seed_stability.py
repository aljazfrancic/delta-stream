"""Tests for the dictionary-level SAE seed-stability statistics (v1.1 E1; synthetic)."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from layer_dynamics.diagnostics.sae_seed_stability import (
    PRIMARY,
    LayerDictionary,
    compare_pair,
    cosine_matrix,
    distant_layer,
    init_retention,
    layer_dictionary,
    load_dictionaries,
    null_status,
    one_to_one_mean_cosine,
    pair_statistics,
    random_dictionary_like,
    reproduction_check,
    summarise,
    top_by_mass,
    unit_rows,
    verdict,
)


def _dict(n=40, d=64, seed=0, layer=1) -> LayerDictionary:
    rng = np.random.default_rng(seed)
    return LayerDictionary(layer=layer, directions=unit_rows(rng.standard_normal((n, d))),
                           mass=rng.random(n) + 0.1, latent_index=np.arange(n))


def _permuted(d: LayerDictionary, seed=1) -> LayerDictionary:
    perm = np.random.default_rng(seed).permutation(d.n_alive)
    return LayerDictionary(layer=d.layer, directions=d.directions[perm], mass=d.mass[perm],
                           latent_index=d.latent_index[perm])


# --- building blocks ---

def test_unit_rows_and_cosine_matrix_are_signed():
    a = np.array([[2.0, 0.0], [0.0, 3.0]])
    assert np.allclose(np.linalg.norm(unit_rows(a), axis=1), 1.0)
    cos = cosine_matrix(a, np.array([[-1.0, 0.0], [0.0, 5.0]]))
    assert np.allclose(cos, [[-1.0, 0.0], [0.0, 1.0]])  # d and −d are different features


def test_top_by_mass_orders_descending_and_truncates():
    assert top_by_mass(np.array([1.0, 5.0, 3.0]), 2).tolist() == [1, 2]
    assert top_by_mass(np.array([1.0, 5.0, 3.0]), 10).tolist() == [1, 2, 0]


def test_one_to_one_is_a_permutation_not_a_best_match():
    # both rows prefer column 0; a one-to-one matching must give one of them column 1
    cos = np.array([[0.9, 0.1], [0.8, 0.2]])
    mean, n = one_to_one_mean_cosine(cos)
    assert n == 2 and mean == pytest.approx((0.9 + 0.2) / 2)
    assert cos.max(axis=1).mean() == pytest.approx(0.85)  # best-match is the looser number


def test_one_to_one_rectangular_and_empty():
    assert one_to_one_mean_cosine(np.zeros((0, 3))) == (0.0, 0)
    mean, n = one_to_one_mean_cosine(np.array([[1.0, 0.0, 0.5]]))
    assert n == 1 and mean == pytest.approx(1.0)


# --- the statistic on known cases ---

def test_permuted_copy_is_perfectly_stable():
    """A latent permutation is invisible to both best-match and one-to-one matching."""
    d = _dict()
    s = pair_statistics(d, _permuted(d), top_n=10)
    assert s["match_top"] == 1.0 and s["match_all"] == 1.0
    assert s["mean_maxcos_all"] == pytest.approx(1.0)
    assert s["onetoone_mean_cos"] == pytest.approx(1.0) and s["onetoone_n"] == d.n_alive


def test_independent_random_dictionaries_are_unstable():
    s = pair_statistics(_dict(seed=0), _dict(seed=1), top_n=10)
    assert s["match_top"] == 0.0 and s["match_all"] == 0.0
    assert s["mean_maxcos_all"] < 0.6  # max of 40 cosines in 64-d, far below the bar


def test_half_shared_dictionary_scores_one_half():
    a, fresh = _dict(seed=0), _dict(seed=2)
    dirs = a.directions.copy()
    dirs[20:] = fresh.directions[20:]  # keep the first 20 directions, replace the rest
    b = LayerDictionary(layer=1, directions=dirs, mass=a.mass, latent_index=a.latent_index)
    s = pair_statistics(a, b, top_n=40)
    assert s["match_all"] == pytest.approx(0.5)


def test_top_n_restricts_the_querying_side_only():
    """Top latents shared, tail replaced: top-n is stable while the full set is not."""
    a = _dict(seed=0)
    a.mass[:] = np.arange(a.n_alive, 0, -1)  # row 0 heaviest
    dirs = a.directions.copy()
    dirs[10:] = _dict(seed=3).directions[10:]
    b = LayerDictionary(layer=1, directions=dirs, mass=a.mass.copy(),
                        latent_index=a.latent_index)
    s = pair_statistics(a, b, top_n=10)
    assert s["match_top"] == 1.0 and s["n_top_a"] == 10
    assert s["match_all"] == pytest.approx(0.25)


def test_statistics_are_symmetric_in_the_pair():
    a, b = _dict(seed=0), _dict(n=30, seed=4)
    s, t = pair_statistics(a, b, top_n=10), pair_statistics(b, a, top_n=10)
    for k in ("match_top", "match_all", "mean_maxcos_top", "mean_maxcos_all",
              "onetoone_mean_cos"):
        assert s[k] == pytest.approx(t[k])
    assert s["onetoone_n"] == 30  # min(n_a, n_b)


# --- nulls ---

def test_random_dictionary_like_matches_shape_and_is_unit_norm():
    d = _dict(seed=0)
    r = random_dictionary_like(d, np.random.default_rng(7))  # not the stream _dict(0) used
    assert r.directions.shape == d.directions.shape
    assert np.allclose(np.linalg.norm(r.directions, axis=1), 1.0)
    assert not np.allclose(r.directions, d.directions)


def test_distant_layer_is_half_the_ring_away():
    layers = list(range(1, 24))  # the 23 trained 410m layers
    assert distant_layer(1, layers) == 12
    assert distant_layer(13, layers) == 1
    assert all(distant_layer(ell, layers) != ell for ell in layers)
    with pytest.raises(ValueError):
        distant_layer(1, [1])


def test_compare_pair_emits_real_nulls_and_diagnostic_per_layer():
    da = {ell: _dict(seed=ell, layer=ell) for ell in (1, 2, 3, 4)}
    db = {ell: _permuted(d, seed=10 + ell) for ell, d in da.items()}
    rows = compare_pair(da, db, top_n=10, rng=np.random.default_rng(0))
    assert len(rows) == 4 * 4
    by_kind = {k: [r for r in rows if r["kind"] == k] for k in {r["kind"] for r in rows}}
    assert set(by_kind) == {"real", "null_random", "null_xlayer_cross_seed",
                            "diag_xlayer_same_seed"}
    assert all(r[PRIMARY] == 1.0 for r in by_kind["real"])        # permuted copies
    assert all(r[PRIMARY] == 0.0 for r in by_kind["null_random"])
    assert all(r["layer_b"] != r["layer"] for r in by_kind["diag_xlayer_same_seed"])
    assert all(r[PRIMARY] == 0.0 for r in by_kind["null_xlayer_cross_seed"])


def test_compare_pair_single_layer_skips_cross_layer_nulls():
    da = {1: _dict()}
    rows = compare_pair(da, {1: _permuted(da[1])}, top_n=10)
    assert {r["kind"] for r in rows} == {"real", "null_random"}


# --- aggregation, verdict ---

def test_summarise_averages_layers_then_pairs():
    base = dict.fromkeys(("match_all", "mean_maxcos_top", "mean_maxcos_all",
                          "onetoone_mean_cos"), 0.0)
    rows = [  # pair A has 1 layer at 1.0; pair B has 3 layers at 0.0 → mean of means 0.5
        {"pair": "A", "kind": "real", "match_top": 1.0, **base},
        *[{"pair": "B", "kind": "real", "match_top": 0.0, **base} for _ in range(3)],
    ]
    out = summarise(rows)
    assert out["real"]["match_top"] == pytest.approx(0.5)
    assert out["real"]["n_pairs"] == 2


def test_null_status_flags_vacuous_and_marginal_statistics():
    def summ(real, null):
        return {"real": {PRIMARY: real}, "null_random": {PRIMARY: null}}

    assert null_status(summ(0.9, 0.1))["informative"]
    assert not null_status(summ(0.9, 0.8))["informative"]   # a null clears the bar
    assert not null_status(summ(0.75, 0.6))["informative"]  # stable, but within the margin
    assert null_status(summ(0.1, 0.05))["informative"]      # unstable at null level is fine
    # the same-seed cross-layer comparison shares an initialisation: a diagnostic, not a null
    lazy = {**summ(0.0, 0.0), "diag_xlayer_same_seed": {PRIMARY: 1.0}}
    assert null_status(lazy)["informative"] and "diag_xlayer_same_seed" not in (
        null_status(lazy)["nulls"])


def test_verdict_rows_cover_the_table():
    assert verdict(1.0, 0.1)["row"].startswith("L1 stable / TopK unstable")
    assert verdict(0.2, 0.1)["row"].startswith("neither")
    assert verdict(0.9, 0.9)["row"].startswith("TopK dictionaries reproduce")
    assert "bug" in verdict(0.1, 0.9)["row"]
    assert verdict(0.7, 0.69) == {**verdict(0.7, 0.69), "l1_stable": True, "topk_stable": False}


def test_reproduction_check_accepts_exact_and_rejects_drift():
    ref = {"alive_count": [0, 100, 200], "recon_r2": [None, 0.95, 0.93], "M_star": 200}
    ok = reproduction_check(np.array([0, 100, 200]), np.array([np.nan, 0.95, 0.93]), ref)
    assert ok["ok"] and ok["alive_identical"] and ok["max_alive_rel_diff"] == 0.0
    near = reproduction_check(np.array([0, 102, 199]), np.array([np.nan, 0.951, 0.929]), ref)
    assert near["ok"] and not near["alive_identical"]
    drift = reproduction_check(np.array([0, 150, 200]), np.array([np.nan, 0.95, 0.93]), ref)
    assert not drift["ok"]
    worse_r2 = reproduction_check(np.array([0, 100, 200]), np.array([np.nan, 0.90, 0.93]), ref)
    assert not worse_r2["ok"]
    assert not reproduction_check(np.array([0, 100]), np.array([np.nan, 0.95]), ref)["ok"]


# --- loading the trainer's artefact pair ---

def test_layer_dictionary_selects_alive_columns_and_mass():
    w_dec = np.eye(4)[:, [0, 1, 2, 3]] * 2.0          # [D=4, M=4], non-unit columns
    feat_index = np.array([3, 1, -1])                  # two alive slots, one pad
    u_layer = np.array([[1.0, -2.0, 0.0], [3.0, 0.0, 0.0]])
    d = layer_dictionary(w_dec, feat_index, 2, u_layer, layer=5)
    assert d.n_alive == 2 and d.layer == 5 and d.latent_index.tolist() == [3, 1]
    assert np.allclose(d.directions, np.eye(4)[[3, 1]])  # renormalised decoder columns
    assert d.mass.tolist() == [4.0, 2.0]
    assert d.init_cos is None and init_retention(d) == {}
    with pytest.raises(ValueError):
        layer_dictionary(w_dec, np.array([7, 1]), 2, u_layer, layer=5)


def test_init_retention_separates_untouched_from_moved_latents():
    w_init = np.eye(4)
    w_dec = np.eye(4).copy()
    w_dec[:, 3] = [1.0, 0.0, 0.0, 0.0]                 # latent 3 left its init entirely
    u_layer = np.array([[5.0, 1.0, 1.0, 1.0]])         # latent 0 carries the mass
    d = layer_dictionary(w_dec, np.arange(4), 4, u_layer, layer=1, w_init=w_init)
    assert d.init_cos.tolist() == [1.0, 1.0, 1.0, 0.0]
    r = init_retention(d, top_n=1)
    assert r["init_frac_all"] == pytest.approx(0.75) and r["init_frac_top"] == 1.0
    assert r["init_cos_mean_all"] == pytest.approx(0.75)


def test_load_dictionaries_roundtrips_a_trainer_artefact(tmp_path):
    rng = np.random.default_rng(0)
    n, n_layers, d, m, m_star = 6, 3, 8, 12, 4
    alive = [np.empty(0, dtype=np.int64), np.array([2, 5, 7, 9]), np.array([1, 4])]
    u = np.zeros((n, n_layers, m_star), dtype=np.float32)
    feat_index = np.full((n_layers, m_star), -1, dtype=np.int64)
    state_dicts: list[dict | None] = [None]
    for layer in (1, 2):
        k = len(alive[layer])
        feat_index[layer, :k] = alive[layer]
        u[:, layer, :k] = rng.random((n, k)).astype(np.float32)
        state_dicts.append({"W_dec": rng.standard_normal((d, m)).astype(np.float32)})
    npz = tmp_path / "sources.npz"
    np.savez(npz, u=u, feat_index=feat_index, alive_count=np.array([0, 4, 2]))
    with open(npz.with_suffix(".weights.pkl"), "wb") as fh:
        pickle.dump({"state_dicts": state_dicts}, fh)

    dicts = load_dictionaries(npz)
    assert sorted(dicts) == [1, 2]                      # v_0 has no SAE
    assert dicts[1].init_cos is None                    # pre-v1.1 pickle: no seed recorded
    assert dicts[1].n_alive == 4 and dicts[2].n_alive == 2
    want = unit_rows(state_dicts[2]["W_dec"][:, [1, 4]].T)
    assert np.allclose(dicts[2].directions, want)
    assert np.allclose(dicts[1].mass, np.abs(u[:, 1, :4]).sum(axis=0))
    # a second, independently "trained" copy compares as a permutation-invariant match
    assert pair_statistics(dicts[1], dicts[1], top_n=4)["match_top"] == 1.0


def test_load_dictionaries_regenerates_the_init_from_the_recorded_seed(tmp_path):
    """A never-trained SAE is its own initialisation: retention must read exactly 1."""
    from layer_dynamics.sae_train import TrainableSAE, initial_decoder

    d, m, seed = 8, 12, 3
    sae = TrainableSAE(d, m, mode="l1", seed=seed)
    assert np.allclose(sae.W_dec.detach().numpy(), initial_decoder(d, m, seed).numpy())
    u = np.ones((2, 2, m), dtype=np.float32)
    feat_index = np.stack([np.full(m, -1), np.arange(m)])
    npz = tmp_path / "sources.npz"
    np.savez(npz, u=u, feat_index=feat_index, alive_count=np.array([0, m]))
    with open(npz.with_suffix(".weights.pkl"), "wb") as fh:
        pickle.dump({"seed": seed, "state_dicts": [
            None, {"W_dec": sae.W_dec.detach().numpy()}]}, fh)
    r = init_retention(load_dictionaries(npz)[1])
    assert r["init_cos_mean_all"] == pytest.approx(1.0) and r["init_frac_all"] == 1.0
