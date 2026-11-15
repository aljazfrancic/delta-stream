"""Dictionary-level seed stability of the trained per-layer SAEs (v1.1, post-hoc E1).

**What this measures, and what it does not.** The Gate-3 seed-stability statistic
(:mod:`layer_dynamics.diagnostics.decompose_sanity`) re-seeds the *SSM*: it re-fits the
state-space model from different initialisations on top of **one** frozen dictionary and
asks whether feature ``j``'s learned write direction ``D_ℓ[:, j]`` stays put, pairing
features by index. It never trains a second dictionary. This module answers the other
question — the one "re-seed" usually means in the SAE literature: when two SAEs are
trained on the **same** activations from different random initialisations, do they
recover the same decoder directions?

Latents permute freely between independently trained dictionaries, so here there *is* a
permutation to search over. Per layer and per seed pair, on **alive** latents only:

* **best-match** — each latent's maximum cosine against every alive latent of the other
  seed (many-to-one allowed), computed in both directions and averaged;
* **one-to-one (PW-MCC)** — the optimal assignment (Hungarian) between the two alive
  sets on **absolute** decoder cosines, mean over the matched pairs. This is the
  pairwise dictionary mean correlation coefficient of Song et al. (ACL 2026), restricted
  to alive latents (``n = min`` of the two alive counts), reported for comparability
  with that literature.

Best-match cosines are **signed**: SAE activations are non-negative, so ``d`` and ``−d``
are different features. PW-MCC takes absolute values because its definition does.

**Primary statistic (fixed before any run).** ``match@0.7`` over the top-100 latents by
activation mass per layer, best-match, averaged over layers and then over seed pairs.

**Nulls.** A statistic that random inputs also satisfy is worthless (this repo's
sorted-modulus cosine of 0.999 was one), so every statistic is also computed against

* a random isotropic unit-norm dictionary of matched size (``null_random``);
* the *other seed's* dictionary at a distant layer (``null_xlayer_cross_seed``): a
  different initialisation *and* the wrong layer, so any match is layer-generic
  structure rather than that layer's features.

**Initialisation diagnostics (not nulls).** :func:`layer_dynamics.sae_train.initial_decoder`
gives every layer's SAE the same starting matrix for a given seed. Two numbers say how
much of a trained dictionary is still that matrix: the *same seed's* dictionary at the
distant layer (``diag_xlayer_same_seed`` — a shared initialisation, different data; a
synthetic smoke test scored 1.00 here for latents that had barely moved, which is why
this is a diagnostic and not a null), and the direct cosine of each alive latent with
its own initial column (:func:`init_retention`). High values mean the dictionary is
initialisation-dominated, which bounds what any cross-seed agreement could be.

All functions are pure; loading and orchestration live in
``scripts/sae_seed_stability.py``.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

STAT_KEYS = (
    "match_top", "match_all", "mean_maxcos_top", "mean_maxcos_all", "pw_mcc",
)
PRIMARY = "match_top"
STABLE_BAR = 0.7

# The verdict table, frozen in runs/v11-seed-stability/memo.md before the runs.
VERDICTS = {
    (True, False): "L1 stable / TopK unstable: the warning holds at both levels in this regime",
    (False, False): "neither mechanism gives reproducible dictionaries here",
    (True, True): "TopK dictionaries reproduce; the instability is in the per-feature readout",
    (False, True): "L1 unstable / TopK stable: suspect a bug first",
}


@dataclass
class LayerDictionary:
    """One layer's alive decoder directions, with the mass used to rank them."""

    layer: int                 # source-state index ℓ' (the SAE reads v_ℓ')
    directions: np.ndarray     # [n_alive, D] unit-norm rows
    mass: np.ndarray           # [n_alive]  Σ_i |u_{i,ℓ',j}| over examples
    latent_index: np.ndarray   # [n_alive]  global latent id of each row
    init_cos: np.ndarray | None = None  # [n_alive] cosine with the latent's own init column

    @property
    def n_alive(self) -> int:
        return int(self.directions.shape[0])


def unit_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Rows of ``x`` scaled to unit Euclidean norm (float64)."""
    x = np.asarray(x, dtype=np.float64)
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), eps, None)


def layer_dictionary(
    w_dec: np.ndarray, feat_index: np.ndarray, alive_count: int, u_layer: np.ndarray,
    layer: int, *, w_init: np.ndarray | None = None,
) -> LayerDictionary:
    """Build a :class:`LayerDictionary` from one layer of a trained-source artefact.

    ``w_dec[D, M]`` is the full decoder; ``feat_index[M*]`` maps each alive-compressed
    slot to its global latent id (``-1`` = padding); ``u_layer[N, M*]`` are the codes in
    the same alive-compressed order. ``w_init[D, M]``, when given, is the decoder the SAE
    started from; each alive latent's cosine with its own initial column is kept.
    """
    n = int(alive_count)
    ids = np.asarray(feat_index[:n], dtype=np.int64)
    if n and (ids.min() < 0 or ids.max() >= w_dec.shape[1]):
        raise ValueError(f"layer {layer}: feat_index outside [0, {w_dec.shape[1]})")
    directions = unit_rows(np.asarray(w_dec)[:, ids].T)
    init_cos = None
    if w_init is not None:
        init_cos = (directions * unit_rows(np.asarray(w_init)[:, ids].T)).sum(axis=1)
    return LayerDictionary(
        layer=int(layer), directions=directions,
        mass=np.abs(np.asarray(u_layer[:, :n], dtype=np.float64)).sum(axis=0),
        latent_index=ids, init_cos=init_cos,
    )


def load_dictionaries(
    npz_path: str | Path, weights_path: str | Path | None = None,
) -> dict[int, LayerDictionary]:
    """Load every trained layer of a ``train_410m_saes.py`` artefact pair.

    ``npz_path`` is the cached sources file (``u``, ``feat_index``, ``alive_count``);
    ``weights_path`` defaults to the sibling ``.weights.pkl`` the trainer writes. Layers
    without an SAE (``v_0``) or without alive latents are skipped. Only the alive decoder
    columns are kept, so the 1.5 GB pickle is not retained. When the pickle records its
    training ``seed`` (v1.1 onwards) the initial decoder is regenerated and each latent's
    cosine with its own initial column is attached (:func:`init_retention`).
    """
    npz_path = Path(npz_path)
    weights_path = Path(weights_path) if weights_path else npz_path.with_suffix(".weights.pkl")
    with np.load(npz_path) as z:
        u, feat_index, alive_count = z["u"], z["feat_index"], z["alive_count"]
    with open(weights_path, "rb") as fh:  # our own artefact, written by train_410m_saes.py
        blob = pickle.load(fh)
    state_dicts = blob["state_dicts"]
    if len(state_dicts) != u.shape[1]:
        raise ValueError(f"{weights_path}: {len(state_dicts)} layers vs {u.shape[1]} in the npz")
    w_init = None
    trained = [sd for sd in state_dicts if sd is not None]
    if blob.get("seed") is not None and trained:
        from layer_dynamics.sae_train import initial_decoder

        w_init = initial_decoder(*trained[0]["W_dec"].shape, int(blob["seed"])).numpy()
    out: dict[int, LayerDictionary] = {}
    for layer, sd in enumerate(state_dicts):
        if sd is None or int(alive_count[layer]) == 0:
            continue
        out[layer] = layer_dictionary(sd["W_dec"], feat_index[layer], alive_count[layer],
                                      u[:, layer], layer, w_init=w_init)
    return out


def init_retention(d: LayerDictionary, *, top_n: int = 100, threshold: float = 0.7) -> dict:
    """How much of a trained layer dictionary is still its random initialisation?

    Mean cosine between each alive latent and its own initial decoder column, and the
    fraction at or above ``threshold``, over all alive latents and over the top-``top_n``
    by mass. ``{}`` when the initialisation is unknown (pre-v1.1 artefacts carry no seed).
    """
    if d.init_cos is None:
        return {}
    top = d.init_cos[top_by_mass(d.mass, top_n)]
    return {
        "init_cos_mean_all": float(d.init_cos.mean()),
        "init_frac_all": float((d.init_cos >= threshold).mean()),
        "init_cos_mean_top": float(top.mean()),
        "init_frac_top": float((top >= threshold).mean()),
    }


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Signed cosine between every row of ``a[n_a, D]`` and every row of ``b[n_b, D]``."""
    return unit_rows(a) @ unit_rows(b).T


def top_by_mass(mass: np.ndarray, top_n: int) -> np.ndarray:
    """Indices of the ``top_n`` largest-mass latents (all of them if fewer exist)."""
    order = np.argsort(-np.asarray(mass), kind="stable")
    return order[: min(int(top_n), len(order))]


def pw_mcc(cos: np.ndarray) -> tuple[float, int]:
    """PW-MCC: mean **absolute** cosine under the optimal one-to-one matching, and ``n``.

    The matching maximises the summed absolute cosine (Song et al.'s definition).
    Rectangular inputs match ``n = min(n_a, n_b)`` pairs; the surplus latents of the
    larger dictionary are left unmatched (and are visible in the alive counts).
    """
    if cos.size == 0:
        return 0.0, 0
    absolute = np.abs(cos)
    rows, cols = linear_sum_assignment(absolute, maximize=True)
    return float(absolute[rows, cols].mean()), int(len(rows))


def pair_statistics(
    da: LayerDictionary, db: LayerDictionary, *, top_n: int = 100, threshold: float = 0.7,
) -> dict[str, float]:
    """All E1 statistics for one (dictionary, dictionary) comparison.

    Best-match statistics are computed in both directions (``a``'s latents against all
    of ``b``, and ``b``'s against all of ``a``) and the two directions averaged with equal
    weight. ``*_top`` restricts the *querying* side to its top-``top_n`` latents by mass;
    the side being searched is always the full alive set.
    """
    cos = cosine_matrix(da.directions, db.directions)
    a_best, b_best = cos.max(axis=1), cos.max(axis=0)
    a_top, b_top = a_best[top_by_mass(da.mass, top_n)], b_best[top_by_mass(db.mass, top_n)]
    mcc, mcc_n = pw_mcc(cos)

    def both(x: np.ndarray, y: np.ndarray, fn) -> float:
        return float(0.5 * (fn(x) + fn(y)))

    def rate(x: np.ndarray) -> float:
        return float((x >= threshold).mean())

    return {
        "n_alive_a": da.n_alive, "n_alive_b": db.n_alive,
        "n_top_a": int(len(a_top)), "n_top_b": int(len(b_top)),
        "match_top": both(a_top, b_top, rate),
        "match_all": both(a_best, b_best, rate),
        "mean_maxcos_top": both(a_top, b_top, np.mean),
        "mean_maxcos_all": both(a_best, b_best, np.mean),
        "pw_mcc": mcc, "pw_mcc_n": mcc_n,
    }


def random_dictionary_like(d: LayerDictionary, rng: np.random.Generator) -> LayerDictionary:
    """An isotropic random unit-norm dictionary with ``d``'s size, ambient dim and masses."""
    g = rng.standard_normal(d.directions.shape)
    return LayerDictionary(layer=d.layer, directions=unit_rows(g), mass=d.mass.copy(),
                           latent_index=d.latent_index.copy(), init_cos=None)


def distant_layer(layer: int, layers: list[int]) -> int:
    """The layer half the stack away from ``layer`` on the ring of trained layers."""
    ordered = sorted(layers)
    if len(ordered) < 2:
        raise ValueError("need at least two trained layers for a cross-layer null")
    return ordered[(ordered.index(layer) + len(ordered) // 2) % len(ordered)]


def compare_pair(
    dicts_a: dict[int, LayerDictionary], dicts_b: dict[int, LayerDictionary], *,
    top_n: int = 100, threshold: float = 0.7, rng: np.random.Generator | None = None,
) -> list[dict]:
    """Per-layer rows for one seed pair: the real comparison, two nulls, one diagnostic.

    Each row carries ``kind`` ∈ {``real``, ``null_random``, ``null_xlayer_cross_seed``,
    ``diag_xlayer_same_seed``}, the two layers compared, and :func:`pair_statistics`.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    layers = sorted(set(dicts_a) & set(dicts_b))
    rows: list[dict] = []
    for layer in layers:
        a, b = dicts_a[layer], dicts_b[layer]
        comparisons = [("real", layer, a, b),
                       ("null_random", layer, a, random_dictionary_like(b, rng))]
        if len(layers) >= 2:
            far = distant_layer(layer, layers)
            comparisons += [("null_xlayer_cross_seed", far, a, dicts_b[far]),
                            ("diag_xlayer_same_seed", far, a, dicts_a[far])]
        for kind, layer_b, x, y in comparisons:
            rows.append({"kind": kind, "layer": layer, "layer_b": layer_b,
                         **pair_statistics(x, y, top_n=top_n, threshold=threshold)})
    return rows


def summarise(rows: list[dict]) -> dict[str, dict[str, float]]:
    """Mean of every statistic per ``kind``: over layers within a pair, then over pairs.

    Rows must carry ``pair`` (any hashable label), ``kind`` and the ``STAT_KEYS``.
    Averaging layers first gives each seed pair equal weight.
    """
    out: dict[str, dict[str, float]] = {}
    for kind in sorted({r["kind"] for r in rows}):
        per_pair = []
        for pair in sorted({r["pair"] for r in rows if r["kind"] == kind}):
            sel = [r for r in rows if r["kind"] == kind and r["pair"] == pair]
            per_pair.append({k: float(np.mean([r[k] for r in sel])) for k in STAT_KEYS})
        out[kind] = {k: float(np.mean([p[k] for p in per_pair])) for k in STAT_KEYS}
        out[kind]["n_pairs"] = len(per_pair)
    return out


def null_status(summary: dict[str, dict[str, float]], *, bar: float = STABLE_BAR,
                margin: float = 0.2) -> dict:
    """Is the primary statistic informative, i.e. clear of every null?

    Uninformative — no stable/unstable verdict is drawn for that mechanism — if any null
    reaches ``bar`` (the statistic is then vacuous), or if the real value clears ``bar``
    by less than ``margin`` over the worst null (a "stable" reading the nulls nearly
    reproduce). A real value below ``bar`` needs no margin: sitting at null level is what
    "unstable" means.
    """
    real = summary["real"][PRIMARY]
    nulls = {k: v[PRIMARY] for k, v in summary.items() if k.startswith("null_")}
    worst = max(nulls.values(), default=0.0)
    informative = worst < bar
    if real >= bar:
        informative = informative and (real - worst) >= margin
    return {"real": real, "nulls": nulls, "worst_null": worst, "informative": bool(informative)}


def verdict(l1_primary: float, topk_primary: float, *, bar: float = STABLE_BAR) -> dict:
    """Map the two primary statistics onto the pre-registered verdict row."""
    key = (bool(l1_primary >= bar), bool(topk_primary >= bar))
    return {"l1_stable": key[0], "topk_stable": key[1], "bar": bar, "row": VERDICTS[key]}


def reproduction_check(
    alive_count: np.ndarray, recon_r2: np.ndarray, reference: dict | str | Path, *,
    alive_rtol: float = 0.05, mstar_rtol: float = 0.02, r2_atol: float = 0.003,
) -> dict:
    """Does a retrained dictionary reproduce a tracked ``sources_trained_*.json``?

    Compares per-layer alive counts (relative), ``M*`` (relative) and the mean
    reconstruction R² (absolute) against the recorded run's table. Used to accept a
    seed-0 retrain — and, with it, the activation store and hyperparameters — before any
    cross-seed number is trusted.
    """
    if not isinstance(reference, dict):
        with open(reference, encoding="utf-8") as fh:
            reference = json.load(fh)
    ref_alive = np.asarray(reference["alive_count"], dtype=np.float64)
    ref_r2 = np.asarray([np.nan if x is None else x for x in reference["recon_r2"]],
                        dtype=np.float64)
    alive = np.asarray(alive_count, dtype=np.float64)
    r2 = np.asarray(recon_r2, dtype=np.float64)
    if alive.shape != ref_alive.shape:
        return {"ok": False, "reason": f"layer count {alive.shape} vs reference {ref_alive.shape}"}
    trained = ref_alive > 0
    alive_rel = np.abs(alive[trained] - ref_alive[trained]) / ref_alive[trained]
    mstar, ref_mstar = float(alive.max()), float(reference["M_star"])
    r2_mean, ref_r2_mean = float(np.nanmean(r2)), float(np.nanmean(ref_r2))
    res = {
        "M_star": int(mstar), "ref_M_star": int(ref_mstar),
        "mstar_rel_diff": abs(mstar - ref_mstar) / ref_mstar,
        "max_alive_rel_diff": float(alive_rel.max()),
        "alive_identical": bool(np.array_equal(alive, ref_alive)),
        "recon_r2_mean": r2_mean, "ref_recon_r2_mean": ref_r2_mean,
        "r2_mean_abs_diff": abs(r2_mean - ref_r2_mean),
        "max_r2_abs_diff": float(np.nanmax(np.abs(r2 - ref_r2))),
        "tolerances": {"alive_rtol": alive_rtol, "mstar_rtol": mstar_rtol, "r2_atol": r2_atol},
    }
    res["ok"] = bool(res["mstar_rel_diff"] <= mstar_rtol
                     and res["max_alive_rel_diff"] <= alive_rtol
                     and res["r2_mean_abs_diff"] <= r2_atol)
    return res
