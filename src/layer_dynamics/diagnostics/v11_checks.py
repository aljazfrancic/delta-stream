"""Verdict logic for the four v1.1 on-box checks (post hoc; paper §8).

Every threshold here is fixed in ``runs/v11-onbox-checks/memo.md``, committed before any
of the checks ran. The functions are pure: they turn numbers a run wrote into the memo's
verdict rows, and they are what the summary script and the tests both call.

  A  the state probe with codes inferred from earlier layers only
  B  the source-prefix sweep with the bias initialised at the per-depth mean
  C  the parameter-matched gap at further seeds, over the L1 dictionary, on the untrained
     network and at 12 layers
  D  AR-kernel transfer between the trained and the untrained network (the verdict for D
     is :func:`layer_dynamics.diagnostics.kernel_transfer.classify_transfer`)
"""

from __future__ import annotations

import math

import numpy as np

# --- A: state probe -----------------------------------------------------------------------
PROBE_PRIMARY = "object/state_causal−prefix_pca_causal/mid"
PROBE_SECONDARY = (
    "object/state_causal−feed_prev_causal/mid",
    "object/state_causal−sae_pca/mid",
    "object/state_causal−resid_pca/mid",
    "object/state−feed_image/mid",
    "object/state−state_causal/mid",
)


def ci_verdict(contrast: dict) -> str:
    """``survives`` if the paired 95% CI lies above 0, ``reversed`` if below, else neither."""
    if contrast["ci_lo"] > 0:
        return "survives"
    if contrast["ci_hi"] < 0:
        return "reversed"
    return "not distinguishable"


def probe_verdict(primary: dict) -> str:
    """Verdict for one dictionary from its primary contrast (state_causal − prefix PCA)."""
    return ci_verdict(primary)


def probe_overall(per_dictionary: dict[str, str]) -> str:
    """The memo's row for check A, from the per-dictionary verdicts."""
    if not per_dictionary:
        return "undetermined"
    n = sum(v == "survives" for v in per_dictionary.values())
    if n == len(per_dictionary):
        return "survives on every dictionary"
    if n == 0:
        return "does not survive"
    return "dictionary-dependent"


def probe_reproduction(new: dict, recorded: dict, arms: list[str], layers: list[int],
                       *, tol: float = 1e-6) -> dict:
    """Do the recorded arms of the new probe reproduce the tracked R7 table?

    Compares per-layer answer-identity and relation accuracy and the confidence ``R²`` of
    ``arms`` at ``layers``, and every band contrast the two tables share.
    """
    diffs: dict[str, float] = {}
    for key in ("object_top1_acc", "relation_acc", "conf_logit_r2"):
        worst = 0.0
        for arm in arms:
            for ell in layers:
                a, b = new[key][arm][ell], recorded[key][arm][ell]
                if a is None or b is None:
                    worst = math.inf
                else:
                    worst = max(worst, abs(float(a) - float(b)))
        diffs[key] = worst
    shared = sorted(set(new.get("band_contrasts", {})) & set(recorded.get("band_contrasts", {})))
    band = 0.0
    for key in shared:
        for field in ("diff", "ci_lo", "ci_hi"):
            band = max(band, abs(new["band_contrasts"][key][field]
                                 - recorded["band_contrasts"][key][field]))
    worst_all = max([*diffs.values(), band])
    return {
        "max_abs_diff_object_top1": diffs["object_top1_acc"],
        "max_abs_diff_relation": diffs["relation_acc"],
        "max_abs_diff_conf_r2": diffs["conf_logit_r2"],
        "max_abs_diff_band_contrasts": band, "n_band_contrasts_compared": len(shared),
        "tol": tol, "ok": bool(worst_all <= tol),
    }


# --- B: source-prefix sweep ---------------------------------------------------------------
PREFIX_FLOOR_TOL = 0.02        # |R²| of the embedding-only point at or below this: at the mean
PREFIX_LATE_GAIN = 0.10        # R²(full) − R²(ℓ_max=19) at or above this: late sources matter
PREFIX_STABLE_TOL = 0.02       # upper-half cutoffs moving less than this: §6.9 stands as is
PREFIX_UPPER_FROM = 15


def prefix_verdict(new_rows: list[dict], recorded_rows: list[dict]) -> dict:
    """The memo's three rows for check B.

    ``new_rows`` / ``recorded_rows`` are the ``rows`` of an ``embed_only.json`` (mean- and
    zero-initialised bias respectively): ``{"cutoff", "train_r2", "test_r2"}`` each.
    """
    new = {int(r["cutoff"]): float(r["test_r2"]) for r in new_rows}
    rec = {int(r["cutoff"]): float(r["test_r2"]) for r in recorded_rows}
    cutoffs = sorted(new)
    full, embed = max(cutoffs), min(cutoffs)
    below_full = [c for c in cutoffs if c < full]
    late_from = max(below_full) if below_full else full
    late_gain = new[full] - new[late_from]
    shared = [c for c in cutoffs if c in rec]
    upper = [c for c in shared if c >= PREFIX_UPPER_FROM]
    lower = [c for c in shared if c < PREFIX_UPPER_FROM]
    upper_move = max((abs(new[c] - rec[c]) for c in upper), default=float("nan"))
    lower_move = max((abs(new[c] - rec[c]) for c in lower), default=float("nan"))
    steps = [new[b] - new[a] for a, b in zip(cutoffs, cutoffs[1:], strict=False)]
    return {
        "embed_only_r2": new[embed], "embed_only_recorded_r2": rec.get(embed),
        "embed_only_at_mean_floor": bool(abs(new[embed]) <= PREFIX_FLOOR_TOL),
        "late_gain": late_gain, "late_gain_from_cutoff": late_from,
        "late_sources_load_bearing": bool(late_gain >= PREFIX_LATE_GAIN),
        "monotone": bool(all(s >= -1e-3 for s in steps)),
        "upper_half_max_move": upper_move, "lower_half_max_move": lower_move,
        "upper_half_stable": bool(upper_move <= PREFIX_STABLE_TOL) if upper else None,
        "test_r2_by_cutoff": {str(c): new[c] for c in cutoffs},
        "recorded_test_r2_by_cutoff": {str(c): rec[c] for c in shared},
    }


# --- C: parameter-matched gap -------------------------------------------------------------
SHARE_ARCHITECTURAL = 0.75     # untrained / trained gap at or above: largely architectural
SHARE_LEARNED = 0.25           # at or below: largely learned
RESOLVABLE_SDS = 3.0           # a gap below this many refit sds is not resolvable


def seeds_verdict(gaps: list[float]) -> dict:
    """Robustness of a gap to refit: every seed positive and ``mean − 2 sd > 0``."""
    g = np.asarray(gaps, dtype=np.float64)
    mean = float(g.mean())
    sd = float(g.std(ddof=1)) if g.size > 1 else float("nan")
    all_pos = bool((g > 0).all())
    robust = bool(all_pos and g.size > 1 and mean - 2 * sd > 0)
    return {"gaps": [float(x) for x in g], "mean": mean, "sd": sd, "all_positive": all_pos,
            "robust": robust,
            "verdict": "robust to refit" if robust else "not robust to refit"}


def paired_gap_verdict(ci: dict) -> str:
    """``holds`` / ``reversed`` / ``not established`` from a paired-bootstrap gap CI."""
    if ci["ci_lo"] > 0:
        return "holds"
    if ci["ci_hi"] < 0:
        return "reversed"
    return "not established"


def learned_share_verdict(trained_gap: float, untrained_gap: float, refit_sd: float) -> dict:
    """How much of the parameter-matched gap an untrained network already has.

    A ratio of two single-fit gaps on two networks; it is not resolvable when the trained
    gap is itself within ``RESOLVABLE_SDS`` refit standard deviations of zero.
    """
    out = {"trained_gap": float(trained_gap), "untrained_gap": float(untrained_gap),
           "refit_sd": float(refit_sd)}
    if not np.isfinite(refit_sd) or trained_gap <= RESOLVABLE_SDS * refit_sd:
        return {**out, "untrained_share": None, "verdict": "not resolvable"}
    share = untrained_gap / trained_gap
    if share >= SHARE_ARCHITECTURAL:
        v = "largely architectural"
    elif share <= SHARE_LEARNED:
        v = "largely learned"
    else:
        v = "about half"
    return {**out, "untrained_share": float(share), "verdict": v}


def depth_verdict(shallow_gaps: list[float], deep_gaps: list[float]) -> dict:
    """Is the parameter-matched gap larger at the deeper model? Two depths, refit sds."""
    a, b = np.asarray(shallow_gaps, float), np.asarray(deep_gaps, float)
    ma, mb = float(a.mean()), float(b.mean())
    sa = float(a.std(ddof=1)) if a.size > 1 else float("nan")
    sb = float(b.std(ddof=1)) if b.size > 1 else float("nan")
    pooled = math.sqrt(sa**2 + sb**2) if np.isfinite(sa) and np.isfinite(sb) else float("nan")
    out = {"shallow_mean": ma, "shallow_sd": sa, "deep_mean": mb, "deep_sd": sb,
           "difference": mb - ma, "pooled_sd": pooled}
    if not np.isfinite(pooled):
        return {**out, "verdict": "undetermined"}
    if mb - ma > 2 * pooled:
        v = "larger at the deeper model"
    elif ma - mb > 2 * pooled:
        v = "larger at the shallower model"
    else:
        v = "no depth trend at matched parameters"
    return {**out, "verdict": v}
