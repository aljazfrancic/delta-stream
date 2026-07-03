#!/usr/bin/env python3
"""Depth-scaling + seed-stability summary figure across models (Phase 2 Step 5 / paper).

Reads each model's Gate-3 Pareto + Step-4 sanity tables and plots two panels:

  (left)  SSM-vs-per-layer-baseline reconstruction gap (λ=1 knee) vs depth L — the
          headline depth-scaling result (monotone within the Pythia family; NB the
          Pythia line mixes two TopK suites — eleuther-32k at 6L, mlsae at 12/24L —
          and depth is confounded with width/params within the family).
          The Pythia points are **mean ± sd over seeds** (S2): every gate3 run
          matching the model's glob is one seed; seed 0 is the original Gate-3
          fit, seeds 1–2 are added by ``scripts/run_s2_seeds.sh``.
  (right) seed-stability match@0.7 vs depth, grouped by SAE *type* — the
          cross-model result (L1 SAEs stay stable where mlsae TopK does not,
          confirmed on two families each: GPT-2/Qwen L1 vs Pythia TopK).

All numbers are pulled from the run JSONs (no hand-entered values) so the figure
regenerates from saved intermediates. gate3 dirs are globbed across seeds; the
step4 dir (seed-stability, single fit) is the newest glob hit.

    python scripts/plot_depth_scaling.py    # -> runs/crossmodel/figures/depth_scaling.svg
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (label, gate3 glob [matches ALL seeds], step4 glob [single], SAE suite, family, SAE type).
# Qwen2-0.5B's Phase-3 is on its 12-layer SAE-coverage prefix (chanind covers blocks 0..11).
# GPT-2-medium/large have no public residual SAEs (H1/H2-only lanes) so they are not here.
# The Pythia gate3 globs deliberately match the seed-0 dir + the S2 ``-s1``/``-s2`` dirs.
MODELS = [
    # NB (4th-pass review 2026-07-03): the 70m sources are EleutherAI
    # sae-pythia-70m-deduped-32k (TopK k=16), NOT the tim-lawson mlsae suite (x64 k=32)
    # used at 160m/410m — same sparsity *type*, different suite. The in-family Pythia
    # line therefore mixes two TopK suites; the caption must say so.
    ("Pythia-70m", "runs/*-ph3-gate3-70md*", "runs/2026-06-24-ph3-step4-70md",
     "eleuther-32k", "Pythia", "TopK"),
    ("GPT-2-small", "runs/*-xm-gpt2-gate3*", "runs/*-xm-gpt2-step4", "res-jb", "GPT-2", "L1"),
    ("Pythia-160m", "runs/*-xm-p160-gate3*", "runs/*-xm-p160-step4", "mlsae", "Pythia", "TopK"),
    ("Qwen2-0.5B", "runs/*-xm-qwen-gate3*", "runs/*-xm-qwen-step4", "chanind", "Qwen", "L1"),
    ("Pythia-410m", "runs/*-ph3-gate3-410md*", "runs/2026-06-25-ph3-step4-410md",
     "mlsae", "Pythia", "TopK"),
]


def _newest(pattern: str) -> str:
    hits = sorted(glob.glob(pattern))
    if not hits:
        raise FileNotFoundError(pattern)
    return hits[-1]


def _gap_max(gate3_dir: str) -> float | None:
    """Max matched-L0 SSM−baseline gap for one gate3 run, or None if it has no gaps."""
    jf = Path(gate3_dir) / "tables" / "gate3_pareto.json"
    if not jf.exists():
        return None
    g = json.loads(jf.read_text())
    gaps = [x["gap"] for x in g.get("matched_l0_gaps", [])]
    return max(gaps) if gaps else None


KNEE_LAMBDA = 1.0  # the λ_L1 knee (memory: ph3-altmin-tuning); present in every gate3 grid


def _gap_knee(gate3_dir: str, lam: float = KNEE_LAMBDA) -> float | None:
    """Raw SSM−baseline held-out R² difference at the λ knee (review 2026-07-02).

    The robust per-seed statistic. ``gap_max`` interpolates the baseline Pareto at
    matched L0 and takes the max over λ — but at high λ the baseline's L0 *saturates*
    while R² keeps falling, so the Pareto is non-monotone in L0 and the interpolation
    can pair an SSM point with a collapsed baseline point (the 160m seed-0 0.037
    outlier: raw same-λ diff only 0.020). At the knee the curve is dense and the two
    arms' L0 mismatch costs ≲3e-4 R², so the raw same-λ difference is the honest
    matched-sparsity gap; across seeds it is tight (sd ~0.001–0.002) where gap_max
    swings by 0.015.
    """
    jf = Path(gate3_dir) / "tables" / "gate3_pareto.json"
    if not jf.exists():
        return None
    g = json.loads(jf.read_text())
    rows = {(r["variant"], r["lambda_l1"]): r["test_r2"] for r in g.get("rows", [])}
    if ("ssm", lam) not in rows or ("baseline", lam) not in rows:
        return None
    return float(rows[("ssm", lam)] - rows[("baseline", lam)])


def _n_layers(gate3_dir: str) -> int:
    return int(json.loads((Path(gate3_dir) / "tables" / "gate3_pareto.json").read_text())
               ["dims"]["n_layers"])


def collect() -> list[dict]:
    rows = []
    for label, gpat, spat, suite, family, sae_type in MODELS:
        gdirs = sorted(set(glob.glob(gpat)))
        per_seed = [(d, _gap_max(d), _gap_knee(d)) for d in gdirs]
        per_seed = [(d, v, k) for d, v, k in per_seed if v is not None and k is not None]
        if not per_seed:
            raise FileNotFoundError(f"no gate3 runs with gaps for {label}: {gpat}")
        gaps = np.array([v for _, v, _ in per_seed])
        knees = np.array([k for _, _, k in per_seed])
        s = json.loads((Path(_newest(spat)) / "tables" / "sanity_stats.json").read_text())
        rows.append({
            "label": label, "suite": suite, "family": family, "sae_type": sae_type,
            "L": _n_layers(per_seed[0][0]),
            "gap_mean": float(gaps.mean()),
            "gap_sd": float(gaps.std(ddof=1)) if len(gaps) > 1 else 0.0,
            "gap_seeds": [float(x) for x in gaps], "n_seeds": int(len(gaps)),
            "knee_lambda": KNEE_LAMBDA,
            "knee_mean": float(knees.mean()),
            "knee_sd": float(knees.std(ddof=1)) if len(knees) > 1 else 0.0,
            "knee_seeds": [float(x) for x in knees],
            "seed": float(s["seed_stability"]["mean_match_rate_0.7"]),
        })
    return sorted(rows, key=lambda r: r["L"])


def main() -> int:
    rows = collect()
    out = Path("runs/crossmodel")
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    with (out / "tables" / "depth_scaling_gaps.json").open("w") as fh:
        json.dump(rows, fh, indent=2)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.6))

    # --- left: depth-scaling of the cross-layer-propagation gap (Pythia = mean±sd over
    # seeds), using the robust λ-knee statistic (gap_max kept in the JSON for the record —
    # its matched-L0 interpolation is fragile on the saturated Pareto tail; review 2026-07-02).
    pyth = [r for r in rows if r["family"] == "Pythia"]
    axL.errorbar([r["L"] for r in pyth], [r["knee_mean"] for r in pyth],
                 yerr=[r["knee_sd"] for r in pyth], fmt="-o", color="tab:blue", capsize=4,
                 label="Pythia / TopK suites (in-family, mean±sd)", zorder=3)
    xfam_color = {"GPT-2": "tab:orange", "Qwen": "tab:green"}
    for r in rows:
        if r["family"] != "Pythia":
            axL.scatter([r["L"]], [r["knee_mean"]], marker="D", s=70,
                        color=xfam_color.get(r["family"], "tab:orange"),
                        zorder=4, label=f"{r['family']} / {r['suite']} (cross-family)")
    xmax = max(r["L"] for r in rows)
    for r in rows:
        tag = (f"+{r['knee_mean']:.3f}±{r['knee_sd']:.3f}" if r["n_seeds"] > 1
               else f"+{r['knee_mean']:.3f}")
        # Qwen's Phase-3 runs on the 12L SAE-coverage prefix of a 24-layer model — its
        # x-position is modeled-prefix length, not model depth (4th-pass review).
        if r["family"] == "Qwen":
            tag += "\n(12L prefix of a 24L model)"
        left = r["L"] == xmax  # rightmost point: label to the left or it clips outside
        axL.annotate(f"{r['label']}\n{tag}", (r["L"], r["knee_mean"]),
                     textcoords="offset points", xytext=(-6, -2) if left else (6, -2),
                     ha="right" if left else "left", fontsize=8)
    axL.axhline(0, color="k", lw=0.8)
    axL.set_xlabel("depth L (number of layers)")
    axL.set_ylabel(f"SSM − per-layer-baseline R² gap at λ_L1={KNEE_LAMBDA:g}")
    # Honest title (4th-pass review): the monotone trend is within-Pythia, where depth is
    # confounded with width/params; cross-family points at 12L span 0.013–0.041.
    axL.set_title("Cross-layer propagation gain vs depth (monotone within Pythia)")
    # lower right is the one empty region (Qwen occupies the upper left at (12, ~0.044));
    # anchored a little up so the box clears the y=0 line
    axL.legend(fontsize=8, loc="lower right", bbox_to_anchor=(0.98, 0.10))
    axL.grid(alpha=0.3)

    # --- right: seed-stability vs depth, grouped by SAE *type* (the actual lever) ---
    for sae_type, color in (("TopK", "tab:blue"), ("L1", "tab:orange")):
        pts = [r for r in rows if r["sae_type"] == sae_type]
        axR.scatter([r["L"] for r in pts], [r["seed"] for r in pts], s=70, color=color,
                    label=f"{sae_type} SAEs")
    for r in rows:
        axR.annotate(f"{r['label']}\n{r['seed']:.2f}", (r["L"], r["seed"]),
                     textcoords="offset points",
                     xytext=(-6, -4) if r["L"] == xmax else (6, -4),
                     ha="right" if r["L"] == xmax else "left", fontsize=8)
    axR.axhline(0.7, color="tab:red", ls="--", lw=1, label="0.7 bar")
    axR.set_xlabel("depth L (number of layers)")
    axR.set_ylabel("seed match@0.7 (decoder-direction stability)")
    axR.set_ylim(0, 1.05)
    axR.set_title("Seed-stability tracks SAE source type, not depth")
    axR.legend(fontsize=8, loc="center left")
    axR.grid(alpha=0.3)

    fig.tight_layout()
    path = out / "figures" / "depth_scaling.svg"
    fig.savefig(path, bbox_inches="tight")
    print(f"wrote {path}")
    for r in rows:
        tag = f"±{r['gap_sd']:.3f} (n={r['n_seeds']})" if r["n_seeds"] > 1 else "(n=1)"
        print(f"  {r['label']:14s} L={r['L']:2d}  gap_max +{r['gap_mean']:.3f}{tag}  "
              f"seed-stab={r['seed']:.2f}  ({r['suite']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
