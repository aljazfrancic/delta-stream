#!/usr/bin/env python3
"""Depth-scaling + seed-stability summary figure across models (Phase 2 Step 5 / paper).

Reads the four models' Gate-3 Pareto + Step-4 sanity tables and plots two panels:

  (left)  max SSM-vs-per-layer-baseline reconstruction gap vs depth L — the
          headline depth-scaling result (monotone in the Pythia/mlsae family);
  (right) seed-stability match@0.7 vs depth, marked by SAE source type — the
          cross-model surprise (res-jb L1 stays stable where mlsae TopK does not).

All numbers are pulled from the run JSONs (no hand-entered values) so the figure
regenerates from saved intermediates. Run dirs are resolved by newest glob for the
cross-model lanes; the 70m/410m primary+debug dirs are pinned.

    python scripts/plot_depth_scaling.py            # -> runs/crossmodel/figures/depth_scaling.png
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (label, gate3 dir, step4 dir, SAE suite, family) — cross-model dirs resolved by glob.
MODELS = [
    ("Pythia-70m", "runs/2026-06-24-ph3-gate3-70md", "runs/2026-06-24-ph3-step4-70md",
     "mlsae", "Pythia"),
    ("GPT-2-small", "runs/*-xm-gpt2-gate3", "runs/*-xm-gpt2-step4", "res-jb", "GPT-2"),
    ("Pythia-160m", "runs/*-xm-p160-gate3", "runs/*-xm-p160-step4", "mlsae", "Pythia"),
    ("Pythia-410m", "runs/2026-06-25-ph3-gate3-410md", "runs/2026-06-25-ph3-step4-410md",
     "mlsae", "Pythia"),
]


def _newest(pattern: str) -> str:
    hits = sorted(glob.glob(pattern))
    if not hits:
        raise FileNotFoundError(pattern)
    return hits[-1]


def collect() -> list[dict]:
    rows = []
    for label, gpat, spat, suite, family in MODELS:
        g = json.loads((Path(_newest(gpat)) / "tables" / "gate3_pareto.json").read_text())
        s = json.loads((Path(_newest(spat)) / "tables" / "sanity_stats.json").read_text())
        gaps = [x["gap"] for x in g["matched_l0_gaps"]]
        rows.append({
            "label": label, "suite": suite, "family": family,
            "L": int(g["dims"]["n_layers"]),
            "gap_max": max(gaps), "gap_min": min(gaps),
            "seed": float(s["seed_stability"]["mean_match_rate_0.7"]),
        })
    return sorted(rows, key=lambda r: r["L"])


def main() -> int:
    rows = collect()
    out = Path("runs/crossmodel/figures")
    out.mkdir(parents=True, exist_ok=True)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.6))

    # --- left: depth-scaling of the cross-layer-propagation gap ---
    pyth = [r for r in rows if r["family"] == "Pythia"]
    axL.plot([r["L"] for r in pyth], [r["gap_max"] for r in pyth], "-o", color="tab:blue",
             label="Pythia / mlsae (in-family)", zorder=3)
    for r in rows:
        if r["family"] != "Pythia":
            axL.scatter([r["L"]], [r["gap_max"]], marker="D", s=70, color="tab:orange",
                        zorder=4, label=f"{r['family']} / {r['suite']} (cross-family)")
    for r in rows:
        axL.annotate(f"{r['label']}\n+{r['gap_max']:.3f}", (r["L"], r["gap_max"]),
                     textcoords="offset points", xytext=(6, -2), fontsize=8)
    axL.axhline(0, color="k", lw=0.8)
    axL.set_xlabel("depth L (number of layers)")
    axL.set_ylabel("max SSM − per-layer-baseline R² gap")
    axL.set_title("Cross-layer propagation gain grows with depth")
    axL.legend(fontsize=8, loc="upper left")
    axL.grid(alpha=0.3)

    # --- right: seed-stability vs depth, marked by SAE source type ---
    for suite, color in (("mlsae", "tab:blue"), ("res-jb", "tab:orange")):
        pts = [r for r in rows if r["suite"] == suite]
        axR.scatter([r["L"] for r in pts], [r["seed"] for r in pts], s=70, color=color,
                    label=f"{suite} SAEs")
    for r in rows:
        axR.annotate(f"{r['label']}\n{r['seed']:.2f}", (r["L"], r["seed"]),
                     textcoords="offset points", xytext=(6, -4), fontsize=8)
    axR.axhline(0.7, color="tab:red", ls="--", lw=1, label="0.7 bar")
    axR.set_xlabel("depth L (number of layers)")
    axR.set_ylabel("seed match@0.7 (decoder-direction stability)")
    axR.set_ylim(0, 1.05)
    axR.set_title("Seed-stability tracks SAE source type, not depth")
    axR.legend(fontsize=8, loc="center left")
    axR.grid(alpha=0.3)

    fig.tight_layout()
    path = out / "depth_scaling.png"
    fig.savefig(path, dpi=140)
    print(f"wrote {path}")
    for r in rows:
        print(f"  {r['label']:14s} L={r['L']:2d}  gap +{r['gap_min']:.3f}..+{r['gap_max']:.3f}  "
              f"seed={r['seed']:.2f}  ({r['suite']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
