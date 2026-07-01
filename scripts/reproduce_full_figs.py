#!/usr/bin/env python3
"""Reproduce the two figures whose arrays were never persisted to ``tables/``.

Both were reduced by ``replot.py`` because table-only data was insufficient; this
recovers the full-fidelity versions from what *is* on disk, **persists the recovered
arrays** (so ``replot.py`` can redraw them table-only from now on), and rewrites the
SVGs in place:

- **gauge** per-feature cosine histogram — recomputed from the three saved SSM
  checkpoints (seed 0 from ``phase4_checkpoint``, seeds 1/2 in the gauge run).
  No refit: the operators are loaded and the invariants recomputed (~seconds).
- **step1 bootstrap** violin — the within-band spread distributions, recomputed
  deterministically from the cached activations with the exact k / rank / seed the
  original Step-1 run used (~1–2 min of AR refits on CPU).

Usage:  python scripts/reproduce_full_figs.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from layer_dynamics import figures
from layer_dynamics.checkpoint import load_ssm
from layer_dynamics.diagnostics.autoregress import fit_delta_pca
from layer_dynamics.diagnostics.dataset import (
    deltas,
    load_residual_stream,
    train_test_split_idx,
)
from layer_dynamics.diagnostics.decompose_sanity import (
    decoder_directions,
    direction_cosines,
    feature_impulse_responses,
    match_rate,
    top_features,
)
from layer_dynamics.diagnostics.stationarity import (
    BAND_NAMES,
    band_depths,
    bootstrap_within_band,
    fit_band_kernels,
)

ROOT = Path(__file__).resolve().parent.parent
PAIRS = [(0, 1), (0, 2), (1, 2)]


def reproduce_gauge() -> None:
    gdir = ROOT / "runs/2026-06-29-ph4-gauge-410md"
    cdir = ROOT / "runs/2026-06-26-ph4-checkpoint-410md/checkpoints"
    ckpts = {0: cdir / "ssm_seed0.pt",
             1: gdir / "checkpoints/ssm_seed1.pt",
             2: gdir / "checkpoints/ssm_seed2.pt"}
    missing = [str(p) for p in ckpts.values() if not p.exists()]
    if missing:
        print(f"  gauge SKIPPED — missing checkpoints: {missing}")
        return

    feats = top_features(torch.from_numpy(np.load(cdir / "u_infer_full.npy")), top_n=200)
    fidx = [f["feature"] for f in feats]
    flayer = [f["home_layer"] for f in feats]

    inv: dict[int, dict] = {}
    for s, p in ckpts.items():
        m = load_ssm(p, device="cpu")
        raw = decoder_directions(m, fidx, flayer)                 # [k, D] gauge-variant
        imp = feature_impulse_responses(m, fidx, flayer)          # [k, L, D] gauge-invariant
        inv[s] = {"raw": raw, "impulse": imp.reshape(imp.shape[0], -1)}

    raw_cos = np.concatenate([direction_cosines(inv[a]["raw"], inv[b]["raw"]) for a, b in PAIRS])
    imp_cos = np.concatenate([direction_cosines(inv[a]["impulse"], inv[b]["impulse"])
                              for a, b in PAIRS])

    (gdir / "tables").mkdir(exist_ok=True)
    pd.DataFrame({"raw_D_cos": raw_cos, "impulse_cos": imp_cos}).to_csv(
        gdir / "tables/gauge_perfeature_cos.csv", index=False)
    summary = json.loads((gdir / "tables/gauge_stability.json").read_text())["summary"]
    figures.save_svg(figures.gauge_full(summary, raw_cos, imp_cos),
                     gdir / "figures/gauge_stability.svg")
    print(f"  gauge: {len(raw_cos)} per-feature cosines "
          f"(raw match@0.7={match_rate(raw_cos):.2f}, impulse={match_rate(imp_cos):.2f}) "
          f"-> 2-panel SVG + gauge_perfeature_cos.csv")


def reproduce_bootstrap() -> None:
    sdir = ROOT / "runs/2026-06-22-ph2-step1-410md"
    stats = json.loads((sdir / "tables/step1_stats.json").read_text())
    k, r = stats["k"], stats["working_rank"]
    boot_seed, input_pcs, n_boot = 20260622, 256, 200  # config: pythia-410m-deduped.yaml

    rs, _, _ = load_residual_stream(ROOT / "data/pythia-410m-deduped/residual_stream.h5")
    n, n_states, _ = rs.shape
    dlt = deltas(rs)
    tr, te = train_test_split_idx(n, test_frac=0.2, seed=0)
    dlt_tr, dlt_te = dlt[tr], dlt[te]
    basis = fit_delta_pca(dlt_tr, input_pcs)
    bands = band_depths(n_states - 1, k)
    kernels = fit_band_kernels(dlt_tr, dlt_te, k, r, basis, bands)

    spreads = {nm: bootstrap_within_band(dlt_tr, dlt_te, bands[nm], k, r, basis,
                                         kernels[nm].coef_r, n_boot=n_boot, seed=boot_seed + bi)
               for bi, nm in enumerate(BAND_NAMES)}

    (sdir / "tables").mkdir(exist_ok=True)
    pd.DataFrame({nm: spreads[nm] for nm in BAND_NAMES}).to_csv(
        sdir / "tables/bootstrap_samples.csv", index=False)
    boot = pd.read_csv(sdir / "tables/bootstrap.csv")
    between = dict(zip(boot["pair"], boot["between"], strict=False))
    figures.save_svg(figures.bootstrap_violin(spreads, between),
                     sdir / "figures/bootstrap_separation.svg")
    means = ", ".join(f"{nm}={spreads[nm].mean():.3f}" for nm in BAND_NAMES)
    print(f"  bootstrap: within-band spreads recomputed ({means}) "
          f"-> violin SVG + bootstrap_samples.csv")


def main() -> int:
    figures.apply_style()
    print("reproducing gauge per-feature histogram (from 3 saved checkpoints)...")
    reproduce_gauge()
    print("reproducing step1 bootstrap violin (from cached activations)...")
    reproduce_bootstrap()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
