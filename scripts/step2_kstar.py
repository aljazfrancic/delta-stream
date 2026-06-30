#!/usr/bin/env python3
"""Effective kernel depth k* as a function of depth (Phase 2 Step 2).

Step 1 found the AR kernel is depth-indexed (H2 fails), so k* is reported per depth.
For each target depth ℓ we fit single-depth AR(k) over the plan sweep (k ≤ ℓ) at the
Step-1 working rank and read k* = the smallest k reaching 95% of the best held-out
R². A scrambled-null curve (mandatory for any v-vs-Δ analysis — the lag-1 ∓v_ℓ
tautology) must collapse, and the gain of lags ≥ 2 beyond lag 1 is reported so the
lag-1-dominated reconstruction k* is not misread as the model's genuine reach. The
Step 2 memo contrasts this reconstruction k* with Phase 1 Step 5's Granger reach.

    python scripts/step2_kstar.py --config configs/pythia-410m-deduped.yaml \
        --desc ph2-step2-410md \
        --step1-stats runs/2026-06-22-ph2-step1-410md/tables/step1_stats.json \
        --granger-csv runs/2026-06-22-ph1-step5-410md/tables/granger_real.csv

Outputs: tables/{per_depth_kstar,per_depth_r2_curve,band_kstar,step2_stats}.{csv,json},
figures/ (R²(k) per depth real vs null, k*(ℓ) vs Granger reach, R²(depth×k) heatmap).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from layer_dynamics.cli import add_common_args, resolve_out
from layer_dynamics.diagnostics.autoregress import LAG_SWEEP, fit_delta_pca, scramble_per_depth
from layer_dynamics.diagnostics.dataset import deltas, load_residual_stream, train_test_split_idx
from layer_dynamics.diagnostics.effective_depth import (
    band_r2_curve,
    depth_r2_curve,
    incremental_over_lag1,
    kstar_from_curve,
)
from layer_dynamics.diagnostics.stationarity import BAND_NAMES, band_depths, select_k
from layer_dynamics.runlog import init_run


def _class_kstar(dlt_c, *, test_frac, seed, sweep, rank, input_pcs, null_seed, tol, k_step1):
    """Compact per-depth + per-band k* summary on one relation-class subset of Δ.

    Mirrors the pooled per-depth/per-band loops in :func:`main` but returns only the
    summary statistics (medians), so the by-class table stays small. Reuses the same
    AR machinery and scrambled null so each class is computed exactly as the pooled run.
    """
    n = dlt_c.shape[0]
    n_delta = dlt_c.shape[1]
    tr, te = train_test_split_idx(n, test_frac=test_frac, seed=seed)
    dtr, dte = dlt_c[tr], dlt_c[te]
    basis = fit_delta_pca(dtr, input_pcs)
    s_tr, s_te = scramble_per_depth(dtr, null_seed), scramble_per_depth(dte, null_seed + 777)
    kw = {"sweep": sweep, "rank": rank, "basis": basis, "input_pcs": input_pcs}
    kstars, incrs, nulls = [], [], []
    for ell in range(1, n_delta):
        real = depth_r2_curve(dtr, dte, ell, **kw)
        kstars.append(kstar_from_curve(real, tol=tol))
        incrs.append(incremental_over_lag1(real))
        nulls.append(max(depth_r2_curve(s_tr, s_te, ell, **kw).values()))
    bands = band_depths(n_delta, k_step1)
    band_ks = {nm: kstar_from_curve(
        band_r2_curve(dtr, dte, bands[nm], sweep=sweep, rank=rank, basis=basis,
                      input_pcs=input_pcs), tol=tol) for nm in BAND_NAMES}
    return {"n": n, "n_train": len(tr), "median_kstar": float(np.median(kstars)),
            "max_kstar": int(np.max(kstars)),
            "median_incremental_over_lag1": float(np.median(incrs)),
            "median_null_best_r2": float(np.median(nulls)), "band_kstar": band_ks,
            "per_depth_kstar": [int(x) for x in kstars]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--step1-stats", type=Path, default=None,
                        help="Step 1 step1_stats.json — pulls the working rank r and bands.")
    parser.add_argument("--granger-csv", type=Path, default=None,
                        help="Step 5 granger_real.csv — to cite the Granger reach for contrast.")
    parser.add_argument("--rank", type=int, default=None, help="Override the working rank r.")
    parser.add_argument("--tol", type=float, default=0.95, help="k* fraction-of-best threshold.")
    parser.add_argument("--h5", type=Path, default=None,
                        help="Override the residual-stream HDF5 (default data/<model>/…). "
                             "Used by the E7 general-text completion.")
    parser.add_argument("--by-class", action="store_true",
                        help="Also report k* stratified by relation_class (Phase-2 Step-2 "
                             "item 4 / proposal §5.4: effective depth per relation type).")
    args = parser.parse_args(argv)

    run = init_run(resolve_out(args), config_path=args.config, seed=args.seed)
    cfg = run.config
    mcfg = cfg["model"]
    dgcfg = cfg.get("diagnostics", {})
    test_frac = float(dgcfg.get("test_frac", 0.2))
    sweep = tuple(dgcfg.get("ar_lags", LAG_SWEEP))
    input_pcs = int(dgcfg.get("ar_input_pcs", 256))
    null_seed = int(dgcfg.get("ar_null_seed", 1234))

    # Working rank from Step 1 (after its r-vs-2r spot-check), overridable.
    r = args.rank
    if r is None and args.step1_stats is not None and args.step1_stats.exists():
        r = int(json.loads(args.step1_stats.read_text())["working_rank"])
    if r is None:
        r = 256
    run.log(f"working rank r={r}, sweep k={list(sweep)}, tol={args.tol}")

    h5 = args.h5 if args.h5 is not None else Path("data") / mcfg["name"] / "residual_stream.h5"
    rs, relation_class, _ = load_residual_stream(h5)
    n, n_states, _ = rs.shape
    n_delta = n_states - 1
    dlt = deltas(rs)
    tr, te = train_test_split_idx(n, test_frac=test_frac, seed=args.seed)
    dtr, dte = dlt[tr], dlt[te]
    run.log(f"loaded {h5}: deltas {dlt.shape}; split {len(tr)} train / {len(te)} test")

    basis = fit_delta_pca(dtr, input_pcs)  # shared predictor coords (as in Step 1)
    s_tr = scramble_per_depth(dtr, null_seed)        # scrambled null (per-depth perm)
    s_te = scramble_per_depth(dte, null_seed + 777)

    # --- per-depth k* (real + scrambled null) ---
    depths = list(range(1, n_delta))  # ℓ=0 has no lag
    kw = {"sweep": sweep, "rank": r, "basis": basis, "input_pcs": input_pcs}
    curve_rows, kstar_rows = [], []
    for ell in depths:
        real = depth_r2_curve(dtr, dte, ell, **kw)
        null = depth_r2_curve(s_tr, s_te, ell, **kw)
        ks = kstar_from_curve(real, tol=args.tol)
        best_k = max(real, key=real.get)
        best_r2, lag1, incr = max(real.values()), real.get(1, np.nan), incremental_over_lag1(real)
        null_best = max(null.values())
        kstar_rows.append({
            "depth": ell, "kstar": ks, "best_k": best_k, "best_r2": best_r2,
            "r2_lag1": lag1, "incremental_over_lag1": incr,
            "null_best_r2": null_best, "max_feasible_k": max(real),
        })
        for k in real:
            curve_rows.append({"depth": ell, "k": k, "r2_real": real[k], "r2_null": null[k]})
        run.log(f"  ℓ={ell:2d}  k*={ks:<2d} best_k={best_k:<2d} bestR²={best_r2:.3f} "
                f"lag1={lag1:.3f} +lags≥2={incr:+.3f} null={null_best:.3f}")

    # --- per-band k* (Step 1 bands; fixed targets, k ≤ min(band)) ---
    k_step1 = int(json.loads(args.step1_stats.read_text())["k"]) if (
        args.step1_stats and args.step1_stats.exists()) else 5
    bands = band_depths(n_delta, k_step1)
    band_rows = []
    for nm in BAND_NAMES:
        curve = band_r2_curve(dtr, dte, bands[nm], sweep=sweep, rank=r, basis=basis,
                              input_pcs=input_pcs)
        ks = kstar_from_curve(curve, tol=args.tol)
        band_rows.append({"band": nm, "depths": str(bands[nm]), "kstar": ks,
                          "best_k": max(curve, key=curve.get), "best_r2": max(curve.values()),
                          "r2_lag1": curve.get(1, np.nan),
                          "incremental_over_lag1": incremental_over_lag1(curve)})
        run.log(f"  band {nm:6s} k*={ks} bestR²={max(curve.values()):.3f} "
                f"(swept k≤{min(bands[nm])})")

    # --- Granger reach for contrast (Step 5) ---
    granger_reach = None
    if args.granger_csv is not None and args.granger_csv.exists():
        import pandas as pd
        gsel = select_k(pd.read_csv(args.granger_csv), n_delta)
        granger_reach = int(gsel["granger_selected_lag"])

    # --- per-class k* (Phase-2 Step-2 item 4 / proposal §5.4) ---
    by_class = {}
    if args.by_class:
        for c in sorted(set(relation_class.tolist())):
            mask = relation_class == c
            summ = _class_kstar(dlt[mask], test_frac=test_frac, seed=args.seed, sweep=sweep,
                                rank=r, input_pcs=input_pcs, null_seed=null_seed, tol=args.tol,
                                k_step1=k_step1)
            by_class[c] = summ
            run.log(f"  [class {c:14s}] n={summ['n']:4d} median k*={summ['median_kstar']:.0f} "
                    f"max={summ['max_kstar']} bands={summ['band_kstar']} "
                    f"null={summ['median_null_best_r2']:.3f}")

    kstars = [row["kstar"] for row in kstar_rows]
    incrs = [row["incremental_over_lag1"] for row in kstar_rows]
    stats = {
        "model": mcfg["name"], "n_train": len(tr), "n_test": len(te), "n_layers": n_delta,
        "working_rank": r, "sweep": list(sweep), "tol": args.tol,
        "per_depth_kstar_median": float(np.median(kstars)),
        "per_depth_kstar_max": int(np.max(kstars)),
        "per_depth_kstar_values": {row["depth"]: row["kstar"] for row in kstar_rows},
        "median_incremental_over_lag1": float(np.median(incrs)),
        "max_incremental_over_lag1": float(np.max(incrs)),
        "median_null_best_r2": float(np.median([row["null_best_r2"] for row in kstar_rows])),
        "band_kstar": {row["band"]: row["kstar"] for row in band_rows},
        "granger_reach_lag": granger_reach,
        "by_class": by_class,
        "interpretation": (
            "Reconstruction-effective depth k* is tiny (median {:.0f}) at every depth: a "
            "single preceding update captures essentially all held-out predictable "
            "variance of Δ_ℓ, and lags ≥ 2 add a median {:+.3f} R². The scrambled null "
            "collapses the whole curve (median best R² {:.3f}), so the signal is real "
            "example-specific structure. BUT k*≈1 is substantially the mechanical Δ_ℓ ⊃ "
            "−v_ℓ ⊂ Δ_{{ℓ-1}} overlap (cca-vΔ tautology), so it is a property of the "
            "AR-on-Δ parameterisation, NOT evidence the transformer's genuine "
            "computational dependence is one layer deep — Step 5's Granger test finds "
            "incremental significance to lag {}. The gap (reconstruction k*≈1 vs Granger "
            "reach {}) is the headline: long-range structure is statistically real but "
            "reconstruction-negligible. k*≈1 is consistent with the Branch-B SSM being "
            "first-order (Markov in depth)."
        ).format(np.median(kstars), np.median(incrs),
                 float(np.median([row["null_best_r2"] for row in kstar_rows])),
                 granger_reach, granger_reach),
    }

    import pandas as pd

    pd.DataFrame(kstar_rows).to_csv(run.path("tables/per_depth_kstar.csv"), index=False)
    pd.DataFrame(curve_rows).to_csv(run.path("tables/per_depth_r2_curve.csv"), index=False)
    pd.DataFrame(band_rows).to_csv(run.path("tables/band_kstar.csv"), index=False)
    if by_class:
        cls_rows = []
        for c, s in by_class.items():
            row = {"relation_class": c, "n": s["n"], "n_train": s["n_train"],
                   "median_kstar": s["median_kstar"], "max_kstar": s["max_kstar"],
                   "median_incremental_over_lag1": s["median_incremental_over_lag1"],
                   "median_null_best_r2": s["median_null_best_r2"]}
            row.update({f"band_{b}": kv for b, kv in s["band_kstar"].items()})
            cls_rows.append(row)
        pd.DataFrame(cls_rows).to_csv(run.path("tables/kstar_by_class.csv"), index=False)
    run.save_json("tables/step2_stats.json", stats)
    _plots(run, kstar_rows, curve_rows, sweep, granger_reach)

    print(f"\n=== Phase 2 Step 2 — effective depth k* (r={r}) ===")
    print(f"per-depth k*: median {np.median(kstars):.0f}, max {np.max(kstars)} "
          f"(values {kstars})")
    print(f"incremental R² of lags≥2 over lag1: median {np.median(incrs):+.3f}, "
          f"max {np.max(incrs):+.3f}")
    print("band k*: " + ", ".join(f"{row['band']}={row['kstar']}" for row in band_rows))
    print("scrambled-null median best R²: "
          f"{np.median([row['null_best_r2'] for row in kstar_rows]):.3f}")
    print(f"Granger reach (Step 5, for contrast): lag {granger_reach}")
    for c, s in by_class.items():
        print(f"  class {c:14s} (n={s['n']:4d}): median k*={s['median_kstar']:.0f} "
              f"bands={s['band_kstar']} null={s['median_null_best_r2']:.3f}")
    return 0


def _plots(run, kstar_rows, curve_rows, sweep, granger_reach) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    cdf = pd.DataFrame(curve_rows)
    depths = sorted(cdf["depth"].unique())

    # 1. R²(k) per depth — real (coloured by depth) with the scrambled null near 0.
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    cmap = plt.get_cmap("viridis")
    for i, ell in enumerate(depths):
        sub = cdf[cdf["depth"] == ell].sort_values("k")
        ax.plot(sub["k"], sub["r2_real"], "-", color=cmap(i / max(1, len(depths) - 1)),
                lw=0.8, alpha=0.7)
    ax.plot(cdf.groupby("k")["r2_null"].mean().index,
            cdf.groupby("k")["r2_null"].mean().to_numpy(), "k--", lw=1.5,
            label="scrambled null (mean)")
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_xlabel("AR order k"); ax.set_ylabel("held-out R² on Δ_ℓ (single depth)")
    ax.set_title("Reconstruction R²(k) per depth — saturates at k≈1 (real); null≈0")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(min(depths), max(depths)))
    fig.colorbar(sm, ax=ax, label="target depth ℓ"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(run.path("figures/r2_vs_k_per_depth.png"), dpi=130)
    plt.close(fig)

    # 2. k*(ℓ) vs the Granger reach (the reconstruction-vs-significance gap).
    kdf = pd.DataFrame(kstar_rows)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(kdf["depth"], kdf["kstar"], color="tab:blue", label="reconstruction k*")
    ax.plot(kdf["depth"], kdf["max_feasible_k"], "k:", lw=0.8, label="max feasible k (=ℓ-cap)")
    if granger_reach is not None:
        ax.axhline(granger_reach, color="tab:red", lw=1.5, ls="--",
                   label=f"Granger reach (lag {granger_reach})")
    ax.set_xlabel("target depth ℓ"); ax.set_ylabel("k")
    ax.set_title("Effective depth k*(ℓ): reconstruction (≈1) vs Granger significance reach")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(run.path("figures/kstar_vs_depth.png"), dpi=130); plt.close(fig)

    # 3. R²(depth × k) heatmap (real).
    piv = cdf.pivot(index="k", columns="depth", values="r2_real")
    fig, ax = plt.subplots(figsize=(9, 3.2))
    im = ax.imshow(piv.to_numpy(), aspect="auto", origin="lower", cmap="magma",
                   extent=[min(depths) - 0.5, max(depths) + 0.5, -0.5, len(piv.index) - 0.5])
    ax.set_yticks(range(len(piv.index)), [str(k) for k in piv.index])
    ax.set_xlabel("target depth ℓ"); ax.set_ylabel("AR order k")
    ax.set_title("Held-out R²(depth, k) — value concentrated at k=1 across all depths")
    fig.colorbar(im, ax=ax, label="held-out R²"); fig.tight_layout()
    fig.savefig(run.path("figures/r2_depth_k_heatmap.png"), dpi=130); plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
