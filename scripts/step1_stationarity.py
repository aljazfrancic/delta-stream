#!/usr/bin/env python3
"""Across-depth stationarity of the AR(k) kernel (Phase 2 Step 1).

Fits one band-shared low-rank AR(k) kernel on each of the early / middle / late
depth bands (same form as Phase 1 Step 5, restricted to the band's target depths,
against a **shared input PCA basis** so the kernels are comparable) and compares the
three with the plan's four diagnostics — per-lag & stacked Frobenius distance,
principal angles, the 3×3 cross-band prediction confusion, and a within-band
bootstrap — then applies the decision matrix (stationary → A / smooth-drift →
depth-smoothed B / pervasive-drift → fully-depth-indexed B / phase-transition → C).

``k`` is selected from the Phase 1 Step 5 Granger profile under the early-band
feasibility cap ⌊L/3⌋−3; ``r`` is the Step 3 ridge saturation rank (median across
layers). A pooled AR(k) spot-check at r vs 2r promotes r→2r if 2r is materially
better (>5% absolute held-out R²), per the plan.

    python scripts/step1_stationarity.py --config configs/pythia-410m-deduped.yaml \
        --desc ph2-step1-410md \
        --step3-csv runs/2026-06-21-ph1-step3-410md/tables/ridge_r2.csv \
        --granger-csv runs/2026-06-22-ph1-step5-410md/tables/granger_real.csv

Outputs: tables/{cross_band_r2,per_lag_frobenius,bootstrap,step1_stats}.{csv,json},
figures/ (cross-band confusion, per-lag Frobenius, bootstrap separation, angles).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from layer_dynamics.cli import add_common_args, resolve_out
from layer_dynamics.diagnostics.autoregress import LAG_SWEEP, fit_ar_pooled, fit_delta_pca
from layer_dynamics.diagnostics.dataset import deltas, load_residual_stream, train_test_split_idx
from layer_dynamics.diagnostics.stationarity import (
    BAND_NAMES,
    band_depths,
    bootstrap_within_band,
    classify_stationarity,
    cross_band_r2,
    cross_band_r2_by_rank,
    fit_band_kernels,
    per_lag_frobenius,
    principal_angles_deg,
    relative_frobenius,
    select_k,
    top_subspaces,
)
from layer_dynamics.runlog import init_run

PAIRS = (("early", "middle"), ("middle", "late"), ("early", "late"))


def saturation_rank(step3_csv: Path, ranks: tuple[int, ...], *, tol: float = 0.05) -> dict:
    """Step 3 ridge saturation rank: smallest rank within ``tol`` of the largest-swept R².

    Per layer, the asymptote is the R² at the largest swept rank; the saturation rank
    is the smallest rank whose R² is within ``tol`` (relative) of it. ``r`` is the
    median across layers (plan: "if layer-dependent, take the median").
    """
    import pandas as pd

    df = pd.read_csv(step3_csv)
    top = max(ranks)
    per_layer = []
    for _, row in df.iterrows():
        asymptote = row[f"r{top}"]
        sat = top
        for r in ranks:
            if asymptote > 0 and row[f"r{r}"] >= (1.0 - tol) * asymptote:
                sat = r
                break
        per_layer.append(sat)
    return {"per_layer": per_layer, "median_rank": int(np.median(per_layer))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--step3-csv", type=Path, default=None,
                        help="Step 3 ridge_r2.csv — locks the saturation rank r "
                             "(omit + pass --rank when no Step-3 run exists, e.g. E7).")
    parser.add_argument("--granger-csv", type=Path, default=None,
                        help="Step 5 granger_real.csv — used to select the AR order k "
                             "(omit and pass --k to set the order directly).")
    parser.add_argument("--k", type=int, default=None, help="Override / set the AR order.")
    parser.add_argument("--rank", type=int, default=None, help="Override / set the rank r.")
    parser.add_argument("--h5", type=Path, default=None,
                        help="Override the residual-stream HDF5 (default data/<model>/…). "
                             "Used by the E7 general-text completion.")
    args = parser.parse_args(argv)

    run = init_run(resolve_out(args), config_path=args.config, seed=args.seed)
    cfg = run.config
    mcfg = cfg["model"]
    dgcfg = cfg.get("diagnostics", {})
    ranks = tuple(dgcfg.get("ranks", [16, 32, 64, 128]))
    test_frac = float(dgcfg.get("test_frac", 0.2))
    sweep = tuple(dgcfg.get("ar_lags", LAG_SWEEP))
    input_pcs = int(dgcfg.get("ar_input_pcs", 256))
    n_boot = int(dgcfg.get("stationarity_n_boot", 200))
    boot_seed = int(dgcfg.get("stationarity_boot_seed", 20260622))

    import pandas as pd

    # --- data + split (shared with Phase 1) ---
    h5 = args.h5 if args.h5 is not None else Path("data") / mcfg["name"] / "residual_stream.h5"
    rs, _, _ = load_residual_stream(h5)
    n, n_states, d = rs.shape
    n_delta = n_states - 1
    dlt = deltas(rs)
    tr, te = train_test_split_idx(n, test_frac=test_frac, seed=args.seed)
    dlt_tr, dlt_te = dlt[tr], dlt[te]
    run.log(f"loaded {h5}: deltas {dlt.shape}; split {len(tr)} train / {len(te)} test")

    # --- lock k (Granger profile + feasibility cap) and r (Step 3 saturation) ---
    # CSVs auto-select k/r; when absent (e.g. E7 general text has no paired Step-3 run),
    # --k/--rank must be supplied directly.
    if args.granger_csv is not None:
        ksel = select_k(pd.read_csv(args.granger_csv), n_delta, sweep=sweep)
        k = args.k if args.k is not None else ksel["k"]
        run.log(f"k selection: Granger-significant to lag {ksel['granger_selected_lag']}, "
                f"cap {ksel['feasibility_cap']} -> k={k}"
                + (" (CAPPED: short-horizon only)" if ksel["capped"] else ""))
    else:
        ksel = None
        if args.k is None:
            parser.error("pass --granger-csv or --k to set the AR order")
        k = args.k
        run.log(f"k set directly to {k} (no --granger-csv)")
    if args.step3_csv is not None:
        sat = saturation_rank(args.step3_csv, ranks)
        r = args.rank if args.rank is not None else sat["median_rank"]
        run.log(f"rank: Step 3 saturation median r={sat['median_rank']} "
                f"(per-layer {sat['per_layer']}) -> r={r}")
    else:
        sat = None
        if args.rank is None:
            parser.error("pass --step3-csv or --rank to set the rank")
        r = args.rank
        run.log(f"rank set directly to {r} (no --step3-csv)")

    bands = band_depths(n_delta, k)
    run.log("bands (target depths): " + ", ".join(f"{nm}={bands[nm]}" for nm in BAND_NAMES))
    for nm in BAND_NAMES:
        if len(bands[nm]) < 1:
            raise SystemExit(f"band {nm} has no feasible depths at k={k}")

    # Shared input PCA basis (global Δ), so all band kernels share one coordinate system.
    basis = fit_delta_pca(dlt_tr, input_pcs)

    # --- spot-check: pooled AR(k) at r vs 2r (promote r if 2r materially better) ---
    spot = fit_ar_pooled(dlt_tr, dlt_te, k, ranks=(r, 2 * r), input_basis=basis)
    gain = spot.r2[2 * r] - spot.r2[r]
    working_r = 2 * r if gain > 0.05 else r
    run.log(f"spot-check pooled AR(k={k}): R²(r={r})={spot.r2[r]:.4f}, "
            f"R²(2r={2 * r})={spot.r2[2 * r]:.4f}, gain={gain:.4f} -> working r={working_r}")

    # --- fit the three band kernels + their cross-band confusion ---
    kernels = fit_band_kernels(dlt_tr, dlt_te, k, working_r, basis, bands)
    for nm in BAND_NAMES:
        run.log(f"  band {nm:6s} within-R²={kernels[nm].within_r2:.4f} "
                f"(depths {kernels[nm].depths}, α={kernels[nm].alpha:.1f})")
    cross = cross_band_r2(kernels, dlt_tr, dlt_te, k, basis)

    # --- rank-robustness: cross-band confusion across ranks (cheap, no bootstrap) ---
    # Guards the gate verdict against the spot-check's rank promotion: if the
    # off-diagonals stay negative from a small rank up, the (non-)stationarity is real.
    rank_sweep = sorted({r for r in (32, 64, 128, 256, 512) if r <= working_r} | {working_r})
    cross_by_rank = cross_band_r2_by_rank(dlt_tr, dlt_te, k, tuple(rank_sweep), basis, bands)
    run.log("rank-robustness (worst off-diagonal cross-band R² by rank): " + ", ".join(
        f"r{r}={min(cross_by_rank[r][f][e] for f in BAND_NAMES for e in BAND_NAMES if f != e):.3f}"
        for r in rank_sweep))

    # --- Frobenius + principal angles between every band pair ---
    between_fro_stack, between_fro_perlag, between_angles = {}, {}, {}
    for a, b in PAIRS:
        ca, cb = kernels[a].coef_r, kernels[b].coef_r
        key = f"{a}-{b}"
        between_fro_stack[key] = relative_frobenius(ca, cb)
        between_fro_perlag[key] = per_lag_frobenius(ca, cb, k)
        ua, va = top_subspaces(ca, working_r)
        ub, vb = top_subspaces(cb, working_r)
        between_angles[key] = {
            "output": _ang(principal_angles_deg(va, vb)),
            "input": _ang(principal_angles_deg(ua, ub)),
        }
        run.log(f"  {key:14s} ‖ΔW‖={between_fro_stack[key]:.3f}  "
                f"out∠(mean/max)={between_angles[key]['output']['mean_deg']:.1f}/"
                f"{between_angles[key]['output']['max_deg']:.1f}°")

    # --- bootstrap: within-band spread vs between-band distance ---
    run.log(f"bootstrap within-band kernels (n_boot={n_boot}) ...")
    spreads = {}
    for bi, nm in enumerate(BAND_NAMES):
        spreads[nm] = bootstrap_within_band(
            dlt_tr, dlt_te, bands[nm], k, working_r, basis, kernels[nm].coef_r,
            n_boot=n_boot, seed=boot_seed + bi,  # deterministic per-band offset
        )
        run.log(f"  band {nm:6s} within-spread mean={spreads[nm].mean():.3f} "
                f"p95={np.percentile(spreads[nm], 95):.3f}")
    bootstrap = {}
    for a, b in PAIRS:
        key = f"{a}-{b}"
        between = between_fro_stack[key]
        wa95, wb95 = np.percentile(spreads[a], 95), np.percentile(spreads[b], 95)
        pooled_std = 0.5 * (spreads[a].std() + spreads[b].std())
        bootstrap[key] = {
            "between": float(between),
            "within_a_p95": float(wa95), "within_b_p95": float(wb95),
            "z_between_over_within_sd": float(between / pooled_std) if pooled_std > 0 else np.inf,
            # Separated = the between-band distance exceeds the summed within-band 95th
            # percentiles (the two bootstrap clouds do not plausibly overlap).
            "separated": bool(between > wa95 + wb95),
        }
        run.log(f"  {key:14s} between={between:.3f}  vs (p95_a+p95_b)="
                f"{wa95 + wb95:.3f}  separated={bootstrap[key]['separated']}")

    # --- decision ---
    decision = classify_stationarity(cross, between_fro_stack, between_angles, bootstrap)
    run.log(f"VERDICT: {decision['verdict']}  -> Phase 3 Branch {decision['phase3_branch']}"
            + (f"  (boundary {decision['segment_boundary']})"
               if decision["segment_boundary"] else ""))

    # --- persist ---
    _save_tables(run, kernels, cross, between_fro_perlag, k, bootstrap, cross_by_rank)
    stats = {
        "model": mcfg["name"], "n_train": len(tr), "n_test": len(te),
        "n_layers": n_delta, "d_model": d,
        "k_selection": ksel, "k": k,
        "saturation_rank": sat, "locked_rank": r, "working_rank": working_r,
        "spot_check": {"r2_r": spot.r2[r], "r2_2r": spot.r2[2 * r], "gain": gain},
        "bands": {nm: bands[nm] for nm in BAND_NAMES},
        "within_band_r2": {nm: kernels[nm].within_r2 for nm in BAND_NAMES},
        "cross_band_r2": cross,
        "cross_band_r2_by_rank": cross_by_rank,
        "between_frobenius_stack": between_fro_stack,
        "between_angles": between_angles,
        "bootstrap": bootstrap,
        "decision": decision,
        "input_pcs": input_pcs, "n_boot": n_boot,
        "caveats": (
            "AR-on-Δ stationarity is a strong proxy for H2's MA/generative "
            "stationarity, not a logical equivalent (plan Step 1); the MA recheck is "
            "Phase 3 Step 4. Consecutive Δ's share a ±v_ℓ term, a mechanism identical "
            "across depth that biases the kernel comparison TOWARD stationarity — so a "
            "stationary verdict is a permissive (not conservative) read; the AR signal "
            "itself was confirmed real by the Step 5 scrambled null. "
            + ("k is CAPPED at the feasibility limit: this tests SHORT-horizon "
               "stationarity only; long-horizon (lags up to the Granger reach) is "
               "deferred to the Phase 3 Step 4 MA recheck. "
               if (ksel is not None and ksel["capped"]) else "")
            + "Dataset is 3 classes / 1555 ex (below the ≥1000/class target); the "
            "per-class cross-relation test is Step 4."
        ),
    }
    run.save_json("tables/step1_stats.json", stats)
    _plots(run, kernels, cross, between_fro_perlag, k, spreads, bootstrap, between_angles,
           working_r)

    print(f"\n=== Phase 2 Step 1 — stationarity (k={k}, r={working_r}) ===")
    print("within-band R²:  " + "  ".join(
        f"{nm}={kernels[nm].within_r2:.3f}" for nm in BAND_NAMES))
    print("cross-band R² (rows=fit, cols=eval):")
    print("            " + "  ".join(f"{nm:>8s}" for nm in BAND_NAMES))
    for fnm in BAND_NAMES:
        print(f"  {fnm:8s}  " + "  ".join(f"{cross[fnm][enm]:8.3f}" for enm in BAND_NAMES))
    print("between-band ‖ΔW‖: " + "  ".join(
        f"{k_}={v:.3f}" for k_, v in between_fro_stack.items()))
    print(f"\nVERDICT: {decision['verdict']} -> Phase 3 Branch {decision['phase3_branch']}")
    return 0


def _ang(angles: np.ndarray) -> dict:
    return {"mean_deg": float(np.mean(angles)), "median_deg": float(np.median(angles)),
            "max_deg": float(np.max(angles))}


def _save_tables(run, kernels, cross, perlag, k, bootstrap, cross_by_rank) -> None:
    import pandas as pd

    pd.DataFrame(
        [{"fit_band": f, **{f"eval_{e}": cross[f][e] for e in BAND_NAMES}} for f in BAND_NAMES]
    ).to_csv(run.path("tables/cross_band_r2.csv"), index=False)

    pd.DataFrame(
        [{"rank": r, "fit_band": f, **{f"eval_{e}": cross_by_rank[r][f][e] for e in BAND_NAMES}}
         for r in sorted(cross_by_rank) for f in BAND_NAMES]
    ).to_csv(run.path("tables/rank_robustness.csv"), index=False)

    rows = []
    for key, dists in perlag.items():
        for j, dist in enumerate(dists, start=1):
            rows.append({"pair": key, "lag": j, "rel_frobenius": dist})
    pd.DataFrame(rows).to_csv(run.path("tables/per_lag_frobenius.csv"), index=False)

    pd.DataFrame(
        [{"pair": key, **{kk: vv for kk, vv in val.items()}} for key, val in bootstrap.items()]
    ).to_csv(run.path("tables/bootstrap.csv"), index=False)


def _plots(run, kernels, cross, perlag, k, spreads, bootstrap, angles, r) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. Cross-band R² confusion (rows=fit band, cols=eval band).
    mat = np.array([[cross[f][e] for e in BAND_NAMES] for f in BAND_NAMES])
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(mat, cmap="viridis", vmin=max(0, mat.min() - 0.05), vmax=mat.max())
    ax.set_xticks(range(3), BAND_NAMES); ax.set_yticks(range(3), BAND_NAMES)
    ax.set_xlabel("evaluated on band"); ax.set_ylabel("kernel fit on band")
    ax.set_title(f"Cross-band held-out R²  (k={k}, r={r})")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center",
                    color="w" if mat[i, j] < mat.max() - 0.15 else "k", fontsize=9)
    fig.colorbar(im, ax=ax, label="held-out R²"); fig.tight_layout()
    fig.savefig(run.path("figures/cross_band_r2.png"), dpi=130); plt.close(fig)

    # 2. Per-lag Frobenius distance, grouped by band pair.
    fig, ax = plt.subplots(figsize=(7, 4))
    lags = range(1, k + 1)
    width = 0.26
    for i, (key, dists) in enumerate(perlag.items()):
        ax.bar([x + (i - 1) * width for x in lags], dists, width=width, label=key)
    ax.set_xlabel("lag j"); ax.set_ylabel("relative ‖W_j^A − W_j^B‖_F")
    ax.set_xticks(list(lags)); ax.set_title("Per-lag kernel-block distance between bands")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(run.path("figures/per_lag_frobenius.png"), dpi=130); plt.close(fig)

    # 3. Bootstrap: within-band spread distributions vs between-band distances.
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.violinplot([spreads[nm] for nm in BAND_NAMES], showmeans=True)
    ax.set_xticks(range(1, 4), [f"within-{nm}" for nm in BAND_NAMES])
    colors = {"early-middle": "tab:blue", "middle-late": "tab:orange", "early-late": "tab:green"}
    for key, c in colors.items():
        ax.axhline(bootstrap[key]["between"], color=c, lw=1.6, ls="--",
                   label=f"between {key}={bootstrap[key]['between']:.3f}")
    ax.set_ylabel("relative Frobenius distance")
    ax.set_title("Within-band bootstrap spread vs between-band distance")
    ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(run.path("figures/bootstrap_separation.png"), dpi=130); plt.close(fig)

    # 4. Principal angle spectra (output / Δ-space): mean & max angle per band pair.
    fig, ax = plt.subplots(figsize=(7, 4))
    pairs = list(angles.keys())
    means = [angles[p]["output"]["mean_deg"] for p in pairs]
    maxes = [angles[p]["output"]["max_deg"] for p in pairs]
    xs = range(len(pairs))
    ax.bar(xs, means, width=0.5, label="mean angle", color="tab:purple", alpha=0.8)
    ax.plot(xs, maxes, "k_", ms=18, label="max angle")
    ax.set_xticks(list(xs), pairs); ax.set_ylabel("principal angle (deg)")
    ax.set_ylim(0, 92); ax.axhline(90, color="grey", lw=0.6, ls=":")
    ax.set_title(f"Top-{r} output-subspace principal angles between bands")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(run.path("figures/principal_angles.png"), dpi=130); plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
