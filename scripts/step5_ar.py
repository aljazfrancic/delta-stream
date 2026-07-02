#!/usr/bin/env python3
"""AR(k) on Δ over depth + nested-F (Granger-style) lag test (Phase 1 Step 5).

Gated on Step 3 beating the Tuned Lens (it does: ridge 0.602 ≫ TL −0.691). Fits a
single low-rank AR(k) kernel shared across depths with a per-depth intercept μ_ℓ,
for k ∈ {1,2,3,5,10,15,20} and the Step-3 rank sweep, and runs the per-(depth, lag)
multivariate nested-F lag test on a top-q PCA projection of Δ. Both are compared to
**Null 1 (scrambled activations)** — the example axis permuted independently per
depth. (Null 2, context perturbation, is a separate model-collection script.)

    python scripts/step5_ar.py --config configs/pythia-410m-deduped.yaml \
        --desc ph1-step5-410md \
        --step3-csv runs/2026-06-21-ph1-step3-410md/tables/ridge_r2.csv

Outputs: tables/ar_r2.csv (k × rank, real + null), tables/granger_real.csv,
tables/granger_null.csv, tables/step5_stats.json, and figures/ (AR R² vs k,
depth×lag −log10 p heatmaps for real & null).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from layer_dynamics.cli import add_common_args, resolve_out
from layer_dynamics.diagnostics.autoregress import (
    LAG_SWEEP,
    fit_ar_pooled,
    fit_delta_pca,
    granger_lag_grid,
    scramble_per_depth,
)
from layer_dynamics.diagnostics.dataset import deltas, load_residual_stream, train_test_split_idx
from layer_dynamics.runlog import init_run


def _ar_sweep(dlt_tr, dlt_te, lags, ranks, label, run, *, input_pcs):
    """Fit AR(k) for every k in the sweep; return rows + the best (k, rank, R²).

    The predictor PCA basis is fit once on the training Δ and shared across k so the
    sweep is fast and every k uses the same compressed predictor coordinates.
    """
    basis = fit_delta_pca(dlt_tr, input_pcs) if input_pcs else None
    rows, best = [], (-1, -1, -np.inf)
    for k in lags:
        fit = fit_ar_pooled(dlt_tr, dlt_te, k, ranks=ranks, input_basis=basis,
                            input_pcs=input_pcs)
        rows.append({
            "label": label, "k": k, "alpha": fit.alpha,
            "n_train_rows": fit.n_train_rows, "n_depths": fit.n_depths,
            **{f"r{r}": fit.r2[r] for r in ranks}, "full": fit.r2["full"],
            "best_rank": fit.best_rank, "best_r2": fit.best_r2,
        })
        if fit.best_r2 > best[2]:
            best = (k, fit.best_rank, fit.best_r2)
        run.log(f"  [{label}] AR(k={k:2d}) best R²={fit.best_r2:.4f} "
                f"(rank {fit.best_rank}, full {fit.r2['full']:.4f}, α={fit.alpha:.1f})")
    return rows, best


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--step3-csv", type=Path, default=None,
                        help="Step 3 ridge_r2.csv to report the v_ℓ-alone comparison.")
    parser.add_argument("--h5", type=Path, default=None,
                        help="Override the residual-stream HDF5 (default: "
                             "data/<model>/residual_stream.h5). Used by the E4 "
                             "generic-text generality control.")
    parser.add_argument("--split-by", choices=["random", "object", "subject"],
                        default="random",
                        help="R5: entity-disjoint train/test split keyed on the "
                             "filtered_triples.parquet column (objects recur heavily; "
                             "random shares answers across the boundary).")
    args = parser.parse_args(argv)

    run = init_run(resolve_out(args), config_path=args.config, seed=args.seed)
    cfg = run.config
    mcfg = cfg["model"]
    dgcfg = cfg.get("diagnostics", {})
    ranks = tuple(dgcfg.get("ranks", [16, 32, 64, 128]))
    test_frac = float(dgcfg.get("test_frac", 0.2))
    lags = tuple(dgcfg.get("ar_lags", LAG_SWEEP))
    q_pca = int(dgcfg.get("granger_pca", 32))
    input_pcs = int(dgcfg.get("ar_input_pcs", 256))
    null_seed = int(dgcfg.get("ar_null_seed", 1234))

    h5 = args.h5 if args.h5 is not None else Path("data") / mcfg["name"] / "residual_stream.h5"
    rs, _, _ = load_residual_stream(h5)
    n, n_states, d = rs.shape
    dlt = deltas(rs)  # [N, L, D]
    run.log(f"loaded {h5}: residual_stream {rs.shape}, deltas {dlt.shape}")

    if args.split_by != "random":
        from layer_dynamics.diagnostics.dataset import entity_split_idx

        pq = Path("data") / mcfg["name"] / "filtered_triples.parquet"
        tr, te = entity_split_idx(n, pq, args.split_by, test_frac=test_frac, seed=args.seed)
    else:
        tr, te = train_test_split_idx(n, test_frac=test_frac, seed=args.seed)
    dlt_tr, dlt_te = dlt[tr], dlt[te]
    run.log(f"split: {len(tr)} train / {len(te)} test (by={args.split_by})")
    run.log(f"AR lag sweep k={list(lags)}, ranks={list(ranks)}, "
            f"predictor PCs={input_pcs}, Granger q_pca={q_pca}")

    # --- AR(k) fit: real + Null 1 (scrambled per depth) ---
    run.log("AR(k) pooled fit (real) ...")
    ar_rows, ar_best = _ar_sweep(dlt_tr, dlt_te, lags, ranks, "real", run, input_pcs=input_pcs)
    run.log("AR(k) pooled fit (scrambled null) ...")
    s_tr = scramble_per_depth(dlt_tr, null_seed)
    s_te = scramble_per_depth(dlt_te, null_seed + 777)
    null_rows, null_best = _ar_sweep(s_tr, s_te, lags, ranks, "null", run, input_pcs=input_pcs)

    # --- Granger nested-F grid: real + Null 1 ---
    run.log("Granger nested-F grid (real) ...")
    g_real = granger_lag_grid(dlt, lags=lags, q_pca=q_pca)
    run.log("Granger nested-F grid (scrambled null) ...")
    g_null = granger_lag_grid(scramble_per_depth(dlt, null_seed), lags=lags, q_pca=q_pca)

    bonf = 0.05 / g_real["n_tests"]
    thr_nlp = -np.log10(bonf)
    real_sig = int(np.nansum(g_real["neg_log10_p"] > thr_nlp))
    null_sig = int(np.nansum(g_null["neg_log10_p"] > thr_nlp))
    run.log(f"Granger: {real_sig}/{g_real['n_tests']} cells Bonferroni-significant "
            f"(α={bonf:.2e}, −log10 thr={thr_nlp:.2f}); scrambled null: {null_sig}")

    import pandas as pd

    df_ar = pd.DataFrame(ar_rows + null_rows)
    df_ar.to_csv(run.path("tables/ar_r2.csv"), index=False)
    _granger_csv(run, "tables/granger_real.csv", g_real)
    _granger_csv(run, "tables/granger_null.csv", g_null)

    # v_ℓ-alone comparison (Step 3): does AR on preceding Δ beat regressing Δ_ℓ on v_ℓ?
    v_alone = None
    if args.step3_csv is not None and args.step3_csv.exists():
        v_alone = float(pd.read_csv(args.step3_csv)["best_r2"].mean())

    # Stationarity-implicit diagnostic (plan Step 5 / Gate-1 routing): low pooled AR R²
    # *and* widely Bonferroni-significant Granger => H1 holds non-stationarily, do NOT
    # route to the negative-result branch. Flag it explicitly for the Gate-1 memo.
    frac_real_sig = real_sig / g_real["n_tests"]
    nonstationary_flag = bool(ar_best[2] < 0.2 and frac_real_sig > 0.5)

    stats = {
        "h5": str(h5),
        "n_train": len(tr), "n_test": len(te), "n_layers": int(n_states - 1), "d_model": d,
        "lags": list(lags), "ranks": list(ranks), "granger_q_pca": q_pca,
        "ar_input_pcs": input_pcs,
        "ar_best": {"k": ar_best[0], "rank": ar_best[1], "r2": ar_best[2]},
        "ar_null_best": {"k": null_best[0], "rank": null_best[1], "r2": null_best[2]},
        "ar_real_minus_null_bestr2": ar_best[2] - null_best[2],
        "step3_ridge_v_alone_r2": v_alone,
        "granger_n_tests": g_real["n_tests"],
        "granger_bonferroni_alpha": bonf,
        "granger_neg_log10_threshold": thr_nlp,
        "granger_real_significant": real_sig,
        "granger_null_significant": null_sig,
        "granger_real_significant_frac": frac_real_sig,
        "nonstationary_h1_flag": nonstationary_flag,
        "interpretation": (
            "Pooled AR(k) R² is the held-out fraction of Δ_ℓ variance (beyond the "
            "per-depth mean μ_ℓ) explained by the k preceding updates; a single kernel "
            "is shared across depth, so a high value is jointly consistent with H1 and "
            "approximate AR-stationarity. The Granger grid is the per-(depth,lag) "
            "nested-F that localises how far back the dependence reaches and is the "
            "stationarity-free read. Null 1 (scrambled per depth) must collapse both."
        ),
    }
    run.save_json("tables/step5_stats.json", stats)

    _plots(run, df_ar, ranks, lags, g_real, g_null, thr_nlp)

    print("\n=== AR(k) held-out pooled R² (best rank) — real vs scrambled null ===")
    real_by_k = {r["k"]: r for r in ar_rows}
    null_by_k = {r["k"]: r for r in null_rows}
    for k in lags:
        print(f"  k={k:2d}  real={real_by_k[k]['best_r2']:.4f} "
              f"(rank {real_by_k[k]['best_rank']:3d})   null={null_by_k[k]['best_r2']:.4f}")
    print(f"\nbest AR: k={ar_best[0]} rank={ar_best[1]} R²={ar_best[2]:.4f}  "
          f"(null best {null_best[2]:.4f})")
    if v_alone is not None:
        print(f"Step 3 ridge from v_ℓ alone: R²={v_alone:.4f} (different predictor set)")
    print(f"Granger: {real_sig}/{g_real['n_tests']} Bonferroni-sig cells "
          f"(null {null_sig}); non-stationary-H1 flag={nonstationary_flag}")
    return 0


def _granger_csv(run, name, grid) -> None:
    import pandas as pd

    lags = grid["lags"]
    rows = []
    n_depth = grid["F"].shape[0]
    for ell in range(n_depth):
        for ji, j in enumerate(lags):
            f = grid["F"][ell, ji]
            if np.isnan(f):
                continue
            rows.append({
                "depth": ell, "lag": j,
                "F": float(f), "neg_log10_p": float(grid["neg_log10_p"][ell, ji]),
                "df1": float(grid["df1"][ell, ji]), "df2": float(grid["df2"][ell, ji]),
            })
    pd.DataFrame(rows).to_csv(run.path(name), index=False)


def _plots(run, df_ar, ranks, lags, g_real, g_null, thr_nlp) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    real = df_ar[df_ar["label"] == "real"]
    null = df_ar[df_ar["label"] == "null"]

    # AR R² vs k: best-rank real vs null, plus the per-rank curves.
    fig, ax = plt.subplots(figsize=(7, 4))
    for r in ranks:
        ax.plot(real["k"], real[f"r{r}"], "-", lw=0.8, alpha=0.6, label=f"real r{r}")
    ax.plot(real["k"], real["best_r2"], "-o", color="tab:blue", lw=2, label="real best rank")
    ax.plot(null["k"], null["best_r2"], "--s", color="tab:red", lw=1.5, label="scrambled null")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("AR order k"); ax.set_ylabel("held-out pooled R² on Δ_ℓ (vs μ_ℓ)")
    ax.set_title("AR(k) depth-local predictability of Δ"); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(run.path("figures/ar_r2_vs_k.svg"), bbox_inches="tight")
    plt.close(fig)

    # Granger (depth × lag) −log10 p heatmaps, real and null, shared colour scale.
    vmax = float(np.nanmax([np.nanmax(g_real["neg_log10_p"]), thr_nlp * 2]))
    for tag, grid in (("real", g_real), ("null", g_null)):
        mat = grid["neg_log10_p"].T  # [lag, depth]
        fig, ax = plt.subplots(figsize=(9, 3.0))
        im = ax.imshow(mat, aspect="auto", cmap="magma", origin="lower", vmin=0, vmax=vmax)
        ax.set_yticks(range(len(lags)), [str(j) for j in lags])
        ax.set_xlabel("target depth ℓ"); ax.set_ylabel("lag j")
        ax.set_title(f"Granger nested-F (−log10 p), {tag}  "
                     f"[Bonferroni thr={thr_nlp:.1f}]")
        fig.colorbar(im, ax=ax, label="−log10 p")
        fig.tight_layout()
        fig.savefig(run.path(f"figures/granger_{tag}.svg"), bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
