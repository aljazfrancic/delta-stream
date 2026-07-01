#!/usr/bin/env python3
"""H1-validity / tautology pressure-tests T1–T3 (gap audit, 2026-06-30).

The headline AR-on-Δ R² is the paper's most contestable number because of the
±v_ℓ running-sum tautology (Δ_ℓ ⊃ −v_ℓ ⊂ Δ_{ℓ-1}): a linear map can score high
just by reading the persistent, high-variance v_ℓ component. The scramble null
(Null 1) destroys the genuine structure *and* the tautology together, so it cannot
separate "real depth computation" from "the residual stream persists". This script
runs the three data-only tests that do separate them, on one residual-stream store:

  T1  persistence-stratified AR — project the **states** v_ℓ onto the orthogonal
      complement of their top-m principal directions, re-difference into Δ, re-run
      the AR sweep (real + scramble null). Sweep m ∈ {0,1,5,20,50}. If R² survives
      as the dominant directions are stripped, the predictability is genuine depth
      structure; if it decays toward the null, it was the ±v_ℓ tautology.

  T2  persistence-matched surrogate null — independent per-layer Gaussian steps
      matched to each Δ_ℓ's mean+covariance (a random walk with no cross-layer
      dependence). The honest floor for the headline is `real − surrogate`, not
      `real − scramble`. Expected ≈0: with independent steps the shared ±v_ℓ terms
      cancel, so any AR-on-Δ R² above this floor is not a persistence artefact.

  T3  lag≥2 Granger re-headline — the tautology lives entirely at lag 1, so the
      lag≥2 Bonferroni-significant count is an intrinsically tautology-free read.

    python scripts/phase4_pressure_tests.py --config configs/pythia-410m-deduped.yaml \
        --desc ph4-pressure-410md
    # generic-text store:
    python scripts/phase4_pressure_tests.py --config configs/pythia-410m-deduped.yaml \
        --desc ph4-pressure-410md-general \
        --h5 data/pythia-410m-deduped-general/residual_stream.h5

Outputs: tables/{t1_persistence_sweep,t2_surrogate,t3_granger_by_lag}.csv,
tables/pressure_stats.json, figures/{t1_persistence_sweep,t2_t3_summary}.svg.
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
    granger_significant_by_lag,
    random_walk_surrogate,
    remove_state_dirs,
    scramble_per_depth,
    state_pca_basis,
)
from layer_dynamics.diagnostics.dataset import deltas, load_residual_stream, train_test_split_idx
from layer_dynamics.runlog import init_run

DEFAULT_M_SWEEP = (0, 1, 5, 20, 50)


def _best_ar_r2(dlt_tr, dlt_te, lags, ranks, input_pcs) -> tuple[float, int, int]:
    """Best held-out AR R² over the (k, rank) sweep; shared predictor PCA basis."""
    basis = fit_delta_pca(dlt_tr, input_pcs) if input_pcs else None
    best = (-np.inf, -1, -1)
    for k in lags:
        fit = fit_ar_pooled(dlt_tr, dlt_te, k, ranks=ranks, input_basis=basis,
                            input_pcs=input_pcs)
        if fit.best_r2 > best[0]:
            best = (fit.best_r2, k, fit.best_rank)
    return best


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--h5", type=Path, default=None,
                        help="Override residual-stream HDF5 (default data/<model>/...).")
    parser.add_argument("--m-sweep", type=int, nargs="+", default=list(DEFAULT_M_SWEEP),
                        help="T1: # of top state-PCA directions to project out.")
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
    dlt = deltas(rs)
    run.log(f"loaded {h5}: residual_stream {rs.shape}, deltas {dlt.shape}")

    tr, te = train_test_split_idx(n, test_frac=test_frac, seed=args.seed)
    run.log(f"split: {len(tr)} train / {len(te)} test; ranks={list(ranks)}, "
            f"lags={list(lags)}, predictor PCs={input_pcs}")

    import pandas as pd

    # State-PCA: report the variance captured by the directions being removed.
    flat_states = rs.reshape(n * n_states, d).astype(np.float64)
    sv = np.linalg.svd(flat_states - flat_states.mean(0), compute_uv=False)
    var_frac = np.cumsum(sv**2) / np.sum(sv**2)

    # ---- T1: persistence-stratified AR over the m-sweep ----
    run.log("T1 — persistence-stratified AR (projecting out top-m state directions) ...")
    t1_rows = []
    for m in args.m_sweep:
        _, dirs = state_pca_basis(rs, m)
        dproj = remove_state_dirs(dlt, dirs)
        real = _best_ar_r2(dproj[tr], dproj[te], lags, ranks, input_pcs)
        s_tr = scramble_per_depth(dproj[tr], null_seed)
        s_te = scramble_per_depth(dproj[te], null_seed + 777)
        null = _best_ar_r2(s_tr, s_te, lags, ranks, input_pcs)
        vf = float(var_frac[m - 1]) if m > 0 else 0.0
        t1_rows.append({"m_removed": m, "state_var_removed": vf,
                        "real_best_r2": real[0], "real_k": real[1], "real_rank": real[2],
                        "scramble_null_r2": null[0], "real_minus_scramble": real[0] - null[0]})
        run.log(f"  m={m:2d} (state var removed {vf:.3f}): real R²={real[0]:.4f} "
                f"(k={real[1]}, rank={real[2]})  scramble={null[0]:.4f}")
    pd.DataFrame(t1_rows).to_csv(run.path("tables/t1_persistence_sweep.csv"), index=False)

    # ---- T2: persistence-matched independent-step surrogate null ----
    run.log("T2 — persistence-matched surrogate null (independent matched steps) ...")
    surr = random_walk_surrogate(dlt, null_seed, rank=input_pcs)
    real_full = _best_ar_r2(dlt[tr], dlt[te], lags, ranks, input_pcs)
    surr_r2 = _best_ar_r2(surr[tr], surr[te], lags, ranks, input_pcs)
    scr = scramble_per_depth(dlt[tr], null_seed), scramble_per_depth(dlt[te], null_seed + 777)
    scr_r2 = _best_ar_r2(scr[0], scr[1], lags, ranks, input_pcs)
    t2 = {"real_best_r2": real_full[0], "real_k": real_full[1], "real_rank": real_full[2],
          "surrogate_r2": surr_r2[0], "scramble_r2": scr_r2[0],
          "real_minus_surrogate": real_full[0] - surr_r2[0],
          "real_minus_scramble": real_full[0] - scr_r2[0]}
    pd.DataFrame([t2]).to_csv(run.path("tables/t2_surrogate.csv"), index=False)
    run.log(f"  real R²={real_full[0]:.4f}  surrogate={surr_r2[0]:.4f}  "
            f"scramble={scr_r2[0]:.4f}  (real−surrogate={t2['real_minus_surrogate']:.4f})")

    # ---- T3: lag≥2 (tautology-free) Granger re-headline ----
    run.log("T3 — Granger significance split by lag (lag1 vs lag≥2) ...")
    g_real = granger_lag_grid(dlt, lags=lags, q_pca=q_pca)
    g_null = granger_lag_grid(scramble_per_depth(dlt, null_seed), lags=lags, q_pca=q_pca)
    by_lag_real = granger_significant_by_lag(g_real)
    by_lag_null = granger_significant_by_lag(g_null)
    pd.DataFrame([{"variant": "real", **by_lag_real},
                  {"variant": "scramble_null", **by_lag_null}]).to_csv(
        run.path("tables/t3_granger_by_lag.csv"), index=False)
    run.log(f"  real: lag1 {by_lag_real['lag1_sig']}/{by_lag_real['lag1_total']}, "
            f"lag≥2 {by_lag_real['lagge2_sig']}/{by_lag_real['lagge2_total']}  | "
            f"null lag≥2 {by_lag_null['lagge2_sig']}/{by_lag_null['lagge2_total']}")

    stats = {
        "h5": str(h5), "n_train": len(tr), "n_test": len(te),
        "n_layers": int(n_states - 1), "d_model": d,
        "state_var_top1": float(var_frac[0]), "state_var_top20": float(var_frac[min(19, d - 1)]),
        "t1_persistence_sweep": t1_rows,
        "t2_surrogate": t2,
        "t3_granger": {"real": by_lag_real, "scramble_null": by_lag_null},
        "interpretation": (
            "T1: if real_best_r2 survives as m grows (state var removed), the AR-on-Δ "
            "predictability is genuine depth structure, not the ±v_ℓ persistence "
            "tautology; if it decays toward scramble_null_r2 it was tautological. "
            "T2: real−surrogate is the honest floor (independent matched steps cancel "
            "the ±v_ℓ term); >0 means the predictability needs genuine step-to-step "
            "structure. T3: lag≥2 significant cells are tautology-free (the shared v_ℓ "
            "only couples lag 1)."
        ),
    }
    run.save_json("tables/pressure_stats.json", stats)
    _plots(run, t1_rows, t2, by_lag_real, by_lag_null)

    print("\n=== T1 persistence-stratified AR (project out top-m state dirs) ===")
    for r in t1_rows:
        print(f"  m={r['m_removed']:2d}  state-var-removed={r['state_var_removed']:.3f}  "
              f"real R²={r['real_best_r2']:.4f}  scramble={r['scramble_null_r2']:.4f}")
    print(f"\n=== T2 surrogate null ===\n  real={t2['real_best_r2']:.4f}  "
          f"surrogate={t2['surrogate_r2']:.4f}  scramble={t2['scramble_r2']:.4f}  "
          f"(real−surrogate={t2['real_minus_surrogate']:.4f})")
    print(f"\n=== T3 Granger by lag ===\n  real lag1 "
          f"{by_lag_real['lag1_sig']}/{by_lag_real['lag1_total']}, "
          f"lag≥2 {by_lag_real['lagge2_sig']}/{by_lag_real['lagge2_total']}  "
          f"(null lag≥2 {by_lag_null['lagge2_sig']}/{by_lag_null['lagge2_total']})")
    return 0


def _plots(run, t1_rows, t2, by_lag_real, by_lag_null) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ms = [r["m_removed"] for r in t1_rows]
    axes[0].plot(ms, [r["real_best_r2"] for r in t1_rows], "-o", label="real")
    axes[0].plot(ms, [r["scramble_null_r2"] for r in t1_rows], "--s", label="scramble null")
    axes[0].axhline(0, color="k", lw=0.6)
    axes[0].set_xlabel("# top state directions removed (m)")
    axes[0].set_ylabel("best held-out AR R² on Δ")
    axes[0].set_title("T1 · persistence-stratified AR"); axes[0].legend(fontsize=8)
    sx = axes[0].twiny()
    sx.set_xlim(axes[0].get_xlim())
    sx.set_xticks(ms, [f"{r['state_var_removed']:.2f}" for r in t1_rows], fontsize=7)
    sx.set_xlabel("state variance removed", fontsize=8)

    labels = ["real", "surrogate\n(T2)", "scramble"]
    vals = [t2["real_best_r2"], t2["surrogate_r2"], t2["scramble_r2"]]
    axes[1].bar(labels, vals, color=["tab:blue", "tab:orange", "tab:red"])
    axes[1].axhline(0, color="k", lw=0.6)
    axes[1].set_ylabel("best held-out AR R² on Δ")
    r = by_lag_real
    axes[1].set_title(f"T2 nulls + T3 lag≥2 Granger: "
                      f"{r['lagge2_sig']}/{r['lagge2_total']} sig")
    fig.tight_layout(); fig.savefig(run.path("figures/t2_t3_summary.svg"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
