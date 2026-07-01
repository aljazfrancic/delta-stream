#!/usr/bin/env python3
"""T5 — analytic tautology floor (Phase 4 H1-validity).

The explicit closed-form companion to T1. T1 *refits* the AR after stripping the
persistent state directions and reports the surviving R² (factual 0.44 / generic
0.29). T5 instead decomposes the **single headline AR fit**: the held-out explained
variance splits orthogonally (Pythagorean) over the top-m persistent state subspace
(``par``, tautological) and its complement (``perp``, genuine), and the two pieces
add back to the headline R² *exactly*:

    R²_full = frac_par + frac_perp

so ``frac_par`` is the explicit tautology floor to subtract from the headline number,
and ``frac_perp`` the genuine remainder. (The naive "predict −v_ℓ" floor is *not*
used — ‖v_ℓ‖ ≫ ‖Δ_ℓ‖ makes it wildly mis-scaled; the tautology lives in the
persistent *directions*, which is what this projects onto.)

    python scripts/phase4_t5_taut_floor.py --config configs/pythia-410m-deduped.yaml \
        --desc ph4-t5-taut-floor-410md
    # generic-text store:
    python scripts/phase4_t5_taut_floor.py --config configs/pythia-410m-deduped.yaml \
        --desc ph4-t5-taut-floor-410md-general \
        --h5 data/pythia-410m-deduped-general/residual_stream.h5

Outputs: tables/t5_taut_floor.csv, tables/t5_stats.json, figures/t5_floor.svg.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from layer_dynamics.cli import add_common_args, resolve_out
from layer_dynamics.diagnostics.autoregress import LAG_SWEEP, tautology_floor_decomposition
from layer_dynamics.diagnostics.dataset import deltas, load_residual_stream, train_test_split_idx
from layer_dynamics.runlog import init_run

DEFAULT_M_SWEEP = (1, 5, 20, 50)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--h5", type=Path, default=None,
                        help="Override residual-stream HDF5 (default data/<model>/...).")
    parser.add_argument("--m-sweep", type=int, nargs="+", default=list(DEFAULT_M_SWEEP),
                        help="Top-m state directions defining the persistent subspace.")
    args = parser.parse_args(argv)

    run = init_run(resolve_out(args), config_path=args.config, seed=args.seed)
    cfg = run.config
    mcfg = cfg["model"]
    dgcfg = cfg.get("diagnostics", {})
    ranks = tuple(dgcfg.get("ranks", [16, 32, 64, 128]))
    test_frac = float(dgcfg.get("test_frac", 0.2))
    lags = tuple(dgcfg.get("ar_lags", LAG_SWEEP))
    input_pcs = int(dgcfg.get("ar_input_pcs", 256))

    h5 = args.h5 if args.h5 is not None else Path("data") / mcfg["name"] / "residual_stream.h5"
    rs, _, _ = load_residual_stream(h5)
    n, n_states, d = rs.shape
    dlt = deltas(rs)
    run.log(f"loaded {h5}: residual_stream {rs.shape}, deltas {dlt.shape}")

    tr, te = train_test_split_idx(n, test_frac=test_frac, seed=args.seed)
    # Headline used the best k over the sweep (k=20 on 410m); match it so the floor is
    # decomposed off the *same* fit that produced the headline number.
    k = max(lags)
    run.log(f"split: {len(tr)} train / {len(te)} test; k={k}, ranks={list(ranks)}, "
            f"predictor PCs={input_pcs}, m-sweep={args.m_sweep}")

    res = tautology_floor_decomposition(
        rs, dlt, tr, te, k, m_sweep=tuple(args.m_sweep), ranks=ranks, input_pcs=input_pcs
    )
    run.log(f"headline AR(k={k}) held-out R²_full = {res['r2_full']:.4f} "
            f"(best rank {res['best_rank']})")

    import pandas as pd

    rows = res["by_m"]
    for r in rows:
        run.log(f"  m={r['m']:2d} (Δ-var in subspace {r['par_var_frac']:.3f}): "
                f"tautology floor frac_par={r['frac_par']:.4f}  genuine frac_perp="
                f"{r['frac_perp']:.4f}  (sum={r['frac_par'] + r['frac_perp']:.4f})  | "
                f"r2_par={r['r2_par']:.3f} r2_perp={r['r2_perp']:.3f}")
    pd.DataFrame(rows).to_csv(run.path("tables/t5_taut_floor.csv"), index=False)

    stats = {
        "h5": str(h5), "n_train": int(len(tr)), "n_test": int(len(te)),
        "n_layers": int(n_states - 1), "d_model": int(d),
        "k": res["k"], "best_rank": res["best_rank"], "r2_full": res["r2_full"],
        "by_m": rows,
        "interpretation": (
            "R²_full splits EXACTLY into frac_par (tautology floor — explained "
            "variance inside the top-m persistent state directions) + frac_perp "
            "(genuine remainder, in the complement). frac_par is the explicit "
            "subtracted baseline for the headline AR R²; r2_par≈1 confirms the "
            "persistent directions are near-perfectly 'predictable' (smooth drift = "
            "the tautology), while frac_perp is the depth structure that survives it. "
            "Cross-checks T1, whose refit-on-complement R² is a re-optimised analog "
            "of r2_perp."
        ),
    }
    run.save_json("tables/t5_stats.json", stats)
    _plot(run, res)

    print(f"\n=== T5 analytic tautology floor (k={res['k']}, R²_full={res['r2_full']:.4f}) ===")
    print(f"  {'m':>3} {'Δvar_in':>8} {'floor(par)':>11} {'genuine(perp)':>13} "
          f"{'sum':>6} {'r2_par':>7} {'r2_perp':>8}")
    for r in rows:
        print(f"  {r['m']:>3} {r['par_var_frac']:>8.3f} {r['frac_par']:>11.4f} "
              f"{r['frac_perp']:>13.4f} {r['frac_par'] + r['frac_perp']:>6.3f} "
              f"{r['r2_par']:>7.3f} {r['r2_perp']:>8.3f}")
    return 0


def _plot(run, res) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = res["by_m"]
    ms = [r["m"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(ms, [r["frac_par"] for r in rows], "-o", color="tab:red",
            label="frac_par (tautology floor)")
    ax.plot(ms, [r["frac_perp"] for r in rows], "-s", color="tab:blue",
            label="frac_perp (genuine remainder)")
    ax.axhline(res["r2_full"], color="k", lw=0.8, ls="--",
               label=f"R²_full = {res['r2_full']:.3f}")
    ax.set_xlabel("# persistent state directions m (subspace)")
    ax.set_ylabel("share of total Δ variance (additive)")
    ax.set_title("T5 · tautology floor (frac_par) vs genuine (frac_perp)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(run.path("figures/t5_floor.svg"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
