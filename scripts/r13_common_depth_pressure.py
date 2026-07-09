#!/usr/bin/env python3
"""R13 — re-run the T1/T4/T5 pressure numbers at a *common depth set* (review fix).

The pooled AR(k) fit scores only the depths where all k lags exist (ℓ ≥ k), so the
Phase-1 sweep's "best over k" silently changes the evaluation set: at L=24, k=1 is an
average over 23 depths and k=20 over the four deepest updates. The consequence, found
in the 2026-07-09 referee pass:

  * the headline AR R² "0.65" is a k=20 fit — a four-depth number — while the T1
    persistence-stripped "0.436" is a k=1 fit, a 23-depth number, and the two were
    quoted as a decay curve;
  * T4's GPT-2 control compared trained (k=1, 11 depths) against untrained (k=10, two
    depths), i.e. a "learned share ~5%" computed across different evaluation sets;
  * the factual-vs-generic T1 comparison in the paper's fig14 mixed k=1 and k=20.

This script recomputes the T1 m-sweep and the T5 exact split at **k=1 on all feasible
depths** for every store the paper's pressure battery uses, so the trained/untrained
and factual/generic contrasts are like-for-like. It also reports the AR(k) sweep
restricted to a common depth set, isolating the lag effect from the depth effect.

    python scripts/r13_common_depth_pressure.py --config configs/pythia-410m-deduped.yaml \
        --desc r13-common-depth-pressure

Outputs: tables/{t1_common_depth,ar_lag_vs_depth_subset,t5_common_depth}.csv,
tables/r13_stats.json.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from layer_dynamics.cli import add_common_args, resolve_out
from layer_dynamics.diagnostics.autoregress import (
    LAG_SWEEP,
    feasible_depths,
    fit_ar_pooled,
    remove_state_dirs,
    scramble_per_depth,
    state_pca_basis,
    tautology_floor_decomposition,
)
from layer_dynamics.diagnostics.dataset import deltas, load_residual_stream, train_test_split_idx
from layer_dynamics.runlog import init_run

RANKS = (16, 32, 64, 128)
M_SWEEP = (0, 1, 5, 20, 50)

# label -> (h5, trained?, store-kind)
STORES = {
    "410m-factual-trained": "data/pythia-410m-deduped/residual_stream.h5",
    "410m-factual-untrained": "data/pythia-410m-deduped-untrained/residual_stream.h5",
    "410m-generic-trained": "data/pythia-410m-deduped-general/residual_stream.h5",
    "410m-generic-untrained": "data/pythia-410m-deduped-untrained-general/residual_stream.h5",
    "gpt2-factual-trained": "data/gpt2/residual_stream.h5",
    "gpt2-factual-untrained": "data/gpt2-untrained/residual_stream.h5",
}


def _ar1_r2(dlt, tr, te, *, depths=None) -> float:
    fit = fit_ar_pooled(dlt[tr], dlt[te], 1, ranks=RANKS, input_pcs=256, depths=depths)
    return float(fit.best_r2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--stores", type=str, nargs="+", default=list(STORES))
    args = parser.parse_args(argv)
    run = init_run(resolve_out(args), config_path=args.config, seed=args.seed)

    import pandas as pd

    # ---- (1) T1 m-sweep at k=1 (all feasible depths) for every store ----
    t1_rows, t5_rows = [], []
    for label in args.stores:
        h5 = Path(STORES[label])
        if not h5.exists():
            run.log(f"skip {label}: {h5} missing")
            continue
        rs, _, _ = load_residual_stream(h5)
        n = rs.shape[0]
        dlt = deltas(rs)
        tr, te = train_test_split_idx(n, test_frac=0.2, seed=args.seed)
        n_depths = len(feasible_depths(dlt.shape[1], 1))
        run.log(f"{label}: {rs.shape}, k=1 over {n_depths} depths")

        for m in M_SWEEP:
            _, dirs = state_pca_basis(rs, m)
            dproj = remove_state_dirs(dlt, dirs) if m > 0 else dlt
            real = _ar1_r2(dproj, tr, te)
            s_tr = scramble_per_depth(dproj[tr], 1234)
            s_te = scramble_per_depth(dproj[te], 2011)
            null = float(fit_ar_pooled(s_tr, s_te, 1, ranks=RANKS, input_pcs=256).best_r2)
            t1_rows.append({"store": label, "m_removed": m, "k": 1, "n_depths": n_depths,
                            "real_r2": real, "scramble_null_r2": null})
            run.log(f"  m={m:2d}  k=1 R²={real:.4f}  null={null:.2e}")

        t5 = tautology_floor_decomposition(rs, dlt, tr, te, 1, m_sweep=(1, 5, 20, 50),
                                           ranks=RANKS, input_pcs=256)
        for row in t5["by_m"]:
            t5_rows.append({"store": label, "k": 1, "n_depths": n_depths,
                            "r2_full": t5["r2_full"], **row})
        m50 = [r for r in t5["by_m"] if r["m"] == 50][0]
        run.log(f"  T5 @k=1: R²_full={t5['r2_full']:.4f} = {m50['frac_par']:.4f} taut "
                f"+ {m50['frac_perp']:.4f} genuine  (m=50)")

    pd.DataFrame(t1_rows).to_csv(run.path("tables/t1_common_depth.csv"), index=False)
    pd.DataFrame(t5_rows).to_csv(run.path("tables/t5_common_depth.csv"), index=False)

    # ---- (2) the lag effect, with and without the depth-subset confound ----
    rs, _, _ = load_residual_stream(Path(STORES["410m-factual-trained"]))
    dlt = deltas(rs)
    tr, te = train_test_split_idx(rs.shape[0], test_frac=0.2, seed=args.seed)
    common = feasible_depths(dlt.shape[1], max(LAG_SWEEP))
    lag_rows = []
    for k in LAG_SWEEP:
        own = fit_ar_pooled(dlt[tr], dlt[te], k, ranks=RANKS, input_pcs=256)
        fixed = fit_ar_pooled(dlt[tr], dlt[te], k, ranks=RANKS, input_pcs=256, depths=common)
        lag_rows.append({"k": k, "n_depths_own": len(own.depths), "r2_own_depths": own.best_r2,
                         "n_depths_common": len(common), "r2_common_depths": fixed.best_r2})
        run.log(f"  AR(k={k:2d}): own-depths({len(own.depths):2d}) R²={own.best_r2:.4f} | "
                f"common-depths({len(common)}) R²={fixed.best_r2:.4f}")
    pd.DataFrame(lag_rows).to_csv(run.path("tables/ar_lag_vs_depth_subset.csv"), index=False)

    lag_df = pd.DataFrame(lag_rows)
    stats = {
        "note": (
            "AR(k) pooled fits score only depths ℓ ≥ k, so 'best over k' changes the "
            "evaluation set. r2_own_depths reproduces the Phase-1 sweep; "
            "r2_common_depths holds the depth set fixed at ℓ ∈ [20,23] and is the "
            "comparison that isolates lag order. T1/T5 are recomputed at k=1 (all "
            "feasible depths) so the trained/untrained and factual/generic contrasts "
            "are like-for-like."
        ),
        "common_depths": common,
        "lag_effect_common_depths_k1_to_k20": float(
            lag_df.r2_common_depths.iloc[-1] - lag_df.r2_common_depths.iloc[0]),
        "lag_effect_own_depths_k1_to_k20": float(
            lag_df.r2_own_depths.iloc[-1] - lag_df.r2_own_depths.iloc[0]),
        "t1_k1": t1_rows,
        "t5_k1": t5_rows,
    }
    run.save_json("tables/r13_stats.json", stats)
    run.log(f"lag effect k1→k20: own-depths {stats['lag_effect_own_depths_k1_to_k20']:+.4f} "
            f"(confounded) vs common-depths "
            f"{stats['lag_effect_common_depths_k1_to_k20']:+.4f} (clean)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
