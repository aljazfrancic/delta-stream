#!/usr/bin/env python3
"""Permutation spot-check of the Granger nested-F calibration (3rd-pass finding 10).

The Step-5 Granger grid quotes Rao-F p-values, which assume Gaussian errors; the
heavy-tailed rogue dims mildly violate that, and so far the only calibration
evidence was the empirical scrambled null landing at 0/112. This runs the direct
check: a Freedman–Lane permutation of the nested-F at spot depths — permute the
reduced-model residuals, recompute F, and compare the permutation null of F to
the parametric F(df1, df2) reference (KS test + tail quantiles) plus the
permutation exceedance p of the observed F. If the tails match, the stats
appendix can say "the F approximation is calibrated under exchangeability"
instead of leaning on the 0/112 alone.

    python scripts/granger_perm_check.py --config configs/pythia-410m-deduped.yaml \
        --desc granger-permcheck-410md --depths 12 22 --n-perm 999
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from layer_dynamics.cli import add_common_args, resolve_out
from layer_dynamics.diagnostics.autoregress import LAG_SWEEP, granger_permutation_check
from layer_dynamics.diagnostics.dataset import deltas, load_residual_stream
from layer_dynamics.runlog import init_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--h5", type=Path, default=None,
                        help="Override the residual-stream HDF5 (default data/<model>/…).")
    parser.add_argument("--depths", type=int, nargs="+", default=[12, 22],
                        help="Spot target depths (22 admits the full lag sweep incl. 20).")
    parser.add_argument("--n-perm", type=int, default=999)
    args = parser.parse_args(argv)

    run = init_run(resolve_out(args), config_path=args.config, seed=args.seed)
    cfg = run.config
    mcfg = cfg["model"]
    dgcfg = cfg.get("diagnostics", {})
    lags = tuple(dgcfg.get("ar_lags", LAG_SWEEP))
    q_pca = int(dgcfg.get("granger_pca", 32))

    h5 = args.h5 if args.h5 is not None else Path("data") / mcfg["name"] / "residual_stream.h5"
    rs, _, _ = load_residual_stream(h5)
    dlt = deltas(rs)
    run.log(f"loaded {h5}: deltas {dlt.shape}; depths {args.depths}, lags {list(lags)}, "
            f"q_pca={q_pca}, n_perm={args.n_perm}")

    rows = []
    for depth in args.depths:
        res = granger_permutation_check(
            dlt, depth=depth, lags=lags, q_pca=q_pca, n_perm=args.n_perm, seed=args.seed)
        for r in res:
            run.log(f"  ℓ={r['depth']:2d} lag={r['lag']:2d}: F_obs={r['F_obs']:8.2f} "
                    f"p_param={r['p_parametric']:.2e} p_perm<={r['p_perm']:.4f} "
                    f"KS p={r['ks_p']:.3f} q95 ratio={r['q95_ratio_emp_over_param']:.3f} "
                    f"q99 ratio={r['q99_ratio_emp_over_param']:.3f}")
        rows.extend(res)

    import pandas as pd

    pd.DataFrame(rows).to_csv(run.path("tables/granger_perm_check.csv"), index=False)
    n_floor = sum(r["p_perm"] <= r["p_perm_floor"] for r in rows)
    worst_q99 = max(abs(np.log(r["q99_ratio_emp_over_param"])) for r in rows)
    calibrated = all(0.8 <= r["q95_ratio_emp_over_param"] <= 1.25 for r in rows)
    run.save_json("tables/granger_perm_check.json", {
        "h5": str(h5), "depths": args.depths, "lags": list(lags),
        "q_pca": q_pca, "n_perm": args.n_perm,
        "n_cells": len(rows), "n_obs_at_perm_floor": int(n_floor),
        "tail_calibrated_q95_within_25pct": bool(calibrated),
        "max_abs_log_q99_ratio": float(worst_q99),
        "rows": rows,
        "interpretation": (
            "p_perm is the Freedman–Lane permutation exceedance of the observed Rao F "
            "(floor 1/(n_perm+1)); ks_p / q95 / q99 compare the permutation null of F "
            "against the parametric F(df1, df2) — tail ratios near 1 mean the "
            "parametric threshold used by the Bonferroni count is calibrated under "
            "exchangeability regardless of the Gaussian-error assumption."
        ),
    })

    print("\n=== Granger permutation spot-check ===")
    print(f"  {len(rows)} cells at depths {args.depths}; observed F at the permutation "
          f"floor in {n_floor}/{len(rows)} cells")
    print(f"  tail calibration: q95 ratios all within ±25% -> {calibrated}; "
          f"max |log q99 ratio| = {worst_q99:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
