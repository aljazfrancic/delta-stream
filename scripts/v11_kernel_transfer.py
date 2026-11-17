#!/usr/bin/env python3
"""v1.1 check D — does the AR kernel transfer between the trained and untrained network?

T4 (paper §3.4) compares levels: on the primary model the untrained network's
persistence-stripped AR(1) ``R²`` exceeds the trained network's. That shows predictability
of this size needs no training; it does not show the trained network's kernel is the
untrained network's. This script fits the pooled AR(1) kernel on each network (the
recorded fit, so the within-network numbers must reproduce the tracked table) and scores
it on the other, with per-depth intercepts refit on the target — the Phase-2 transfer
test with the two networks in place of two depth bands. Both stores hold the same 1,555
prompts, so the split is shared.

Run on the raw updates (``m = 0``) and with each network's own top-``m`` persistent state
directions removed (``m = 50``, the T1 setting). The primary statistic, fixed in
``runs/v11-onbox-checks/memo.md`` before any run, is the retention
``R²(untrained → trained) / R²(trained → trained)`` at ``m = 50``, rank 128.

    uv run python scripts/v11_kernel_transfer.py --config configs/pythia-410m-deduped.yaml \\
        --out runs/<date>-v11c-kernel-transfer-410md \\
        --trained data/pythia-410m-deduped/residual_stream.h5 \\
        --untrained data/pythia-410m-deduped-untrained/residual_stream.h5 \\
        --recorded-csv runs/2026-07-09-r13-common-depth-pressure/tables/t1_common_depth.csv \\
        --recorded-labels 410m-factual-trained 410m-factual-untrained
"""

from __future__ import annotations

import argparse
from pathlib import Path

from layer_dynamics.cli import add_common_args, resolve_out
from layer_dynamics.diagnostics.autoregress import (
    eval_kernel_on_band,
    feasible_depths,
    fit_ar_pooled,
    remove_state_dirs,
    scramble_per_depth,
    state_pca_basis,
)
from layer_dynamics.diagnostics.dataset import deltas, load_residual_stream, train_test_split_idx
from layer_dynamics.diagnostics.kernel_transfer import classify_transfer, transfer_table
from layer_dynamics.runlog import init_run

RANKS = (16, 32, 64, 128)
PRIMARY_M, PRIMARY_RANK = 50, 128


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--trained", type=Path, required=True)
    parser.add_argument("--untrained", type=Path, required=True)
    parser.add_argument("--m-sweep", type=int, nargs="+", default=[0, 50])
    parser.add_argument("--recorded-csv", type=Path, default=None,
                        help="Tracked t1_common_depth.csv the within-network fits must "
                             "reproduce.")
    parser.add_argument("--recorded-labels", type=str, nargs=2, default=None,
                        metavar=("TRAINED", "UNTRAINED"),
                        help="The two store labels in --recorded-csv.")
    parser.add_argument("--tol", type=float, default=1e-6)
    args = parser.parse_args(argv)
    run = init_run(resolve_out(args), config_path=args.config, seed=args.seed)

    rs = {}
    for name, h5 in (("trained", args.trained), ("untrained", args.untrained)):
        rs[name], _, _ = load_residual_stream(h5)
        run.log(f"{name}: {h5} {rs[name].shape}")
    if rs["trained"].shape != rs["untrained"].shape:
        raise SystemExit("the two stores must hold the same prompts (shapes differ)")
    n = rs["trained"].shape[0]
    tr, te = train_test_split_idx(n, test_frac=0.2, seed=args.seed)
    dlt = {name: deltas(x) for name, x in rs.items()}
    k = 1
    depths = feasible_depths(dlt["trained"].shape[1], k)

    recorded = {}
    if args.recorded_csv is not None:
        import pandas as pd

        ref = pd.read_csv(args.recorded_csv)
        for name, label in zip(("trained", "untrained"), args.recorded_labels, strict=True):
            for _, row in ref[ref["store"] == label].iterrows():
                recorded[(name, int(row["m_removed"]))] = float(row["real_r2"])

    by_m, repro = {}, []
    for m in args.m_sweep:
        stripped = {}
        for name in dlt:
            _, dirs = state_pca_basis(rs[name], m)
            stripped[name] = remove_state_dirs(dlt[name], dirs) if m > 0 else dlt[name]
        table = transfer_table(stripped, tr, te, k=k, ranks=RANKS, input_pcs=256)
        # scrambled-source null: a kernel fit on example-scrambled updates, applied across
        null = {}
        for fit_name, eval_name in (("untrained", "trained"), ("trained", "untrained")):
            s_tr = scramble_per_depth(stripped[fit_name][tr], 1234)
            s_te = scramble_per_depth(stripped[fit_name][te], 2011)
            fit = fit_ar_pooled(s_tr, s_te, k, ranks=RANKS, input_pcs=256)
            null[f"{fit_name}->{eval_name}"] = eval_kernel_on_band(
                fit.coef_at_rank(PRIMARY_RANK), stripped[eval_name][tr], stripped[eval_name][te],
                depths, k, fit.basis)
        table["scrambled_null_transfer_r2"] = null
        by_m[str(m)] = table
        for name in dlt:
            got = table["fits"][name]["best_r2"]
            want = recorded.get((name, m))
            if want is not None:
                repro.append({"store": name, "m": m, "r2": got, "recorded_r2": want,
                              "abs_diff": abs(got - want)})
        p = table["by_rank"][str(PRIMARY_RANK)]
        run.log(f"m={m:2d} rank {PRIMARY_RANK}: within trained "
                f"{p['r2']['trained']['trained']:+.4f} "
                f"untrained {p['r2']['untrained']['untrained']:+.4f} | untrained→trained "
                f"{p['r2']['untrained']['trained']:+.4f} trained→untrained "
                f"{p['r2']['trained']['untrained']:+.4f} | retention "
                f"{p['retention']['untrained->trained']:+.3f} / "
                f"{p['retention']['trained->untrained']:+.3f}")

    prim = by_m[str(PRIMARY_M)]["by_rank"][str(PRIMARY_RANK)] if str(PRIMARY_M) in by_m else None
    stats = {
        "trained": str(args.trained), "untrained": str(args.untrained), "k": k,
        "n_depths": len(depths), "n_train": int(len(tr)), "n_test": int(len(te)),
        "ranks": list(RANKS), "m_sweep": list(args.m_sweep), "by_m": by_m,
        "primary": None if prim is None else {
            "m": PRIMARY_M, "rank": PRIMARY_RANK,
            "retention_untrained_to_trained": prim["retention"]["untrained->trained"],
            "retention_trained_to_untrained": prim["retention"]["trained->untrained"],
            "retention_gain_refit_untrained_to_trained":
                prim["retention_gain_refit"]["untrained->trained"],
            "retention_gain_refit_trained_to_untrained":
                prim["retention_gain_refit"]["trained->untrained"],
            "frobenius": prim["frobenius"]["trained-untrained"],
            "verdict": classify_transfer(prim["retention"]["untrained->trained"],
                                         prim["retention"]["trained->untrained"]),
        },
        "reproduction": {"rows": repro, "tol": args.tol,
                         "ok": bool(repro) and all(r["abs_diff"] <= args.tol for r in repro)},
        "note": ("Each kernel is the recorded pooled AR(1) fit on its own network (own input "
                 "PCA basis), scored on the other network with per-depth intercepts refit "
                 "there. retention[a->b] = R²(fit a, eval b) / R²(fit b, eval b). The "
                 "gain-refit score additionally refits one scalar per target depth. m = "
                 "number of each network's own top state directions removed (T1)."),
    }
    run.save_json("tables/kernel_transfer.json", stats)
    import pandas as pd

    rows = []
    for m, table in by_m.items():
        for r, cell in table["by_rank"].items():
            for a in cell["r2"]:
                for b in cell["r2"][a]:
                    rows.append({"m_removed": int(m), "rank": int(r), "fit": a, "eval": b,
                                 "r2": cell["r2"][a][b],
                                 "r2_gain_refit": cell["r2_gain_refit"][a][b]})
    pd.DataFrame(rows).to_csv(run.path("tables/kernel_transfer.csv"), index=False)

    print("\n=== v1.1 kernel transfer, trained vs untrained ===")
    for r in repro:
        print(f"  within {r['store']:9s} m={r['m']:2d}: R²={r['r2']:.6f} "
              f"(recorded {r['recorded_r2']:.6f}, |Δ|={r['abs_diff']:.1e})")
    if prim is not None:
        pr = stats["primary"]
        print(f"  primary (m={PRIMARY_M}, rank {PRIMARY_RANK}): retention untrained→trained "
              f"{pr['retention_untrained_to_trained']:+.3f}, trained→untrained "
              f"{pr['retention_trained_to_untrained']:+.3f} -> {pr['verdict']}")
    if args.recorded_csv is not None and not stats["reproduction"]["ok"]:
        print("  within-network fits do NOT reproduce the tracked table")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
