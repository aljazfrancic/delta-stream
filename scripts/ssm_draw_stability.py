#!/usr/bin/env python3
"""v1.1 E2 (post-hoc) — SSM write-direction stability across dictionary draws.

``scripts/step4_sanity.py`` measures whether the SSM's per-feature write directions are
reproduced when the SSM is re-fit from different initialisations over **one**
dictionary (index-paired match@0.7). v1.0 had one trained dictionary per sparsity
mechanism, so "it is the TopK mechanism" rested on n = 1 per arm. This script tabulates
the same statistic over several independently trained dictionaries (draws) per
mechanism — pure post-processing of each step4 run's tracked
``tables/sanity_stats.json``; nothing is re-fit.

    uv run python scripts/ssm_draw_stability.py --config configs/pythia-410m-deduped.yaml \
        --desc v11-e2-ssm-draw-stability --seed 0 \
        --draw l1:0:runs/2026-06-29-ph4-e2-step4-l1-410md \
        --draw l1:1:runs/<date>-v11-e2-step4-l1-s1-410md \
        --draw l1:2:runs/<date>-v11-e2-step4-l1-s2-410md \
        --draw topk:0:runs/2026-06-30-ph4-e2-step4-topk-410md  ... \
        --check l1:0:runs/<date>-v11-e2-step4-l1-s0-410md

``--draw MECHANISM:DICT_SEED:RUN_DIR`` counts toward the per-mechanism mean and range;
``--check`` rows (a retrained seed 0, which should reproduce the recorded seed-0 run and
is not a new draw) are listed but not counted. The reading, fixed in
``runs/v11-seed-stability/memo.md`` before the runs: the mechanism claim stands at n
draws iff every L1 draw is ≥ 0.7 and every TopK draw is below it; anything else is
"mixed" and reported per draw.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from layer_dynamics.cli import add_common_args, resolve_out
from layer_dynamics.runlog import init_run

BAR = 0.7


def _parse(spec: str) -> tuple[str, int, Path]:
    try:
        mech, seed, path = spec.split(":", 2)
        return mech, int(seed), Path(path)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected MECHANISM:DICT_SEED:RUN_DIR, got {spec!r}") from exc


def draw_row(mech: str, dict_seed: int, run_dir: Path, role: str) -> dict:
    """One step4 run → its SSM-re-seed stability numbers (from the tracked table)."""
    with open(run_dir / "tables" / "sanity_stats.json", encoding="utf-8") as fh:
        s = json.load(fh)
    pairs = [p["match_rate_0.7"] for p in s["seed_stability"]["pairs"]]
    return {
        "mechanism": mech, "dict_seed": dict_seed, "role": role, "run": str(run_dir),
        "match_0.7": float(s["seed_stability"]["mean_match_rate_0.7"]),
        "match_0.7_min_pair": float(min(pairs)), "match_0.7_max_pair": float(max(pairs)),
        "n_ssm_seed_pairs": len(pairs),
        "mean_cos": float(np.mean([p["mean_cos"] for p in s["seed_stability"]["pairs"]])),
        "halves_shared_dir_match_0.7": float(s["dataset_stability"]["shared_dir_match_0.7"]),
        "halves_jaccard": float(s["dataset_stability"]["overlap"]["jaccard"]),
        "train_r2": float(s["main_train"]["r2"]), "d_input": int(s["dims"]["d_input"]),
        "n_layers": int(s["dims"]["n_layers"]),
    }


def summarise(rows: list[dict], bar: float = BAR) -> dict:
    """Per-mechanism mean / range over counted draws, plus the pre-registered reading."""
    out: dict = {"bar": bar, "mechanisms": {}}
    for mech in sorted({r["mechanism"] for r in rows}):
        vals = [r["match_0.7"] for r in rows if r["mechanism"] == mech and r["role"] == "draw"]
        if not vals:
            continue
        out["mechanisms"][mech] = {
            "n_draws": len(vals), "per_draw": vals, "mean": float(np.mean(vals)),
            "min": float(min(vals)), "max": float(max(vals)),
            "all_stable": bool(all(v >= bar for v in vals)),
            "all_unstable": bool(all(v < bar for v in vals)),
        }
    m = out["mechanisms"]
    if {"l1", "topk"} <= set(m):
        stands = m["l1"]["all_stable"] and m["topk"]["all_unstable"]
        out["mechanism_claim"] = (
            f"stands at n = {min(m['l1']['n_draws'], m['topk']['n_draws'])} draws per "
            "mechanism" if stands else "mixed — report per draw and downgrade the sentence")
        out["mechanism_claim_stands"] = bool(stands)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--draw", type=_parse, action="append", default=[],
                        metavar="MECH:SEED:RUN_DIR", help="a counted dictionary draw")
    parser.add_argument("--check", type=_parse, action="append", default=[],
                        metavar="MECH:SEED:RUN_DIR",
                        help="a reproduction check (listed, not counted)")
    args = parser.parse_args(argv)
    if not args.draw:
        parser.error("at least one --draw is required")

    run = init_run(resolve_out(args), config_path=args.config, seed=args.seed)
    rows = [draw_row(*d, role="draw") for d in args.draw]
    rows += [draw_row(*c, role="check") for c in args.check]
    pd.DataFrame(rows).to_csv(run.path("tables/e2_draws.csv"), index=False)
    summary = summarise(rows)
    run.save_json("tables/e2_summary.json", summary)

    print("\n=== v1.1 E2 — SSM write-direction stability across dictionary draws ===")
    for r in rows:
        tag = "" if r["role"] == "draw" else "  (reproduction check, not counted)"
        print(f"  [{r['mechanism']:>4}] dictionary seed {r['dict_seed']}: match@0.7 = "
              f"{r['match_0.7']:.2f} (SSM-seed pairs {r['match_0.7_min_pair']:.2f}–"
              f"{r['match_0.7_max_pair']:.2f}){tag}")
    for mech, m in summary["mechanisms"].items():
        print(f"  {mech}: mean {m['mean']:.2f}, range {m['min']:.2f}–{m['max']:.2f} "
              f"over {m['n_draws']} draw(s)")
    if "mechanism_claim" in summary:
        print(f"  'it is the sparsity mechanism': {summary['mechanism_claim']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
