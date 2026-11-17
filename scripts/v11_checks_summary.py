#!/usr/bin/env python3
"""Collect the four v1.1 on-box checks into one table of verdict rows (no GPU).

Reads the tables the checks wrote and applies the thresholds of
``runs/v11-onbox-checks/memo.md`` (fixed before any run) through
``layer_dynamics.diagnostics.v11_checks``. A check whose inputs are missing is reported as
``not run``; nothing is inferred for it.

    uv run python scripts/v11_checks_summary.py --config configs/pythia-410m-deduped.yaml \\
        --out runs/<date>-v11c-summary \\
        --probe mlsae=<probe run> --probe l1=<probe run> \\
        --prefix <mean-init sweep run> \\
        --prefix-recorded runs/2026-06-30-ph4-e5-prefix-410md/tables/embed_only.json \\
        --pm-ci <rank-64 CI run, MLSAE> --pm-seed <rank-64 baseline run> <recorded rank-32 run> \\
        --pm-l1-ci <rank-64 CI run, L1> \\
        --pm-untrained <rank-64 baseline run> <rank-32 run> \\
        --pm-shallow <rank-64 baseline run> <recorded rank-32 run> \\
        --transfer <kernel-transfer run>
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from layer_dynamics.cli import add_common_args, resolve_out
from layer_dynamics.diagnostics.v11_checks import (
    PROBE_PRIMARY,
    PROBE_SECONDARY,
    depth_verdict,
    learned_share_verdict,
    paired_gap_verdict,
    prefix_verdict,
    probe_overall,
    seeds_verdict,
)
from layer_dynamics.runlog import init_run

RECORDED_REFIT_SD = 0.0014   # sd of the rank-matched knee gap over three refits (paper §6.1)


def _json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _knee(run_dir: Path, variant: str, lam: float = 1.0) -> float | None:
    f = run_dir / "tables" / "pareto.csv"
    if not f.exists():
        return None
    with open(f, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["variant"] == variant and abs(float(row["lambda_l1"]) - lam) < 1e-9:
                return float(row["test_r2"])
    return None


def _pm_gap(base64_dir: Path, ssm32_dir: Path) -> dict | None:
    """Parameter-matched knee gap: rank-32 SSM − rank-64 per-layer baseline (same split)."""
    ssm, base = _knee(ssm32_dir, "ssm"), _knee(base64_dir, "baseline")
    if ssm is None or base is None:
        return None
    return {"ssm_r32_test_r2": ssm, "baseline_r64_test_r2": base, "gap": ssm - base,
            "rank_matched_gap": (ssm - b32) if (b32 := _knee(ssm32_dir, "baseline")) is not None
            else None,
            "baseline_r64_dir": str(base64_dir), "ssm_r32_dir": str(ssm32_dir)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--probe", action="append", default=[], metavar="NAME=DIR")
    parser.add_argument("--prefix", type=Path, default=None)
    parser.add_argument("--prefix-recorded", type=Path, default=None)
    parser.add_argument("--pm-ci", type=Path, default=None,
                        help="r0_bootstrap_ci run at rank 64 over the MLSAE dictionary (seed 0).")
    parser.add_argument("--pm-seed", action="append", nargs=2, type=Path, default=[],
                        metavar=("BASELINE64", "SSM32"))
    parser.add_argument("--pm-l1-ci", type=Path, default=None)
    parser.add_argument("--pm-untrained", nargs=2, type=Path, default=None,
                        metavar=("BASELINE64", "SSM32"))
    parser.add_argument("--pm-shallow", action="append", nargs=2, type=Path, default=[],
                        metavar=("BASELINE64", "SSM32"))
    parser.add_argument("--transfer", type=Path, default=None)
    args = parser.parse_args(argv)
    run = init_run(resolve_out(args), config_path=args.config, seed=args.seed)
    out: dict = {}

    # --- A: probe ---------------------------------------------------------------------
    probes, verdicts = {}, {}
    for item in args.probe:
        name, _, d = item.partition("=")
        t = _json(Path(d) / "tables" / "state_probe_causal.json")
        if t is None or "primary" not in t:
            probes[name] = {"status": "not run", "dir": d}
            continue
        verdicts[name] = t["verdict"]
        probes[name] = {
            "dir": d, "verdict": t["verdict"], "primary": t["primary"],
            "secondary": {k: t["band_contrasts"][k] for k in PROBE_SECONDARY},
            "recorded_state_minus_prefix_mid":
                t["band_contrasts"]["object/state−code_prefix_pca/mid"],
            "reproduces_recorded": t.get("reproduction", {}).get("ok"),
        }
    complete = bool(probes) and len(verdicts) == len(probes)
    out["A_probe"] = {"primary_contrast": PROBE_PRIMARY, "per_dictionary": probes,
                      "verdict": probe_overall(verdicts) if complete else "not run"}

    # --- B: prefix sweep --------------------------------------------------------------
    new = _json(args.prefix / "tables" / "embed_only.json") if args.prefix else None
    rec = _json(args.prefix_recorded) if args.prefix_recorded else None
    if new is None or rec is None:
        out["B_prefix"] = {"status": "not run"}
    else:
        out["B_prefix"] = {"dir": str(args.prefix), "bias_init": new.get("bias_init"),
                           **prefix_verdict(new["rows"], rec["rows"])}

    # --- C: parameter-matched gap -----------------------------------------------------
    c: dict = {}
    seed0 = _json(args.pm_ci / "tables" / "r0_bootstrap_ci.json") if args.pm_ci else None
    gaps = []
    if seed0 is not None:
        g = seed0["gate3"]
        c["mlsae_seed0"] = {"gap_ci": g["gap_ci"], "rank": g["rank"],
                            "verdict": paired_gap_verdict(g["gap_ci"])}
        gaps.append(g["gap_ci"]["gap"])
    extra = [_pm_gap(b, s) for b, s in args.pm_seed]
    c["mlsae_extra_seeds"] = extra
    gaps += [e["gap"] for e in extra if e is not None]
    if seed0 is not None and len(gaps) == 1 + len(args.pm_seed) and len(gaps) > 1:
        c["mlsae_seeds"] = seeds_verdict(gaps)
        refit_sd = c["mlsae_seeds"]["sd"]
    else:
        c["mlsae_seeds"] = {"status": "not run"}
        refit_sd = RECORDED_REFIT_SD
    l1 = _json(args.pm_l1_ci / "tables" / "r0_bootstrap_ci.json") if args.pm_l1_ci else None
    if l1 is None:
        c["l1"] = {"status": "not run"}
    else:
        g = l1["gate3"]
        c["l1"] = {"gap_ci": g["gap_ci"], "rank": g["rank"],
                   "verdict": paired_gap_verdict(g["gap_ci"])}
    un = _pm_gap(*args.pm_untrained) if args.pm_untrained else None
    if un is None or l1 is None:
        c["untrained"] = {"status": "not run"}
    else:
        c["untrained"] = {**un, **learned_share_verdict(l1["gate3"]["gap_ci"]["gap"], un["gap"],
                                                        refit_sd),
                          "refit_sd_source": ("the MLSAE parameter-matched gaps of this run"
                                              if "sd" in c["mlsae_seeds"] else
                                              "the recorded rank-matched gap (0.0014)")}
    shallow = [_pm_gap(b, s) for b, s in args.pm_shallow]
    if shallow and all(s is not None for s in shallow) and "gaps" in c["mlsae_seeds"]:
        c["depth"] = {"shallow": shallow,
                      **depth_verdict([s["gap"] for s in shallow], c["mlsae_seeds"]["gaps"])}
    else:
        c["depth"] = {"status": "not run"}
    out["C_param_matched"] = c

    # --- D: kernel transfer -----------------------------------------------------------
    t = _json(args.transfer / "tables" / "kernel_transfer.json") if args.transfer else None
    if t is None or t.get("primary") is None:
        out["D_kernel_transfer"] = {"status": "not run"}
    else:
        out["D_kernel_transfer"] = {"dir": str(args.transfer), **t["primary"],
                                    "reproduces_recorded": t["reproduction"]["ok"]}

    run.save_json("tables/checks_summary.json", out)

    def line(label: str, d: dict, key: str = "verdict") -> None:
        print(f"  {label:44s} {d.get(key, d.get('status', '—'))}")

    print("\n=== v1.1 on-box checks ===")
    line("A  state probe without look-ahead", out["A_probe"])
    for name, p in probes.items():
        if "primary" in p:
            pr = p["primary"]
            print(f"       {name}: {pr['diff']:+.4f} [{pr['ci_lo']:+.4f}, {pr['ci_hi']:+.4f}] "
                  f"-> {p['verdict']}")
    b = out["B_prefix"]
    if "embed_only_r2" in b:
        print(f"  B  prefix sweep, mean-initialised bias: embedding-only R² "
              f"{b['embed_only_r2']:+.4f} (at floor: {b['embed_only_at_mean_floor']}); late gain "
              f"{b['late_gain']:+.4f} (load-bearing: {b['late_sources_load_bearing']}); upper "
              f"half stable: {b['upper_half_stable']}")
    else:
        line("B  prefix sweep, mean-initialised bias", b)
    line("C  parameter-matched gap, MLSAE refits", c["mlsae_seeds"])
    line("C  parameter-matched gap, L1 dictionary", c["l1"])
    line("C  untrained share of that gap", c["untrained"])
    line("C  12 vs 24 layers", c["depth"])
    line("D  kernel transfer, trained vs untrained", out["D_kernel_transfer"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
