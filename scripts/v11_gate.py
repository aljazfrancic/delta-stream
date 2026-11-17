#!/usr/bin/env python3
"""Reproduction gates for the v1.1 on-box checks: does a rebuilt artefact match its record?

The checks of ``scripts/run_v11_checks.sh`` start from artefacts that are no longer on
disk (the MLSAE sources, the SSM checkpoints, the untrained-network store and dictionary,
the 12-layer store). Each is rebuilt with its recorded recipe and must reproduce the
tracked table it originally produced before anything is measured on it. One subcommand
per artefact; exit status 0 = reproduced, 1 = not. No run directory is written — the
caller logs the one-line result.

    v11_gate.py sources    <new.npz> <recorded sources_*.json>
    v11_gate.py checkpoint <new run dir> <recorded run dir>
    v11_gate.py knee       <new run dir> <recorded run dir> [--variants ssm baseline]
    v11_gate.py ci         <new run dir> <recorded run dir>
    v11_gate.py prefix     <new run dir> <recorded embed_only.json> [--cutoffs 0]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from layer_dynamics.diagnostics.sae_seed_stability import reproduction_check


def _json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _report(label: str, pairs: list[tuple[str, float, float]], tol: float,
            extra_ok: bool = True) -> int:
    worst = max((abs(a - b) for _, a, b in pairs), default=float("inf"))
    ok = bool(worst <= tol and extra_ok)
    detail = ", ".join(f"{k} {a:.6f} vs {b:.6f}" for k, a, b in pairs[:6])
    print(f"  gate {label}: {'OK' if ok else 'FAILED'}  max |Δ| = {worst:.2e} (tol {tol:g}); "
          f"{detail}{' …' if len(pairs) > 6 else ''}")
    return 0 if ok else 1


def gate_sources(args) -> int:
    ref = _json(args.ref)
    with np.load(args.new) as z:
        alive, r2 = z["alive_count"], z["recon_r2"]
    r = reproduction_check(alive, r2, ref)
    print(f"  gate sources {args.new}: {'OK' if r['ok'] else 'FAILED'}  alive identical: "
          f"{r.get('alive_identical')}, M* {int(alive.max())} vs {ref['M_star']}, mean recon R² "
          f"diff {r.get('r2_mean_abs_diff', float('nan')):.2e}, max per-layer "
          f"{r.get('max_r2_abs_diff', float('nan')):.2e}")
    return 0 if r["ok"] else 1


def gate_checkpoint(args) -> int:
    new = _json(args.new / "tables" / "candidate_pool.json")
    ref = _json(args.ref / "tables" / "candidate_pool.json")
    keys = ("train_r2", "test_r2", "train_l0", "test_l0", "pool_match_rate_0.7")
    pairs = [(k, float(new[k]), float(ref[k])) for k in keys]
    same_pool = ([(c["feature"], c["home_layer"]) for c in new["candidate_pool"]]
                 == [(c["feature"], c["home_layer"]) for c in ref["candidate_pool"]])
    same_stable = new["n_stable"] == ref["n_stable"]
    rc = _report(f"checkpoint {args.new}", pairs, args.tol, extra_ok=same_pool and same_stable)
    print(f"    candidate pool identical: {same_pool}; stable features {new['n_stable']} vs "
          f"{ref['n_stable']}")
    return rc


def _knee_rows(run_dir: Path, lam: float) -> dict[str, dict]:
    with open(run_dir / "tables" / "pareto.csv", encoding="utf-8", newline="") as fh:
        return {row["variant"]: row for row in csv.DictReader(fh)
                if abs(float(row["lambda_l1"]) - lam) < 1e-9}


def gate_knee(args) -> int:
    new, ref = _knee_rows(args.new, args.lam), _knee_rows(args.ref, args.lam)
    pairs = []
    for v in args.variants:
        if v not in new or v not in ref:
            print(f"  gate knee {args.new}: FAILED  no λ={args.lam:g} row for '{v}'")
            return 1
        for k in ("test_r2", "train_r2", "test_l0"):
            pairs.append((f"{v}.{k}", float(new[v][k]), float(ref[v][k])))
    return _report(f"knee {args.new}", pairs, args.tol)


def gate_ci(args) -> int:
    new = _json(args.new / "tables" / "r0_bootstrap_ci.json")["gate3"]
    ref = _json(args.ref / "tables" / "r0_bootstrap_ci.json")["gate3"]
    pairs = [("ssm_test_r2", new["ssm_test_r2"], ref["ssm_test_r2"]),
             ("baseline_test_r2", new["baseline_test_r2"], ref["baseline_test_r2"])]
    pairs += [(k, new["gap_ci"][k], ref["gap_ci"][k]) for k in ("gap", "ci_lo", "ci_hi")]
    return _report(f"ci {args.new}", pairs, args.tol, extra_ok=new["rank"] == ref["rank"])


def gate_prefix(args) -> int:
    new = _json(args.new / "tables" / "embed_only.json")
    ref = _json(args.ref)
    rows_new = {int(r["cutoff"]): r for r in new["rows"]}
    rows_ref = {int(r["cutoff"]): r for r in ref["rows"]}
    pairs = []
    for c in args.cutoffs:
        if c not in rows_new or c not in rows_ref:
            print(f"  gate prefix {args.new}: FAILED  cutoff {c} missing")
            return 1
        for k in ("test_r2", "train_r2"):
            pairs.append((f"ℓmax={c}.{k}", float(rows_new[c][k]), float(rows_ref[c][k])))
        a = np.asarray(new["test_per_layer_r2_by_cutoff"][str(c)], float)
        b = np.asarray(ref["test_per_layer_r2_by_cutoff"][str(c)], float)
        pairs.append((f"ℓmax={c}.per-layer(max)", float(np.abs(a - b).max()), 0.0))
    return _report(f"prefix {args.new}", pairs, args.tol)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, fn in (("sources", gate_sources), ("checkpoint", gate_checkpoint),
                     ("knee", gate_knee), ("ci", gate_ci), ("prefix", gate_prefix)):
        sp = sub.add_parser(name)
        sp.add_argument("new", type=Path)
        sp.add_argument("ref", type=Path)
        sp.add_argument("--tol", type=float, default=1e-4)
        sp.set_defaults(fn=fn)
        if name == "knee":
            sp.add_argument("--variants", nargs="+", default=["ssm", "baseline"])
            sp.add_argument("--lam", type=float, default=1.0)
        if name == "prefix":
            sp.add_argument("--cutoffs", type=int, nargs="+", default=[0])
    args = parser.parse_args(argv)
    try:
        return args.fn(args)
    except (FileNotFoundError, KeyError) as exc:
        print(f"  gate {args.cmd} {args.new}: FAILED  {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
