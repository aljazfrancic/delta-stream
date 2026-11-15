#!/usr/bin/env python3
"""Verify that the activation store on disk is the one the tracked tables were made from.

Written for the v1.1 re-collection: the original ``data/<model>/residual_stream.h5`` and
``filtered_triples.parquet`` no longer exist, so both were rebuilt from the cached model
and dataset with the original scripts. Before any new number is computed on the rebuilt
store, this script checks it against three **tracked** records of the original:

1. **Dataset** — row count, per-class and per-relation counts of
   ``filtered_triples.parquet`` against the recorded dataset run's ``stats.json``
   (and, when the rebuild's own ``stats.json`` is given, its candidate / correct-top-1
   counts too).
2. **Store shape** against the recorded collection run's ``collect_stats.json``.
3. **Fingerprint** — the Phase-1 Step-3 per-layer table (``ridge_r2.csv``) recomputed
   from the store with the same functions, split and seed. Its ``zero`` column is pure
   arithmetic on the test-split updates (no fit), ``identity`` and ``full`` are
   closed-form ridge fits, and every column depends on the **row order** through the
   split — so agreement at float precision means the same activations in the same order.
   The Tuned-Lens column is not recomputed (it needs the optional lens extra and is a
   function of the lens, not only of the store).

    uv run python scripts/verify_store.py --config configs/pythia-410m-deduped.yaml \
        --desc v11-recollect-verify-410md --seed 0 \
        --dataset-ref runs/2026-06-21-ph1-dataset-410md/tables/stats.json \
        --collect-ref runs/2026-06-21-ph1-collect-410md/tables/collect_stats.json \
        --ridge-ref runs/2026-06-21-ph1-step3-410md/tables/ridge_r2.csv \
        --dataset-new runs/<date>-v11-recollect-dataset-410md/tables/stats.json

Exit code 0 only if every check passes (fingerprint: max |Δ| ≤ ``--tol`` in R² units on
every recomputed column).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from layer_dynamics.cli import add_common_args, resolve_out
from layer_dynamics.diagnostics.dataset import deltas, load_residual_stream, train_test_split_idx
from layer_dynamics.diagnostics.lowrank_ridge import fit_lowrank_ridge, identity_r2, pooled_r2
from layer_dynamics.runlog import init_run


def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def ridge_table(rs: np.ndarray, *, ranks: tuple[int, ...], test_frac: float, seed: int,
                log=None) -> pd.DataFrame:
    """The Phase-1 Step-3 per-layer table — same computation as ``scripts/step3_ridge.py``
    (the source of truth), minus the Tuned-Lens column."""
    n, n_states, _ = rs.shape
    dlt = deltas(rs)
    tr, te = train_test_split_idx(n, test_frac=test_frac, seed=seed)
    rows = []
    for ell in range(n_states - 1):
        x_tr, x_te = rs[tr, ell], rs[te, ell]
        y_tr, y_te = dlt[tr, ell], dlt[te, ell]
        res = fit_lowrank_ridge(x_tr, y_tr, x_te, y_te, ranks=ranks)
        ref = np.broadcast_to(y_tr.mean(axis=0), y_te.shape)
        rows.append({
            "layer": ell, **{f"r{r}": res[r] for r in ranks}, "full": res["full"],
            "zero": pooled_r2(y_te, np.zeros_like(y_te), ref),
            "identity": identity_r2(x_tr, rs[tr, ell + 1], x_te, rs[te, ell + 1]),
            "alpha": res["alpha"],
        })
        if log:
            log(f"  layer {ell:2d}: zero={rows[-1]['zero']:+.6f} "
                f"identity={rows[-1]['identity']:.6f} full={rows[-1]['full']:+.6f}")
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--dataset-ref", type=Path, required=True,
                        help="recorded dataset run's tables/stats.json")
    parser.add_argument("--collect-ref", type=Path, required=True,
                        help="recorded collection run's tables/collect_stats.json")
    parser.add_argument("--ridge-ref", type=Path, required=True,
                        help="recorded Step-3 tables/ridge_r2.csv")
    parser.add_argument("--dataset-new", type=Path, default=None,
                        help="the rebuild's own tables/stats.json (optional, compared in full)")
    parser.add_argument("--tol", type=float, default=1e-3,
                        help="max |recomputed − recorded| allowed on any R² column")
    args = parser.parse_args(argv)

    run = init_run(resolve_out(args), config_path=args.config, seed=args.seed)
    mcfg = run.config["model"]
    dgcfg = run.config.get("diagnostics", {})
    data_dir = Path("data") / mcfg["name"]
    checks: dict[str, dict] = {}

    # --- 1. dataset ---
    ref = _load(args.dataset_ref)
    df = pd.read_parquet(data_dir / "filtered_triples.parquet")
    got = {"n_kept": int(len(df)),
           "kept_per_class": {k: int(v) for k, v in df["relation_class"].value_counts().items()},
           "kept_per_relation": {k: int(v) for k, v in df["relation"].value_counts().items()}}
    ds = {k: {"recorded": ref[k], "rebuilt": got[k], "match": ref[k] == got[k]} for k in got}
    if args.dataset_new is not None:
        new = _load(args.dataset_new)
        for k in ("n_candidates", "n_correct_top1", "threshold", "min_examples_per_relation",
                  "dropped_classes"):
            ds[k] = {"recorded": ref[k], "rebuilt": new[k], "match": ref[k] == new[k]}
    checks["dataset"] = {"ok": all(v["match"] for v in ds.values()), "fields": ds}
    run.log(f"[1] dataset: {len(df)} rows, classes {got['kept_per_class']} -> "
            f"{'MATCH' if checks['dataset']['ok'] else 'MISMATCH'} "
            f"({sum(v['match'] for v in ds.values())}/{len(ds)} fields)")

    # --- 2. store shape ---
    cref = _load(args.collect_ref)
    rs, rel_class, _ = load_residual_stream(data_dir / "residual_stream.h5")
    aligned = bool((df["relation_class"].to_numpy() == rel_class).all()) if len(df) == len(
        rel_class) else False
    checks["store_shape"] = {"recorded": cref["shape"], "rebuilt": list(rs.shape),
                             "rows_aligned_to_parquet": aligned,
                             "ok": list(rs.shape) == cref["shape"] and aligned}
    run.log(f"[2] store shape {list(rs.shape)} vs recorded {cref['shape']}, "
            f"aligned to parquet: {aligned} -> "
            f"{'MATCH' if checks['store_shape']['ok'] else 'MISMATCH'}")

    # --- 3. fingerprint ---
    rref = pd.read_csv(args.ridge_ref)
    ranks = tuple(dgcfg.get("ranks", [16, 32, 64, 128]))
    table = ridge_table(rs, ranks=ranks, test_frac=float(dgcfg.get("test_frac", 0.2)),
                        seed=args.seed, log=run.log)
    cols = [c for c in table.columns if c != "layer" and c in rref.columns]
    out = table[["layer"]].copy()
    max_diff = {}
    same_layers = len(table) == len(rref) and (table["layer"].to_numpy()
                                               == rref["layer"].to_numpy()).all()
    if same_layers:
        for c in cols:
            out[f"{c}_recorded"] = rref[c].to_numpy()
            out[f"{c}_rebuilt"] = table[c].to_numpy()
            out[f"{c}_absdiff"] = np.abs(table[c].to_numpy() - rref[c].to_numpy())
            max_diff[c] = float(out[f"{c}_absdiff"].max())
    out.to_csv(run.path("tables/store_fingerprint.csv"), index=False)
    r2_cols = [c for c in cols if c != "alpha"]
    worst = max((max_diff[c] for c in r2_cols), default=float("inf"))
    alpha_same = bool(same_layers and max_diff.get("alpha", 1.0) <= 1e-9)
    checks["fingerprint"] = {
        "columns": cols, "max_abs_diff": max_diff, "worst_r2_abs_diff": worst,
        "ridge_alpha_same": alpha_same, "tol": args.tol,
        "bit_level": bool(worst <= 1e-9), "ok": bool(same_layers and worst <= args.tol),
    }
    run.log(f"[3] fingerprint vs {args.ridge_ref}: max |Δ| per column "
            + ", ".join(f"{c} {max_diff[c]:.2e}" for c in cols) if same_layers
            else "[3] fingerprint: layer sets differ")

    ok = all(c["ok"] for c in checks.values())
    # Content hashes of what was verified, so any later rebuild can be checked directly.
    hashes = {
        "residual_stream_float32_sha256": hashlib.sha256(
            np.ascontiguousarray(rs, dtype=np.float32).tobytes()).hexdigest(),
        "prompts_sha256": hashlib.sha256("\n".join(df["prompt"]).encode()).hexdigest(),
    }
    run.log(f"store sha256 (float32 array bytes): {hashes['residual_stream_float32_sha256']}")
    run.save_json("tables/verify_stats.json", {
        "model": mcfg["name"], "ok": ok, "checks": checks, "hashes": hashes,
        "references": {"dataset": str(args.dataset_ref), "collect": str(args.collect_ref),
                       "ridge": str(args.ridge_ref)},
    })

    print(f"\n=== store verification ({mcfg['name']}) ===")
    print(f"  [1] dataset vs recorded stats:      "
          f"{'OK' if checks['dataset']['ok'] else 'MISMATCH'}")
    print(f"  [2] store shape + row alignment:    "
          f"{'OK' if checks['store_shape']['ok'] else 'MISMATCH'}")
    fp = checks["fingerprint"]
    level = ("bit-level" if fp["bit_level"] else "float-noise" if fp["ok"] else "DIFFERENT")
    print(f"  [3] Step-3 fingerprint (24 layers): {'OK' if fp['ok'] else 'MISMATCH'} — worst "
          f"|ΔR²| {fp['worst_r2_abs_diff']:.2e} ({level}; tolerance {args.tol:g})")
    print("store:", "VERIFIED — matches the tracked records" if ok
          else "NOT VERIFIED — do not compute new numbers on it")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
