#!/usr/bin/env python3
"""Collect residual-stream trajectories for the filtered dataset (Phase 1 Step 2).

Reads ``filtered_triples.parquet``, runs the primary model, and stores
``v_0 … v_L`` at the final prompt token for every example into an HDF5 file:

    residual_stream[N, L+1, D]   float32
    example_id[N]                int64   (row index into filtered_triples)
    relation_class[N]            str
    relation[N]                  str

    python scripts/collect_activations.py --config configs/pythia-410m-deduped.yaml \
        --desc ph1-collect-410m

Sanity check (Step 2): for 5 random examples, ln_final + unembed of v_L must
recover the model's top-1 token (matches the parquet's top1_token).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from layer_dynamics.activations import collect_trajectories, recovers_top1
from layer_dynamics.cli import add_common_args, resolve_out
from layer_dynamics.device import describe
from layer_dynamics.modeling import load_model
from layer_dynamics.runlog import init_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args(argv)

    run = init_run(resolve_out(args), config_path=args.config, seed=args.seed)
    cfg = run.config
    mcfg, ccfg = cfg["model"], cfg.get("collection", {})

    parquet = Path("data") / mcfg["name"] / "filtered_triples.parquet"
    df = pd.read_parquet(parquet).reset_index(drop=True)
    run.log(f"loaded {len(df)} examples from {parquet}")

    model = load_model(mcfg["name"], device=mcfg.get("device", "auto"), dtype=mcfg["dtype"])
    run.log(f"model {mcfg['name']} on {describe(next(model.parameters()).device.type)}")
    n_layers, d_model = model.cfg.n_layers, model.cfg.d_model

    traj = collect_trajectories(
        model, df["prompt"].tolist(), batch_size=int(ccfg.get("batch_size", 8))
    )
    assert traj.shape == (len(df), n_layers + 1, d_model)
    run.log(f"collected residual_stream {traj.shape} ({traj.nbytes / 2**20:.0f} MiB)")

    # Sanity: ln_final + unembed of v_L recovers the model's top-1 token.
    rng = np.random.default_rng(args.seed)
    checks = []
    for i in rng.choice(len(df), size=min(5, len(df)), replace=False):
        rec = model.tokenizer.decode([recovers_top1(model, traj[i, -1])])
        expected = df.iloc[i]["top1_token"]
        checks.append((int(i), rec, expected, rec == expected))
    n_ok = sum(c[3] for c in checks)
    for i, rec, exp, ok in checks:
        run.log(f"  sanity ex {i}: ln_final+unembed(v_L)={rec!r} vs top1={exp!r} "
                f"{'OK' if ok else 'MISMATCH'}")
    if n_ok != len(checks):
        run.log("WARNING: sanity check failed — investigate before trusting activations")

    # Write HDF5 (canonical for downstream + provenance copy path logged).
    out_h5 = Path("data") / mcfg["name"] / "residual_stream.h5"
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    str_dt = h5py.string_dtype(encoding="utf-8")
    with h5py.File(out_h5, "w") as f:
        f.create_dataset("residual_stream", data=traj, compression="gzip", compression_opts=4)
        f.create_dataset("example_id", data=df.index.to_numpy(dtype=np.int64))
        f.create_dataset("relation_class", data=df["relation_class"].to_numpy(), dtype=str_dt)
        f.create_dataset("relation", data=df["relation"].to_numpy(), dtype=str_dt)
        f.attrs.update(
            model=mcfg["name"], n_examples=len(df), n_layers=n_layers, d_model=d_model,
            token_position=ccfg.get("token_position", "final_prompt"), seed=args.seed,
        )
    run.log(f"wrote {out_h5}")

    run.save_json("tables/collect_stats.json", {
        "n_examples": len(df), "n_layers": n_layers, "d_model": d_model,
        "shape": list(traj.shape), "mib": round(traj.nbytes / 2**20, 1),
        "sanity_ok": f"{n_ok}/{len(checks)}", "hdf5": str(out_h5),
    })
    return 0 if n_ok == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
