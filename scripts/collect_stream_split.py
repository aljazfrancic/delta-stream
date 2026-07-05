#!/usr/bin/env python3
"""R8 — collect the attn-out vs MLP-out split of Δ (optional mechanism polish).

Every Δ-based result so far treats the block write as one vector, but the block
writes twice: ``Δ_ℓ = attn_out_ℓ + mlp_out_ℓ`` (exactly — TransformerLens sums
these terms into ``resid_post``; Pythia adds them in parallel, GPT-2 in series).
R8 asks *where the cross-layer AR structure lives* by re-running the H1/H2
diagnostics per stream. This script is the one new collection pass it needs: at
the final prompt token of every filtered example it records both per-layer
writes plus the residual states (same forward pass), verifies additivity, and
writes two **pseudo-stream** stores

    data/<model>-attnstream/residual_stream.h5
    data/<model>-mlpstream/residual_stream.h5

with ``w_0 = v_0`` and ``w_{ℓ+1} = w_ℓ + write_ℓ``, so ``deltas(w)`` in every
downstream script recovers exactly that stream's writes — step5_ar.py,
step1_stationarity.py and step2_kstar.py then run unchanged via ``--h5``.
Note the split streams carry **no ±v_ℓ tautology**: consecutive attn writes
share no mechanical term (unlike Δ_ℓ ⊃ −v_ℓ ⊂ Δ_{ℓ-1}), so their AR numbers
are tautology-free reads by construction.

    python scripts/collect_stream_split.py --config configs/pythia-410m-deduped.yaml \
        --desc r8-collect-streams-410md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from layer_dynamics.activations import (
    build_pseudo_stream,
    collect_stream_writes,
    stream_additivity_stats,
)
from layer_dynamics.cli import add_common_args, resolve_out
from layer_dynamics.device import describe
from layer_dynamics.diagnostics.dataset import deltas
from layer_dynamics.modeling import load_model
from layer_dynamics.runlog import init_run


def _write_store(path: Path, w: np.ndarray, df: pd.DataFrame, *, model_name: str,
                 stream: str, n_layers: int, d_model: int, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    str_dt = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as f:
        f.create_dataset("residual_stream", data=w, compression="gzip", compression_opts=4)
        f.create_dataset("example_id", data=df.index.to_numpy(dtype=np.int64))
        f.create_dataset("relation_class", data=df["relation_class"].to_numpy(), dtype=str_dt)
        f.create_dataset("relation", data=df["relation"].to_numpy(), dtype=str_dt)
        f.attrs.update(
            model=model_name, n_examples=len(df), n_layers=n_layers, d_model=d_model,
            token_position="final_prompt", seed=seed,
            stream=stream, pseudo_stream=True,
            note=("cumulative single-stream pseudo-stream: only deltas() is meaningful; "
                  "w_ell for ell>=1 is NOT a real model state"),
        )


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
    run.log(f"parallel_attn_mlp={getattr(model.cfg, 'parallel_attn_mlp', False)}")

    attn, mlp, resid = collect_stream_writes(
        model, df["prompt"].tolist(), batch_size=int(ccfg.get("batch_size", 8))
    )
    assert attn.shape == mlp.shape == (len(df), n_layers, d_model)
    assert resid.shape == (len(df), n_layers + 1, d_model)
    run.log(f"collected attn/mlp writes {attn.shape} + residuals {resid.shape}")

    # Sanity 1 — additivity against the SAME forward pass: attn + mlp = Δ.
    dlt = deltas(resid)
    stats = stream_additivity_stats(attn, mlp, dlt)
    run.log(f"additivity: max|attn+mlp-Δ|={stats['max_abs_err']:.3e}, "
            f"rel Frobenius={stats['rel_frobenius_err']:.3e}")
    additivity_ok = stats["rel_frobenius_err"] < 1e-3
    if not additivity_ok:
        run.log("WARNING: additivity violated — hook semantics changed, investigate")

    # Sanity 2 — the residual trajectory matches the Phase-1 store (same prompts,
    # same token position; any drift means the streams describe a different run).
    stored_h5 = Path("data") / mcfg["name"] / "residual_stream.h5"
    store_dev = None
    if stored_h5.exists():
        with h5py.File(stored_h5, "r") as f:
            stored = f["residual_stream"][:].astype(np.float32)
        if stored.shape == resid.shape:
            store_dev = float(np.abs(stored - resid).max())
            run.log(f"vs stored {stored_h5}: max|v - v_stored| = {store_dev:.3e}")
        else:
            run.log(f"vs stored {stored_h5}: SHAPE MISMATCH {stored.shape} vs {resid.shape}")

    # Write the two pseudo-stream stores (+ round-trip check on each).
    roundtrip = {}
    for stream, writes in (("attn_out", attn), ("mlp_out", mlp)):
        w = build_pseudo_stream(resid[:, 0], writes)
        rt = float(np.abs(deltas(w) - writes).max())
        roundtrip[stream] = rt
        tag = stream.removesuffix("_out")
        out_h5 = Path("data") / f"{mcfg['name']}-{tag}stream" / "residual_stream.h5"
        _write_store(out_h5, w, df, model_name=mcfg["name"], stream=stream,
                     n_layers=n_layers, d_model=d_model, seed=args.seed)
        run.log(f"wrote {out_h5} (round-trip max|deltas(w) - writes| = {rt:.3e})")

    share = np.array(stats["attn_energy_share"])
    run.log("attn energy share by layer: " +
            " ".join(f"{s:.2f}" for s in share))

    run.save_json("tables/stream_split_stats.json", {
        "n_examples": len(df), "n_layers": n_layers, "d_model": d_model,
        "parallel_attn_mlp": bool(getattr(model.cfg, "parallel_attn_mlp", False)),
        "additivity": {k: stats[k] for k in ("max_abs_err", "rel_frobenius_err")},
        "additivity_ok": additivity_ok,
        "max_dev_vs_stored_residual_stream": store_dev,
        "pseudo_stream_roundtrip_max_abs": roundtrip,
        "mean_norm_attn_by_layer": stats["mean_norm_attn"],
        "mean_norm_mlp_by_layer": stats["mean_norm_mlp"],
        "mean_norm_delta_by_layer": stats["mean_norm_delta"],
        "attn_energy_share_by_layer": stats["attn_energy_share"],
    })

    print("\n=== R8 stream-split collection ===")
    print(f"  additivity rel-Frobenius {stats['rel_frobenius_err']:.2e} "
          f"(ok={additivity_ok}); vs stored store max dev "
          f"{store_dev if store_dev is not None else float('nan'):.2e}")
    print(f"  attn energy share: min {share.min():.2f} / median {np.median(share):.2f} "
          f"/ max {share.max():.2f}")
    print("  next: step5_ar / step1_stationarity / step2_kstar with "
          f"--h5 data/{mcfg['name']}-{{attn,mlp}}stream/residual_stream.h5")
    return 0 if additivity_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
