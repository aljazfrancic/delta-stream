#!/usr/bin/env python3
"""T4 — untrained-model control for the H1 headline (Phase 4 H1-validity).

T1–T3 hardened the AR-on-Δ headline against the ±v_ℓ *tautology*. T4 closes the
orthogonal *plumbing* objection: a residual stream is a running sum (with
LayerNorm), so *any* such architecture — trained or not — may show some AR-on-Δ
structure. This script collects the residual stream of a **randomly-initialised**
Pythia-410m (same architecture, random weights, forward passes only — no SAEs) on
the *same* prompts/windows as the trained store, so the only thing that changes is
trained-vs-random weights.

The analysis is then the *identical* ``scripts/phase4_pressure_tests.py --h5`` run
on the resulting store; the T1 persistence-stratified curve is the headline read.
If the trained model's persistence-removed R² (T1 at m=50: **0.44** factual / 0.29
generic) far exceeds the untrained model's, H1's depth structure is a product of
**learning, not plumbing**.

    # factual store (same 1555 CounterFact prompts as data/<model>/residual_stream.h5):
    python scripts/phase4_t4_untrained.py --config configs/pythia-410m-deduped.yaml \
        --desc ph4-t4-untrained-410md --source factual
    # generic store (same length-matched Pile windows as the -general store):
    python scripts/phase4_t4_untrained.py --config configs/pythia-410m-deduped.yaml \
        --desc ph4-t4-untrained-410md-general --source generic

Writes ``data/<model>-untrained{,-general}/residual_stream.h5`` (same schema as the
trained stores). Then run ``phase4_pressure_tests.py --h5`` on each.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from layer_dynamics.activations import (
    collect_trajectories,
    collect_trajectories_from_ids,
    recovers_top1,
)
from layer_dynamics.cli import add_common_args, resolve_out
from layer_dynamics.data.general_text import length_matched_windows
from layer_dynamics.device import describe
from layer_dynamics.modeling import load_untrained_model
from layer_dynamics.runlog import init_run

PILE = "NeelNanda/pile-10k"


def _content_lengths(model, prompts: list[str]) -> list[int]:
    return [len(model.tokenizer(p, add_special_tokens=False)["input_ids"]) for p in prompts]


def _pile_doc_ids(model, n_docs: int, char_cap: int, tok_cap: int) -> list[list[int]]:
    """Tokenize the first ``n_docs`` Pile docs to content-id lists (capped length)."""
    from datasets import load_dataset

    ds = load_dataset(PILE, split="train")
    docs: list[list[int]] = []
    for i in range(min(n_docs, len(ds))):
        ids = model.tokenizer(ds[i]["text"][:char_cap], add_special_tokens=False)["input_ids"]
        ids = ids[:tok_cap]
        if ids:
            docs.append(list(ids))
    return docs


def _plumbing_sanity(run, model, traj: np.ndarray) -> dict:
    """Untrained nets do not recall facts, so the trained top-1 check is N/A. Validate
    instead that the extraction is sound: residuals finite, the running sum grows with
    depth, and v_L still decodes to *some* token (the pipeline runs, no correctness)."""
    finite = bool(np.isfinite(traj).all())
    norms = np.linalg.norm(traj, axis=2).mean(axis=0)  # mean ‖v_ℓ‖ by depth
    decoded = model.tokenizer.decode([recovers_top1(model, traj[0, -1])])
    run.log(f"  plumbing sanity: finite={finite}  mean‖v_0‖={norms[0]:.2f} "
            f"mean‖v_L‖={norms[-1]:.2f}  v_L decodes→{decoded!r} (no recall expected)")
    if not finite:
        run.log("WARNING: non-finite residuals from the untrained model — investigate")
    return {"finite": finite, "norm_v0": float(norms[0]), "norm_vL": float(norms[-1]),
            "vL_decodes_to": decoded}


def _collect_factual(run, model, mcfg, ccfg) -> tuple[np.ndarray, pd.DataFrame]:
    parquet = Path("data") / mcfg["name"] / "filtered_triples.parquet"
    df = pd.read_parquet(parquet).reset_index(drop=True)
    run.log(f"loaded {len(df)} factual prompts from {parquet}")
    traj = collect_trajectories(model, df["prompt"].tolist(),
                                batch_size=int(ccfg.get("batch_size", 8)))
    return traj, df


def _collect_generic(run, model, mcfg, ccfg, n_docs, seed) -> tuple[np.ndarray, list[list[int]]]:
    """Rebuild the *same* length-matched Pile windows as the trained -general store
    (same tokenizer, doc pool and seed → identical windows; only the weights differ)."""
    parquet = Path("data") / mcfg["name"] / "filtered_triples.parquet"
    df = pd.read_parquet(parquet).reset_index(drop=True)
    target_lengths = _content_lengths(model, df["prompt"].tolist())
    docs = _pile_doc_ids(model, n_docs, char_cap=2000, tok_cap=256)
    windows = length_matched_windows(docs, target_lengths, seed=seed)
    assert [len(w) for w in windows] == target_lengths
    run.log(f"rebuilt {len(windows)} length-matched generic windows from {len(docs)} Pile docs")
    traj = collect_trajectories_from_ids(model, windows,
                                         batch_size=int(ccfg.get("batch_size", 8)))
    return traj, windows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--source", choices=["factual", "generic"], default="factual",
                        help="Which prompt set to re-collect under random weights.")
    parser.add_argument("--n-docs", type=int, default=4000,
                        help="Pile documents to tokenize as the window pool (generic only).")
    args = parser.parse_args(argv)

    run = init_run(resolve_out(args), config_path=args.config, seed=args.seed)
    cfg = run.config
    mcfg, ccfg = cfg["model"], cfg.get("collection", {})

    model = load_untrained_model(mcfg["name"], device=mcfg.get("device", "auto"),
                                 dtype=mcfg["dtype"], seed=args.seed)
    run.log(f"UNTRAINED {mcfg['name']} (random init, seed={args.seed}) on "
            f"{describe(next(model.parameters()).device.type)}")
    n_layers, d_model = model.cfg.n_layers, model.cfg.d_model

    if args.source == "factual":
        traj, df = _collect_factual(run, model, mcfg, ccfg)
        rel_class = df["relation_class"].to_numpy()
        relation = df["relation"].to_numpy()
        example_id = df.index.to_numpy(dtype=np.int64)
        suffix, token_position = "-untrained", "final_prompt"
        extra: dict = {}
    else:
        traj, windows = _collect_generic(run, model, mcfg, ccfg, args.n_docs, args.seed)
        n = len(windows)
        rel_class = np.array(["general"] * n, dtype=object)
        relation = np.array(["general"] * n, dtype=object)
        example_id = np.arange(n, dtype=np.int64)
        suffix, token_position = "-untrained-general", "final_generic_token"
        extra = {"window_len": np.array([len(w) for w in windows], dtype=np.int64)}

    assert traj.shape == (len(example_id), n_layers + 1, d_model)
    run.log(f"collected residual_stream {traj.shape} ({traj.nbytes / 2**20:.0f} MiB)")
    sanity = _plumbing_sanity(run, model, traj)

    out_h5 = Path("data") / f"{mcfg['name']}{suffix}" / "residual_stream.h5"
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    str_dt = h5py.string_dtype(encoding="utf-8")
    with h5py.File(out_h5, "w") as f:
        f.create_dataset("residual_stream", data=traj, compression="gzip", compression_opts=4)
        f.create_dataset("example_id", data=example_id)
        f.create_dataset("relation_class", data=rel_class, dtype=str_dt)
        f.create_dataset("relation", data=relation, dtype=str_dt)
        for key, val in extra.items():
            f.create_dataset(key, data=val)
        f.attrs.update(
            model=mcfg["name"], weights="untrained_random_init", source=args.source,
            n_examples=len(example_id), n_layers=n_layers, d_model=d_model,
            token_position=token_position, seed=args.seed,
        )
    run.log(f"wrote {out_h5}")

    run.save_json("tables/t4_collect_stats.json", {
        "source": args.source, "weights": "untrained_random_init", "seed": args.seed,
        "n_examples": int(len(example_id)), "n_layers": n_layers, "d_model": d_model,
        "shape": list(traj.shape), "mib": round(traj.nbytes / 2**20, 1),
        "sanity": sanity, "hdf5": str(out_h5),
        "next": f"phase4_pressure_tests.py --h5 {out_h5}",
    })
    print(f"\n=== T4 untrained collection ({args.source}) ===")
    print(f"  wrote {out_h5}  shape={traj.shape}  finite={sanity['finite']}")
    print(f"  next: python scripts/phase4_pressure_tests.py --config {args.config} "
          f"--desc ph4-t4-pressure{suffix.replace('-untrained','')} --h5 {out_h5}")
    return 0 if sanity["finite"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
