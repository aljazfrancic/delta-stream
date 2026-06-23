#!/usr/bin/env python3
"""Collect residual-stream trajectories at four token positions (Phase 2 Step 3).

Phase 1 Step 2 collected ``v_0 … v_L`` at the final prompt token only. The
token-position-invariance test needs the same trajectories at three more
positions per example — the first subject token, the last subject token, and a
mid-prompt non-subject token — so the AR(k*) kernel can be refit at each and
compared (is the depth dynamics position-invariant, or query-position-specific?).

Positions are resolved in content-token coordinates from the fast tokenizer's
offset mapping (see :mod:`layer_dynamics.token_positions`); one forward pass per
batch reads all four. Output HDF5 mirrors the Step 2 layout with an extra
position axis:

    residual_stream[N, P=4, L+1, D]  float32   (P order: subj_first, subj_last, mid, final)
    token_index[N, P]                int64      absolute index read (BOS at 0)
    valid[N, P]                      bool       false where a position is a placeholder
    multi_token_subject[N]           bool
    example_id[N], relation_class[N], relation[N]

    python scripts/collect_multipos.py --config configs/pythia-410m-deduped.yaml \
        --desc ph2-step3-collect-410md

Sanity checks: (1) the final-position trajectory must match the Phase 1
``residual_stream.h5`` (same token, same model); (2) ln_final + unembed of the
final-position v_L recovers the model's top-1 token.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from layer_dynamics.activations import collect_trajectories_at_positions, recovers_top1
from layer_dynamics.cli import add_common_args, resolve_out
from layer_dynamics.device import describe
from layer_dynamics.modeling import load_model
from layer_dynamics.runlog import init_run
from layer_dynamics.token_positions import (
    POSITION_NAMES,
    resolve_probe_positions,
    subject_char_span,
    token_span_from_offsets,
)


def _resolve_all(df: pd.DataFrame, tokenizer, log) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resolve absolute probe indices, validity, and the multi-token-subject flag.

    Returns ``(abs_pos[N, 4], valid[N, 4], multi[N])``. Absolute index = content
    index + 1 (BOS at 0). Invalid/unresolved positions are placeholdered to the
    final token and flagged ``valid=False``.
    """
    n, p = len(df), len(POSITION_NAMES)
    abs_pos = np.zeros((n, p), dtype=np.int64)
    valid = np.zeros((n, p), dtype=bool)
    multi = np.zeros(n, dtype=bool)
    n_no_span = n_no_mid = 0

    for row in df.itertuples(index=True):
        i = row.Index
        enc = tokenizer(row.prompt, add_special_tokens=False, return_offsets_mapping=True)
        offsets = enc["offset_mapping"]
        final = len(offsets) - 1
        # Default everything to the final token (a guaranteed-valid placeholder).
        abs_pos[i, :] = final + 1
        span_chars = subject_char_span(row.prompt, row.subject, row.relation_template)
        span = token_span_from_offsets(offsets, *span_chars) if span_chars else None
        if span is None:
            n_no_span += 1
            valid[i, POSITION_NAMES.index("final")] = True
            continue
        pos = resolve_probe_positions(offsets, span)
        multi[i] = pos["multi_token_subject"]
        for pj, name in enumerate(POSITION_NAMES):
            idx = pos[name]
            if pos["valid"][name] and idx is not None:
                abs_pos[i, pj] = idx + 1  # +1 for BOS
                valid[i, pj] = True
        if not pos["valid"]["mid"]:
            n_no_mid += 1

    log(f"resolved positions: {n} examples; {n_no_span} with no locatable subject span, "
        f"{n_no_mid} with no mid non-subject token; multi-token subjects {int(multi.sum())}, "
        f"single-token {int((~multi).sum())}")
    return abs_pos, valid, multi


def main(argv: list[str] | None = None) -> int:
    import argparse

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

    abs_pos, valid, multi = _resolve_all(df, model.tokenizer, run.log)

    traj = collect_trajectories_at_positions(
        model, df["prompt"].tolist(), abs_pos, batch_size=int(ccfg.get("batch_size", 8))
    )
    n_pos = len(POSITION_NAMES)
    assert traj.shape == (len(df), n_pos, n_layers + 1, d_model)
    run.log(f"collected residual_stream {traj.shape} ({traj.nbytes / 2**20:.0f} MiB)")

    # Sanity 1: the final-position trajectory must match the Phase 1 single-position
    # collection (same token, same model) on a few random rows.
    final_j = POSITION_NAMES.index("final")
    ref_h5 = Path("data") / mcfg["name"] / "residual_stream.h5"
    rng = np.random.default_rng(args.seed)
    max_abs_diff = None
    if ref_h5.exists():
        with h5py.File(ref_h5, "r") as f:
            ref = f["residual_stream"]
            idx = sorted(rng.choice(len(df), size=min(20, len(df)), replace=False).tolist())
            diffs = [float(np.max(np.abs(traj[i, final_j] - ref[i]))) for i in idx]
        max_abs_diff = max(diffs)
        run.log(f"  sanity1 final-position vs Phase-1 h5: max|Δ|={max_abs_diff:.2e} "
                f"over {len(idx)} rows")
    else:
        run.log("  sanity1 skipped: Phase-1 residual_stream.h5 not found")

    # Sanity 2: ln_final + unembed of the final-position v_L recovers top-1.
    checks = []
    for i in rng.choice(len(df), size=min(5, len(df)), replace=False):
        rec = model.tokenizer.decode([recovers_top1(model, traj[i, final_j, -1])])
        expected = df.iloc[i]["top1_token"]
        checks.append((int(i), rec == expected))
    n_ok = sum(ok for _, ok in checks)
    run.log(f"  sanity2 ln_final+unembed recovers top-1: {n_ok}/{len(checks)}")

    out_h5 = Path("data") / mcfg["name"] / "residual_stream_multipos.h5"
    str_dt = h5py.string_dtype(encoding="utf-8")
    with h5py.File(out_h5, "w") as f:
        f.create_dataset("residual_stream", data=traj, compression="gzip", compression_opts=4)
        f.create_dataset("token_index", data=abs_pos)
        f.create_dataset("valid", data=valid)
        f.create_dataset("multi_token_subject", data=multi)
        f.create_dataset("example_id", data=df.index.to_numpy(dtype=np.int64))
        f.create_dataset("relation_class", data=df["relation_class"].to_numpy(), dtype=str_dt)
        f.create_dataset("relation", data=df["relation"].to_numpy(), dtype=str_dt)
        f.attrs.update(
            model=mcfg["name"], n_examples=len(df), n_layers=n_layers, d_model=d_model,
            position_names=list(POSITION_NAMES), seed=args.seed,
        )
    run.log(f"wrote {out_h5}")

    run.save_json("tables/collect_multipos_stats.json", {
        "n_examples": len(df), "n_positions": n_pos, "position_names": list(POSITION_NAMES),
        "shape": list(traj.shape), "mib": round(traj.nbytes / 2**20, 1),
        "multi_token_subject": int(multi.sum()), "single_token_subject": int((~multi).sum()),
        "valid_per_position": {POSITION_NAMES[j]: int(valid[:, j].sum()) for j in range(n_pos)},
        "final_vs_phase1_max_abs_diff": max_abs_diff,
        "sanity_top1": f"{n_ok}/{len(checks)}", "hdf5": str(out_h5),
    })
    ok = n_ok == len(checks) and (max_abs_diff is None or max_abs_diff < 1e-3)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
