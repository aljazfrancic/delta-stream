#!/usr/bin/env python3
"""Phase 4 Step 1 — Baseline C: the Tuned-Lens implied-Δ reconstruction.

The Tuned Lens learns a per-layer affine translator toward the final residual; its implied
update Δ̂^TL_ℓ = r̂_{ℓ+1} − r̂_ℓ is a (non-feature-producing) reference for how fast information
settles with depth. Gate 1 already established TL ≪ ridge as a Δ predictor; this scores its
held-out Δ reconstruction R² so the Step-1 comparison table is complete. Reuses
:func:`layer_dynamics.diagnostics.tuned_lens.tl_implied_deltas` (lens translators only; the
HF model is loaded just to attach the public lens).

    python scripts/phase4_baseline_lens.py --config configs/pythia-410m-deduped.yaml \
        --desc ph4-baseline-lens-410md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from layer_dynamics.cli import add_common_args, resolve_out
from layer_dynamics.diagnostics.dataset import deltas, load_residual_stream, train_test_split_idx
from layer_dynamics.diagnostics.tuned_lens import load_lens, tl_implied_deltas
from layer_dynamics.runlog import init_run


def _r2(delta: np.ndarray, pred: np.ndarray) -> float:
    mean = delta.mean(axis=0, keepdims=True)
    ss_res = float(((delta - pred) ** 2).sum())
    ss_tot = float(((delta - mean) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_args(parser)
    parser.add_argument("--test-frac", type=float, default=0.2)
    args = parser.parse_args(argv)

    run = init_run(resolve_out(args), config_path=args.config, seed=args.seed)
    mcfg = run.config.get("model", {})
    resource_id = run.config.get("tuned_lens", {}).get("resource_id", mcfg["name"])

    rs, _, _ = load_residual_stream(Path("data") / mcfg["name"] / "residual_stream.h5")
    delta = deltas(rs)
    n = rs.shape[0]
    run.log(f"loaded v {rs.shape}; lens {resource_id}")

    lens = load_lens(resource_id)
    delta_hat = tl_implied_deltas(lens, rs)
    _, test_idx = train_test_split_idx(n, test_frac=args.test_frac, seed=args.seed)
    r2_full = _r2(delta, delta_hat)
    r2_test = _r2(delta[test_idx], delta_hat[test_idx])
    run.log(f"Tuned-Lens implied-Δ R²: full {r2_full:+.3f} | test {r2_test:+.3f}")

    run.save_json("tables/baseline_lens.json", {
        "model": mcfg.get("name"), "baseline": "tuned_lens_implied_delta",
        "resource_id": resource_id, "delta_r2_pooled_full": r2_full,
        "delta_r2_pooled_test": r2_test,
        "note": "non-feature baseline; settles-with-depth reference (Gate 1: TL ≪ ridge on Δ).",
    })
    print(f"\n=== Phase 4 Baseline C — Tuned Lens implied Δ ({mcfg.get('name')}) ===")
    print(f"  implied-Δ R² (held-out test): {r2_test:+.3f}  [SSM +0.857, per-layer-dict +0.816]")
    print(f"  implied-Δ R² (pooled full):   {r2_full:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
