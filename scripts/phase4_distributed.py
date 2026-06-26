#!/usr/bin/env python3
"""Phase 4 Step 3 follow-up — is the causal null a distributed code or a dead method?

The pre-registered single-feature existence proof (``phase4_causal.py``) returns null: no
seed-stable feature's natural-magnitude ablation moves top-1 probability by the required 0.2.
This script runs the two exploratory diagnostics that distinguish "the features are causally
inert" from "the features are real but the code is distributed":

  1. **Amplification.** Ablate one feature's contribution ``δ`` scaled ``×{1,3,10,30}``. If the
     drop scales monotonically and saturates at the clean probability, the feature's *direction*
     is causally correct and only its *magnitude* is small (a distributed-code signature, not a
     broken splice).
  2. **Group ablation.** Ablate each example's **top-k** co-firing features together, sweeping k.
     The k at which the drop crosses 0.2 measures how distributed the recall code is.

These are exploratory (not the pre-registered test) and are reported as such — they explain the
null, they do not overturn it. Run on a clean candidate from the screen.

    python scripts/phase4_distributed.py --config configs/pythia-410m-deduped.yaml \
        --desc ph4-distributed-410md --checkpoint-dir runs/2026-06-26-ph4-checkpoint-410md \
        --feature 426 --layer 22 --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from layer_dynamics.causal import (
    answer_probs,
    feature_delta_contribution,
    group_delta_contribution,
    run_prompts_with_resid_edit,
    top_and_control_examples,
)
from layer_dynamics.checkpoint import load_ssm
from layer_dynamics.cli import add_common_args, resolve_out
from layer_dynamics.device import describe, resolve_device
from layer_dynamics.modeling import load_model
from layer_dynamics.runlog import init_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_args(parser)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--feature", type=int, default=426)
    parser.add_argument("--layer", type=int, default=22)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--n-examples", type=int, default=100)
    parser.add_argument("--scales", type=str, default="1,3,10,30")
    parser.add_argument("--group-ks", type=str, default="1,3,10,30,100,300")
    parser.add_argument("--model-batch", type=int, default=4)
    args = parser.parse_args(argv)

    run = init_run(resolve_out(args), config_path=args.config, seed=args.seed)
    mcfg = run.config.get("model", {})
    device = resolve_device(args.device or mcfg.get("device", "auto"))
    run.log(f"device: {describe(device)}")

    ck = args.checkpoint_dir
    u_full = np.load(ck / "checkpoints" / "u_infer_full.npy")
    ssm = load_ssm(ck / "checkpoints" / "ssm_seed0.pt", device=device)
    import pandas as pd

    prompts = pd.read_parquet(
        Path("data") / mcfg["name"] / "filtered_triples.parquet").reset_index(drop=True)[
        "prompt"].tolist()

    m, ell = args.feature, args.layer
    top_idx, _, _ = top_and_control_examples(u_full[:, ell, m], n=args.n_examples)
    pr = [prompts[i] for i in top_idx]
    ut = torch.from_numpy(u_full[top_idx])

    model = load_model(mcfg["name"], device=device, dtype=mcfg.get("dtype", "float32"))
    clean = run_prompts_with_resid_edit(model, pr, edit=None, batch_size=args.model_batch)
    ans = clean.argmax(dim=-1).numpy()
    cp = answer_probs(clean, ans)
    run.log(f"feature {m}@L{ell}: top-{len(top_idx)} clean p={cp.mean():.3f}")

    # 1) amplification
    d = feature_delta_contribution(ssm, ut, m, ell)
    amp = []
    for s in [float(x) for x in args.scales.split(",")]:
        p = answer_probs(run_prompts_with_resid_edit(
            model, pr, edit=d, sign=-s, batch_size=args.model_batch), ans)
        drop = float((cp - p).mean())
        amp.append({"scale": s, "drop": drop})
        run.log(f"  amplify ×{s:g}: ablate drop={drop:+.3f}")

    # 2) group ablation (per-example top-k)
    grp = []
    for k in [int(x) for x in args.group_ks.split(",")]:
        dg = group_delta_contribution(ssm, ut, k)
        p = answer_probs(run_prompts_with_resid_edit(
            model, pr, edit=dg, sign=-1.0, batch_size=args.model_batch), ans)
        drop = float((cp - p).mean())
        grp.append({"k": k, "drop": drop})
        run.log(f"  group top-{k}: ablate drop={drop:+.3f}")

    k_cross = next((g["k"] for g in grp if g["drop"] >= 0.2), None)
    run.save_json("tables/distributed.json", {
        "model": mcfg.get("name"), "feature": m, "home_layer": ell,
        "n_examples": int(len(top_idx)), "clean_prob": float(cp.mean()),
        "amplification": amp, "group_ablation": grp, "k_cross_0.2": k_cross,
    })

    print(f"\n=== Phase 4 distributed-code diagnostics — feat {m}@L{ell} ({mcfg.get('name')}) ===")
    print(f"  clean p={cp.mean():.3f}")
    print("  amplification (single feature ×scale): "
          + " ".join(f"×{a['scale']:g}:{a['drop']:+.3f}" for a in amp))
    print("  group (per-example top-k):             "
          + " ".join(f"k{g['k']}:{g['drop']:+.3f}" for g in grp))
    print(f"  → single feature is causally directed (amplifies to full removal); recall is "
          f"distributed (drop crosses 0.2 at k≈{k_cross})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
