#!/usr/bin/env python3
"""Phase 4 Step 3 item 6 — the symmetric activation test for a chosen candidate.

The ablation screen (``phase4_causal.py``) establishes that removing a feature damages
recall. This is the complementary direction: **inject** the feature on examples where it is
normally silent and check that it pushes the model toward the feature's concept. Run on the
candidate that survives the screen, with a pre-specified semantic token set drawn from its
human-legible description (item 3).

Procedure (pre-registered): on the 100 silent (zero-activation) control examples, set the
candidate coordinate ``u[:, ℓ*, m]`` to the candidate's **mean signed activation** on its
top-100 firing examples, compute the injected contribution ``δ_ℓ`` (SSM forward; from-zero by
construction), **add** it to the residual stream at every ℓ ≥ ℓ*, and measure the increase in
the max log-probability over the semantic token set. **Threshold:** ≥ 1.0 nat increase on ≥ 70
of the 100 examples. The token set is restricted to single-token, leading-space realisations.

    python scripts/phase4_activation.py --config configs/pythia-410m-deduped.yaml \
        --desc ph4-activation-410md --checkpoint-dir runs/2026-06-26-ph4-checkpoint-410md \
        --feature 12345 --layer 7 --tokens "Paris,London,Berlin,Madrid,Rome,Vienna,Warsaw"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from layer_dynamics.causal import (
    feature_delta_contribution,
    max_token_logprob,
    run_prompts_with_resid_edit,
    single_token_ids,
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
    parser.add_argument("--feature", type=int, required=True)
    parser.add_argument("--layer", type=int, required=True, help="The feature's home layer ℓ*.")
    parser.add_argument("--tokens", type=str, required=True,
                        help="Comma-separated semantic words (matched as ' word', single-token).")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--n-examples", type=int, default=100)
    parser.add_argument("--nat-threshold", type=float, default=1.0)
    parser.add_argument("--frac-threshold", type=float, default=0.70)
    parser.add_argument("--model-batch", type=int, default=4)
    args = parser.parse_args(argv)

    run = init_run(resolve_out(args), config_path=args.config, seed=args.seed)
    mcfg = run.config.get("model", {})
    device = resolve_device(args.device or mcfg.get("device", "auto"))
    run.log(f"device: {describe(device)}")

    ck = args.checkpoint_dir
    u_full = np.load(ck / "checkpoints" / "u_infer_full.npy")
    ssm = load_ssm(ck / "checkpoints" / "ssm_seed0.pt", device=device)
    meta_df = __import__("pandas").read_parquet(
        Path("data") / mcfg["name"] / "filtered_triples.parquet").reset_index(drop=True)
    prompts = meta_df["prompt"].tolist()

    m, ell = args.feature, args.layer
    a = u_full[:, ell, m]
    top_idx, ctrl_idx, ctrl_rule = top_and_control_examples(a, n=args.n_examples)
    mean_signed = float(u_full[top_idx, ell, m].mean())
    run.log(f"feature {m}@L{ell}: mean signed activation on top-{len(top_idx)} = "
            f"{mean_signed:+.3f} (control rule {ctrl_rule}, {len(ctrl_idx)} silent examples)")

    model = load_model(mcfg["name"], device=device, dtype=mcfg.get("dtype", "float32"))
    token_ids, kept = single_token_ids(model, args.tokens.split(","))
    run.log(f"semantic tokens kept ({len(kept)}/{len(args.tokens.split(','))}): {kept}")
    if len(token_ids) < 5:
        run.log("WARNING: <5 single-token semantic realisations — pick another concept (item 3)")

    # inject the candidate at its mean signed activation on the silent examples (from zero)
    u_inj = u_full[ctrl_idx].copy()
    u_inj[:, ell, m] = mean_signed
    d_inj = feature_delta_contribution(ssm, torch.from_numpy(u_inj), m, ell)

    ctrl_prompts = [prompts[i] for i in ctrl_idx]
    clean_logits = run_prompts_with_resid_edit(model, ctrl_prompts, edit=None,
                                               batch_size=args.model_batch)
    inj_logits = run_prompts_with_resid_edit(model, ctrl_prompts, edit=d_inj, sign=+1.0,
                                             batch_size=args.model_batch)
    clean_lp = max_token_logprob(clean_logits, token_ids)
    inj_lp = max_token_logprob(inj_logits, token_ids)
    increase = inj_lp - clean_lp
    frac = float((increase >= args.nat_threshold).mean())
    passed = bool(frac >= args.frac_threshold)

    stats = {
        "model": mcfg.get("name"), "feature": m, "home_layer": ell,
        "mean_signed_activation": mean_signed, "control_rule": ctrl_rule,
        "n_silent": int(len(ctrl_idx)), "semantic_tokens": kept,
        "mean_logprob_increase": float(increase.mean()),
        "median_logprob_increase": float(np.median(increase)),
        "frac_above_threshold": frac, "nat_threshold": args.nat_threshold,
        "frac_threshold": args.frac_threshold, "pass": passed,
    }
    run.save_json("tables/activation_test.json", stats)

    print(f"\n=== Phase 4 activation test — feature {m}@L{ell} ({mcfg.get('name')}) ===")
    print(f"  semantic tokens: {kept}")
    print(f"  mean Δlog-prob increase: {increase.mean():+.3f} nat "
          f"(median {np.median(increase):+.3f})")
    print(f"  fraction ≥ {args.nat_threshold} nat: {frac:.2f}  "
          f"(threshold {args.frac_threshold}) -> {'PASS' if passed else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
