#!/usr/bin/env python3
"""Phase 4 Step 3 follow-up — is the causal null a distributed code or a dead method?

The pre-registered single-feature existence proof (``phase4_causal.py``) returns null: no
seed-stable feature's natural-magnitude ablation moves top-1 probability by the required 0.2.
This script runs the diagnostics that distinguish "the features are causally inert" from "the
features are real but the code is distributed":

  1. **Amplification** (single feature). Ablate one feature's contribution ``δ`` scaled
     ``×{1,3,10,30}``. If the drop scales monotonically and saturates at the clean probability,
     the feature's *direction* is causally correct and only its *magnitude* is small.
  2. **Group ablation** (single feature's examples). Ablate each example's **top-k** co-firing
     features together, sweeping k. The k at which the drop crosses 0.2 measures how distributed
     the recall code is.
  3. **Matched-random-group control** (pooled over the candidate pool — the S3 result). At each k,
     compare the top-k ablation against a **random** group ablation matched to the same example's
     active features (``causal.group_random_delta_contribution``): count-matched (same number of
     features) for the k-axis curve, plus a mass-matched point (same total |u| removed). If top-k
     crosses 0.2 while the matched-random group does not, the distributed code is *structured* —
     carried by the high-magnitude co-firing features, not by "any ~k directions span the answer".
     Averaged over the seed-stable candidate pool so it is a measured result with error bars, not
     a single-feature anecdote.

Diagnostics 1–2 are exploratory; diagnostic 3 is the control that lets the distributed-code
finding enter the paper. The pre-registered single-feature null stands either way.

    python scripts/phase4_distributed.py --config configs/pythia-410m-deduped.yaml \
        --desc ph4-distributed-410md --checkpoint-dir runs/2026-06-26-ph4-checkpoint-410md \
        --feature 426 --layer 22 --device cuda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from layer_dynamics.causal import (
    answer_probs,
    feature_delta_contribution,
    group_delta_contribution,
    group_random_delta_contribution,
    run_prompts_with_resid_edit,
    top_and_control_examples,
)
from layer_dynamics.checkpoint import load_ssm
from layer_dynamics.cli import add_common_args, resolve_out
from layer_dynamics.device import describe, resolve_device
from layer_dynamics.modeling import load_model
from layer_dynamics.runlog import init_run


def _select_stable(pool: list[dict], n: int) -> list[dict]:
    """Top-n seed-stable candidates by purity then |u| mass (same order as phase4_causal)."""
    stable = [p for p in pool if p.get("stable")]
    stable.sort(key=lambda p: (p["purity"], p["mass"]), reverse=True)
    return stable[:n]


def _ablate_drop(model, prompts, edit, clean_p, answer_ids, batch) -> float:
    """Mean drop in answer prob from subtracting ``edit`` (sign −1) at the query token."""
    p = answer_probs(
        run_prompts_with_resid_edit(model, prompts, edit=edit, sign=-1.0, batch_size=batch),
        answer_ids,
    )
    return float((clean_p - p).mean())


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
    parser.add_argument("--pool-candidates", type=int, default=8,
                        help="Seed-stable candidates to average the control over (× n-examples).")
    parser.add_argument("--control-seed", type=int, default=0,
                        help="Base RNG seed for the matched-random group (offset per candidate).")
    parser.add_argument("--mass-match-k", type=int, default=100,
                        help="k at which to also report the mass-matched random group.")
    parser.add_argument("--skip-single", action="store_true",
                        help="Skip the single-feature amplification/group section (control only).")
    parser.add_argument("--model-batch", type=int, default=4)
    args = parser.parse_args(argv)

    run = init_run(resolve_out(args), config_path=args.config, seed=args.seed)
    mcfg = run.config.get("model", {})
    device = resolve_device(args.device or mcfg.get("device", "auto"))
    run.log(f"device: {describe(device)}")

    ck = args.checkpoint_dir
    u_full = np.load(ck / "checkpoints" / "u_infer_full.npy")
    ssm = load_ssm(ck / "checkpoints" / "ssm_seed0.pt", device=device)
    pool = json.loads((ck / "tables" / "candidate_pool.json").read_text())["candidate_pool"]
    import pandas as pd

    prompts = pd.read_parquet(
        Path("data") / mcfg["name"] / "filtered_triples.parquet").reset_index(drop=True)[
        "prompt"].tolist()
    ks = [int(x) for x in args.group_ks.split(",")]

    model = load_model(mcfg["name"], device=device, dtype=mcfg.get("dtype", "float32"))

    # ---------------------------------------------------------------- single feature (1 + 2)
    if not args.skip_single:
        m, ell = args.feature, args.layer
        top_idx, _, _ = top_and_control_examples(u_full[:, ell, m], n=args.n_examples)
        pr = [prompts[i] for i in top_idx]
        ut = torch.from_numpy(u_full[top_idx])
        clean = run_prompts_with_resid_edit(model, pr, edit=None, batch_size=args.model_batch)
        ans = clean.argmax(dim=-1).numpy()
        cp = answer_probs(clean, ans)
        run.log(f"feature {m}@L{ell}: top-{len(top_idx)} clean p={cp.mean():.3f}")

        d = feature_delta_contribution(ssm, ut, m, ell)
        amp = []
        for s in [float(x) for x in args.scales.split(",")]:
            drop = _ablate_drop(model, pr, d * s, cp, ans, args.model_batch)
            amp.append({"scale": s, "drop": drop})
            run.log(f"  amplify ×{s:g}: ablate drop={drop:+.3f}")

        grp = []
        for k in ks:
            dg = group_delta_contribution(ssm, ut, k)
            drop = _ablate_drop(model, pr, dg, cp, ans, args.model_batch)
            grp.append({"k": k, "drop": drop})
            run.log(f"  group top-{k}: ablate drop={drop:+.3f}")
        k_cross = next((g["k"] for g in grp if g["drop"] >= 0.2), None)
        run.save_json("tables/distributed.json", {
            "model": mcfg.get("name"), "feature": m, "home_layer": ell,
            "n_examples": int(len(top_idx)), "clean_prob": float(cp.mean()),
            "amplification": amp, "group_ablation": grp, "k_cross_0.2": k_cross,
        })

    # ----------------------------------------------- matched-random-group control (3, pooled)
    candidates = _select_stable(pool, args.pool_candidates)
    run.log(f"=== matched-random-group control over {len(candidates)} stable candidates "
            f"(×{args.n_examples} examples; ks={ks}) ===")
    top_rows, rnd_rows, mass_rows, percand = [], [], [], []
    for ci, cand in enumerate(candidates):
        m, ell = cand["feature"], cand["home_layer"]
        top_idx, _, _ = top_and_control_examples(u_full[:, ell, m], n=args.n_examples)
        pr = [prompts[i] for i in top_idx]
        ut = torch.from_numpy(u_full[top_idx])
        absu = ut.reshape(ut.shape[0], -1).abs()
        clean = run_prompts_with_resid_edit(model, pr, edit=None, batch_size=args.model_batch)
        ans = clean.argmax(dim=-1).numpy()
        cp = answer_probs(clean, ans)

        d_top, d_rnd, mass_top, mass_rnd, cnt_rnd = [], [], [], [], []
        for k in ks:
            dg = group_delta_contribution(ssm, ut, k)
            d_top.append(_ablate_drop(model, pr, dg, cp, ans, args.model_batch))
            dr, n_abl, m_abl = group_random_delta_contribution(
                ssm, ut, k, seed=args.control_seed + ci, match="count")
            d_rnd.append(_ablate_drop(model, pr, dr, cp, ans, args.model_batch))
            mass_top.append(float(absu.topk(min(k, absu.shape[1]), dim=1).values.sum(1).mean()))
            mass_rnd.append(float(m_abl.float().mean()))
            cnt_rnd.append(float(n_abl.float().mean()))
        # mass-matched random point: same total |u| removed as top-(mass_match_k), random pick
        dmm, n_mm, m_mm = group_random_delta_contribution(
            ssm, ut, args.mass_match_k, seed=args.control_seed + ci, match="mass")
        drop_mm = _ablate_drop(model, pr, dmm, cp, ans, args.model_batch)

        top_rows.append(d_top)
        rnd_rows.append(d_rnd)
        mass_rows.append((mass_top, mass_rnd, cnt_rnd))
        percand.append({
            "feature": int(m), "home_layer": int(ell), "top_class": cand["top_class"],
            "purity": float(cand["purity"]), "clean_prob": float(cp.mean()),
            "drop_top": d_top, "drop_random": d_rnd,
            "mass_match_k": args.mass_match_k, "drop_massmatched": drop_mm,
            "massmatched_n_features": float(n_mm.float().mean()),
        })
        mmk_i = ks.index(args.mass_match_k) if args.mass_match_k in ks else len(ks) - 1
        run.log(f"  [{ci + 1}/{len(candidates)}] {m}@L{ell} ({cand['top_class']}): "
                f"top-{ks[mmk_i]} drop={d_top[mmk_i]:+.3f} | rand-count drop={d_rnd[mmk_i]:+.3f} "
                f"| rand-mass drop={drop_mm:+.3f} (n≈{float(n_mm.float().mean()):.0f})")

    top_arr = np.array(top_rows)   # [n_cand, n_k]
    rnd_arr = np.array(rnd_rows)
    top_mean, top_sd = top_arr.mean(0), top_arr.std(0)
    rnd_mean, rnd_sd = rnd_arr.mean(0), rnd_arr.std(0)
    k_cross_top = next((k for k, v in zip(ks, top_mean, strict=True) if v >= 0.2), None)
    k_cross_rnd = next((k for k, v in zip(ks, rnd_mean, strict=True) if v >= 0.2), None)
    mass_top_mean = np.array([m[0] for m in mass_rows]).mean(0)
    mass_rnd_mean = np.array([m[1] for m in mass_rows]).mean(0)
    cnt_rnd_mean = np.array([m[2] for m in mass_rows]).mean(0)

    control = {
        "model": mcfg.get("name"), "n_candidates": len(candidates),
        "n_examples": args.n_examples, "ks": ks, "control_seed": args.control_seed,
        "top_drop_mean": top_mean.tolist(), "top_drop_sd": top_sd.tolist(),
        "random_drop_mean": rnd_mean.tolist(), "random_drop_sd": rnd_sd.tolist(),
        "k_cross_0.2_top": k_cross_top, "k_cross_0.2_random": k_cross_rnd,
        "topk_mass_mean": mass_top_mean.tolist(), "random_mass_mean": mass_rnd_mean.tolist(),
        "random_count_mean": cnt_rnd_mean.tolist(),
        "mass_match_k": args.mass_match_k,
        "massmatched_drop_mean": float(np.mean([p["drop_massmatched"] for p in percand])),
        "per_candidate": percand,
    }
    run.save_json("tables/distributed_control.json", control)
    mm_drop = control["massmatched_drop_mean"]
    mm_n = float(np.mean([p["massmatched_n_features"] for p in percand]))
    mmk_i = ks.index(args.mass_match_k) if args.mass_match_k in ks else len(ks) - 1
    top_at_mmk = float(top_mean[mmk_i])
    _plot_control(ks, top_mean, top_sd, rnd_mean, rnd_sd, args.mass_match_k, mm_drop,
                  run.path("figures/distributed_control.png"), mcfg.get("name"))

    print(f"\n=== Phase 4 distributed-code control ({mcfg.get('name')}) ===")
    print(f"  pooled over {len(candidates)} seed-stable candidates × {args.n_examples} examples")
    print(f"  {'k':>5} {'top-k drop':>14} {'random drop':>14} {'top mass':>10} {'rand mass':>10}")
    for i, k in enumerate(ks):
        print(f"  {k:>5} {top_mean[i]:>+9.3f}±{top_sd[i]:.3f} "
              f"{rnd_mean[i]:>+9.3f}±{rnd_sd[i]:.3f} "
              f"{mass_top_mean[i]:>10.1f} {mass_rnd_mean[i]:>10.1f}")
    print(f"  crosses 0.2:  top-k at k={k_cross_top}   count-matched random at k={k_cross_rnd}")
    print(f"  mass-matched random (≈top-{args.mass_match_k} mass via n≈{mm_n:.0f} random feats): "
          f"drop={mm_drop:+.3f}  (top-{args.mass_match_k}={top_at_mmk:+.3f})")
    rules_out_count = (k_cross_top is not None) and (k_cross_rnd is None)
    mass_reproduces = abs(mm_drop - top_at_mmk) <= 0.05
    if rules_out_count and mass_reproduces:
        verdict = (
            f"DISTRIBUTED & mass-driven: top-k crosses 0.2 at k={k_cross_top}; "
            f"count-matched random does NOT (rules out 'any k features'); mass-matched "
            f"random reproduces it ({mm_drop:+.3f}~{top_at_mmk:+.3f}) -> no privileged "
            f"subset, effect tracks total ablated feature-mass"
        )
    elif rules_out_count:
        verdict = (
            f"STRUCTURED: top-k crosses 0.2 at k={k_cross_top}; count- and mass-matched "
            f"random stay below ({mm_drop:+.3f}) -> specific co-firing features carry recall"
        )
    else:
        verdict = "count-matched random also crosses 0.2 -- code distributed broadly across actives"
    print(f"  → {verdict}")
    return 0


def _plot_control(ks, top_mean, top_sd, rnd_mean, rnd_sd, mm_k, mm_drop, path, model_name):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    x = np.asarray(ks, dtype=float)
    ax.plot(x, top_mean, "-o", color="C3", label="top-k co-firing")
    ax.fill_between(x, top_mean - top_sd, top_mean + top_sd, color="C3", alpha=0.18)
    ax.plot(x, rnd_mean, "-s", color="C0", label="matched-random (count)")
    ax.fill_between(x, rnd_mean - rnd_sd, rnd_mean + rnd_sd, color="C0", alpha=0.18)
    ax.plot([mm_k], [mm_drop], marker="^", color="C2", ms=10, ls="",
            label=f"matched-random (mass, ≈top-{mm_k})")
    ax.axhline(0.20, ls="--", color="0.4", lw=1, label="0.20 threshold")
    ax.set_xscale("log")
    ax.set_xlabel("group size k (features ablated per example)")
    ax.set_ylabel("mean drop in answer prob")
    ax.set_title(f"Distributed-code control — {model_name}")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
