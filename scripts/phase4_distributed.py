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
  4. **Norm-matched random-*direction* control** (E1; the magnitude-vs-direction crux). The
     mass-matched group (3) stays inside the decoder subspace, so it cannot rule out "subtracting
     *any* vector of this residual norm breaks recall". At each k, subtract a random direction of
     the **same per-(example, layer) norm** as the top-k δ (``causal.norm_matched_random_delta``),
     both ``isotropic`` (random in R^D) and ``shaped`` (random within the contribution subspace).
     If these stay below the top-k curve, the effect is carried by the contribution *direction*,
     not its bulk — the decomposition is a causally-privileged subspace, not a magnitude sink.
  5. **Off-target specificity** (E1b). At the crossing k, compare the drop in log-prob of each
     example's own gold answer (on-target) against a basket of other-relation-class gold tokens
     (off-target) in the same forward pass. On-target ≫ off-target means the distributed ablation
     suppresses *the specific answer*, not the output distribution at large.

Diagnostics 1–2 are exploratory; 3–5 are the controls that let the distributed-code finding enter
the paper as a *specific, structured* code rather than a generic magnitude effect. The
pre-registered single-feature null stands either way.

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
    norm_matched_random_delta,
    run_prompts_with_resid_edit,
    single_token_ids,
    token_logprobs,
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

    triples = pd.read_parquet(
        Path("data") / mcfg["name"] / "filtered_triples.parquet").reset_index(drop=True)
    prompts = triples["prompt"].tolist()
    ks = [int(x) for x in args.group_ks.split(",")]

    model = load_model(mcfg["name"], device=device, dtype=mcfg.get("dtype", "float32"))

    # off-target token baskets (diagnostic 5): single-token, leading-space gold answers per
    # relation class. The control loop reads each class's *other*-class tokens to ask whether the
    # top-k ablation suppresses the gold answer specifically or flattens unrelated answers too.
    class_gold_ids: dict[str, list[int]] = {}
    for cls, grp in triples.groupby("relation_class"):
        ids, _ = single_token_ids(model, grp["top1_token"].tolist())
        class_gold_ids[str(cls)] = sorted(set(ids))
    offclass_ids = {
        cls: sorted(set().union(*(set(v) for c, v in class_gold_ids.items() if c != cls)))
        for cls in class_gold_ids
    }

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
    top_rows, rnd_rows, iso_rows, shp_rows, mass_rows, percand = [], [], [], [], [], []
    for ci, cand in enumerate(candidates):
        m, ell = cand["feature"], cand["home_layer"]
        top_idx, _, _ = top_and_control_examples(u_full[:, ell, m], n=args.n_examples)
        pr = [prompts[i] for i in top_idx]
        ut = torch.from_numpy(u_full[top_idx])
        absu = ut.reshape(ut.shape[0], -1).abs()
        clean = run_prompts_with_resid_edit(model, pr, edit=None, batch_size=args.model_batch)
        ans = clean.argmax(dim=-1).numpy()
        cp = answer_probs(clean, ans)
        seed_ci = args.control_seed + ci

        d_top, d_rnd, d_iso, d_shp, mass_top, mass_rnd, cnt_rnd = [], [], [], [], [], [], []
        for k in ks:
            dg = group_delta_contribution(ssm, ut, k)
            d_top.append(_ablate_drop(model, pr, dg, cp, ans, args.model_batch))
            dr, n_abl, m_abl = group_random_delta_contribution(
                ssm, ut, k, seed=seed_ci, match="count")
            d_rnd.append(_ablate_drop(model, pr, dr, cp, ans, args.model_batch))
            # norm-matched random *direction* of the top-k δ (magnitude held, direction randomised)
            d_iso.append(_ablate_drop(model, pr, norm_matched_random_delta(
                dg, seed=seed_ci, dist="isotropic"), cp, ans, args.model_batch))
            d_shp.append(_ablate_drop(model, pr, norm_matched_random_delta(
                dg, seed=seed_ci, dist="shaped"), cp, ans, args.model_batch))
            mass_top.append(float(absu.topk(min(k, absu.shape[1]), dim=1).values.sum(1).mean()))
            mass_rnd.append(float(m_abl.float().mean()))
            cnt_rnd.append(float(n_abl.float().mean()))
        # mass-matched random point: same total |u| removed as top-(mass_match_k), random pick
        dmm, n_mm, m_mm = group_random_delta_contribution(
            ssm, ut, args.mass_match_k, seed=seed_ci, match="mass")
        drop_mm = _ablate_drop(model, pr, dmm, cp, ans, args.model_batch)

        # off-target specificity (diagnostic 5) at the crossing k: gold-token drop vs other-class
        dg_mm = group_delta_contribution(ssm, ut, args.mass_match_k)
        abl_mm = run_prompts_with_resid_edit(model, pr, edit=dg_mm, sign=-1.0,
                                             batch_size=args.model_batch)
        rows = np.arange(len(top_idx))
        on_dlp = float((torch.log_softmax(clean, -1)[rows, ans]
                        - torch.log_softmax(abl_mm, -1)[rows, ans]).mean())
        basket = offclass_ids.get(str(cand["top_class"]), [])
        off_dlp = (float((token_logprobs(clean, basket) - token_logprobs(abl_mm, basket)).mean())
                   if basket else float("nan"))

        top_rows.append(d_top)
        rnd_rows.append(d_rnd)
        iso_rows.append(d_iso)
        shp_rows.append(d_shp)
        mass_rows.append((mass_top, mass_rnd, cnt_rnd))
        percand.append({
            "feature": int(m), "home_layer": int(ell), "top_class": cand["top_class"],
            "purity": float(cand["purity"]), "clean_prob": float(cp.mean()),
            "drop_top": d_top, "drop_random": d_rnd,
            "drop_normrand_iso": d_iso, "drop_normrand_shaped": d_shp,
            "mass_match_k": args.mass_match_k, "drop_massmatched": drop_mm,
            "massmatched_n_features": float(n_mm.float().mean()),
            "offtarget_on_dlogprob": on_dlp, "offtarget_off_dlogprob": off_dlp,
        })
        mmk_i = ks.index(args.mass_match_k) if args.mass_match_k in ks else len(ks) - 1
        run.log(f"  [{ci + 1}/{len(candidates)}] {m}@L{ell} ({cand['top_class']}): "
                f"top-{ks[mmk_i]} {d_top[mmk_i]:+.3f} | rnd-cnt {d_rnd[mmk_i]:+.3f} | rnd-mass "
                f"{drop_mm:+.3f} | norm-rand iso {d_iso[mmk_i]:+.3f}/shp {d_shp[mmk_i]:+.3f} "
                f"| off-tgt Δlp on {on_dlp:+.2f}/off {off_dlp:+.2f}")

    def msd(rows):  # mean, sd over the candidate axis → [n_k]
        arr = np.array(rows)
        return arr.mean(0), arr.std(0)

    def cross(mean):  # smallest k whose mean drop reaches the 0.2 threshold (or None)
        return next((k for k, v in zip(ks, mean, strict=True) if v >= 0.2), None)

    top_mean, top_sd = msd(top_rows)
    rnd_mean, rnd_sd = msd(rnd_rows)
    iso_mean, iso_sd = msd(iso_rows)
    shp_mean, shp_sd = msd(shp_rows)
    k_cross_top, k_cross_rnd = cross(top_mean), cross(rnd_mean)
    k_cross_iso, k_cross_shp = cross(iso_mean), cross(shp_mean)
    mass_top_mean = np.array([m[0] for m in mass_rows]).mean(0)
    mass_rnd_mean = np.array([m[1] for m in mass_rows]).mean(0)
    cnt_rnd_mean = np.array([m[2] for m in mass_rows]).mean(0)
    mm_drop = float(np.mean([p["drop_massmatched"] for p in percand]))
    mm_n = float(np.mean([p["massmatched_n_features"] for p in percand]))
    mmk_i = ks.index(args.mass_match_k) if args.mass_match_k in ks else len(ks) - 1
    top_at_mmk = float(top_mean[mmk_i])
    iso_at, shp_at = float(iso_mean[mmk_i]), float(shp_mean[mmk_i])
    on_dlp_mean = float(np.mean([p["offtarget_on_dlogprob"] for p in percand]))
    off_dlp_mean = float(np.nanmean([p["offtarget_off_dlogprob"] for p in percand]))

    # verdict: layer the four controls (count / mass / direction / off-target) into one honest read
    crosses = k_cross_top is not None
    direction_matters = crosses and k_cross_iso is None and k_cross_shp is None
    mass_reproduces = abs(mm_drop - top_at_mmk) <= 0.05
    semantic_specific = (on_dlp_mean - off_dlp_mean) > 0.5
    parts = []
    if not crosses:
        parts.append("top-k never crosses 0.2 — no recoverable distributed code at these k")
    else:
        parts.append(f"top-k crosses 0.2 at k={k_cross_top}")
        parts.append("count-matched random stays below (not 'any k features')"
                     if k_cross_rnd is None else
                     f"count-matched random also crosses at k={k_cross_rnd}")
        parts.append(
            f"norm-matched random DIRECTION stays below (iso {iso_at:+.3f}, shaped {shp_at:+.3f} "
            f"vs top {top_at_mmk:+.3f}) → causally-privileged SUBSPACE, not a magnitude sink"
            if direction_matters else
            f"norm-matched random direction reproduces it (iso {iso_at:+.3f}/shaped {shp_at:+.3f}) "
            f"→ MAGNITUDE-driven, subspace not privileged")
        parts.append(f"mass-matched random {'reproduces' if mass_reproduces else 'stays below'} "
                     f"({mm_drop:+.3f})")
        parts.append(f"off-target Δlogprob on {on_dlp_mean:+.2f} vs off {off_dlp_mean:+.2f} nat → "
                     + ("answer-specific" if semantic_specific else "non-specific suppression"))
    verdict = "; ".join(parts)

    control = {
        "model": mcfg.get("name"), "n_candidates": len(candidates),
        "n_examples": args.n_examples, "ks": ks, "control_seed": args.control_seed,
        "top_drop_mean": top_mean.tolist(), "top_drop_sd": top_sd.tolist(),
        "random_drop_mean": rnd_mean.tolist(), "random_drop_sd": rnd_sd.tolist(),
        "normrand_iso_drop_mean": iso_mean.tolist(), "normrand_iso_drop_sd": iso_sd.tolist(),
        "normrand_shaped_drop_mean": shp_mean.tolist(), "normrand_shaped_drop_sd": shp_sd.tolist(),
        "k_cross_0.2_top": k_cross_top, "k_cross_0.2_random": k_cross_rnd,
        "k_cross_0.2_normrand_iso": k_cross_iso, "k_cross_0.2_normrand_shaped": k_cross_shp,
        "topk_mass_mean": mass_top_mean.tolist(), "random_mass_mean": mass_rnd_mean.tolist(),
        "random_count_mean": cnt_rnd_mean.tolist(),
        "mass_match_k": args.mass_match_k, "massmatched_drop_mean": mm_drop,
        "offtarget_on_dlogprob_mean": on_dlp_mean, "offtarget_off_dlogprob_mean": off_dlp_mean,
        "verdict": verdict, "per_candidate": percand,
    }
    run.save_json("tables/distributed_control.json", control)
    _plot_control(ks, top_mean, top_sd, rnd_mean, rnd_sd, iso_mean, iso_sd, shp_mean, shp_sd,
                  args.mass_match_k, mm_drop, run.path("figures/distributed_control.svg"),
                  mcfg.get("name"))

    print(f"\n=== Phase 4 distributed-code control ({mcfg.get('name')}) ===")
    print(f"  pooled over {len(candidates)} seed-stable candidates × {args.n_examples} examples")
    print(f"  {'k':>5} {'top-k':>9} {'rnd-cnt':>9} {'norm-iso':>9} {'norm-shp':>9} {'top mass':>9}")
    for i, k in enumerate(ks):
        print(f"  {k:>5} {top_mean[i]:>+9.3f} {rnd_mean[i]:>+9.3f} {iso_mean[i]:>+9.3f} "
              f"{shp_mean[i]:>+9.3f} {mass_top_mean[i]:>9.1f}")
    print(f"  mass-matched random (≈top-{args.mass_match_k} mass via n≈{mm_n:.0f} random feats): "
          f"drop={mm_drop:+.3f}  (top-{args.mass_match_k}={top_at_mmk:+.3f})")
    print(f"  off-target Δlogprob: on {on_dlp_mean:+.2f} nat vs off-class {off_dlp_mean:+.2f}")
    print(f"  → {verdict}")
    return 0


def _plot_control(ks, top_mean, top_sd, rnd_mean, rnd_sd, iso_mean, iso_sd, shp_mean, shp_sd,
                  mm_k, mm_drop, path, model_name):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    x = np.asarray(ks, dtype=float)
    ax.plot(x, top_mean, "-o", color="C3", label="top-k co-firing")
    ax.fill_between(x, top_mean - top_sd, top_mean + top_sd, color="C3", alpha=0.18)
    ax.plot(x, rnd_mean, "-s", color="C0", label="matched-random (count)")
    ax.fill_between(x, rnd_mean - rnd_sd, rnd_mean + rnd_sd, color="C0", alpha=0.18)
    ax.plot(x, iso_mean, "-D", color="C1", ms=4, label="norm-matched random dir (isotropic)")
    ax.fill_between(x, iso_mean - iso_sd, iso_mean + iso_sd, color="C1", alpha=0.12)
    ax.plot(x, shp_mean, "-v", color="C4", ms=4, label="norm-matched random dir (shaped)")
    ax.fill_between(x, shp_mean - shp_sd, shp_mean + shp_sd, color="C4", alpha=0.12)
    ax.plot([mm_k], [mm_drop], marker="^", color="C2", ms=10, ls="",
            label=f"matched-random (mass, ≈top-{mm_k})")
    ax.axhline(0.20, ls="--", color="0.4", lw=1, label="0.20 threshold")
    ax.set_xscale("log")
    ax.set_xlabel("group size k (features ablated per example)")
    ax.set_ylabel("mean drop in answer prob")
    ax.set_title(f"Distributed-code control — {model_name}")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
