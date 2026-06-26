#!/usr/bin/env python3
"""Phase 4 Step 3 — causal screen of seed-stable features (items 4–5: ablation null).

Loads the canonical decomposition (``scripts/phase4_checkpoint.py``) and, for a shortlist of
**seed-stable** candidate features, runs the pre-registered ablation test on the *real*
Pythia-410m:

  * **Ablation (item 4).** For the 100 examples where the feature fires most strongly, subtract
    its estimated contribution ``δ_ℓ`` (SSM forward twice: natural vs. coordinate zeroed) from
    the residual stream at every layer ``ℓ ≥ ℓ*`` and measure the drop in top-1 probability of
    the correct answer. **Threshold:** mean drop ≥ 0.20, paired-t ``p < 0.01`` (Bonferroni over
    the shortlist).
  * **Controls (item 4, mechanism-sanity).** Same on 100 silent (zero-activation) examples; the
    subtract-δ splice must be a near-no-op (mean drop ≤ 0.05) — catches splice/index bugs.
  * **Random-feature null (item 5).** 10 magnitude-matched random features (mean |activation| on
    the candidate's top-100 within ±20%, widening to ±50%), ablated on the *same* top-100;
    **threshold:** mean drop ≤ 0.05. This is what establishes specificity.

The answer token is the clean run's top-1 at the query position (the dataset is filtered to
top-1 == object, so this is the correct object token). The semantic-token *activation* test
(item 6) needs a human-legible description of the chosen candidate and is run separately once a
candidate clears this screen. A feature that survives ablation+controls+random null here is the
input to that test.

    python scripts/phase4_causal.py --config configs/pythia-410m-deduped.yaml \
        --desc ph4-causal-410md --checkpoint-dir runs/2026-06-26-ph4-checkpoint-410md \
        --n-candidates 6 --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from layer_dynamics.causal import (
    answer_probs,
    feature_delta_contribution,
    magnitude_matched_features,
    run_prompts_with_resid_edit,
    top_and_control_examples,
)
from layer_dynamics.checkpoint import load_ssm
from layer_dynamics.cli import add_common_args, resolve_out
from layer_dynamics.device import describe, resolve_device
from layer_dynamics.modeling import load_model
from layer_dynamics.runlog import init_run


def _select_candidates(pool: list[dict], args) -> list[dict]:
    """The shortlist: explicit --feature/--layer, else top stable features by purity then mass."""
    if args.feature is not None:
        return [next(p for p in pool if p["feature"] == args.feature
                     and (args.layer is None or p["home_layer"] == args.layer))]
    stable = [p for p in pool if p["stable"]]
    stable.sort(key=lambda p: (p["purity"], p["mass"]), reverse=True)
    return stable[: args.n_candidates]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_args(parser)
    parser.add_argument("--checkpoint-dir", type=Path, required=True,
                        help="Run dir from scripts/phase4_checkpoint.py (checkpoints/ + tables/).")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--n-candidates", type=int, default=6)
    parser.add_argument("--feature", type=int, default=None, help="Ablate this feature only.")
    parser.add_argument("--layer", type=int, default=None, help="...at this home layer.")
    parser.add_argument("--n-examples", type=int, default=100)
    parser.add_argument("--n-rand", type=int, default=10)
    parser.add_argument("--model-batch", type=int, default=4)
    args = parser.parse_args(argv)

    run = init_run(resolve_out(args), config_path=args.config, seed=args.seed)
    mcfg = run.config.get("model", {})
    device = resolve_device(args.device or mcfg.get("device", "auto"))
    run.log(f"device: {describe(device)}")

    # --- load checkpoint artefacts ---
    ck = args.checkpoint_dir
    import json

    pool_stats = json.loads((ck / "tables" / "candidate_pool.json").read_text())
    pool = pool_stats["candidate_pool"]
    u_full = np.load(ck / "checkpoints" / "u_infer_full.npy")
    ssm = load_ssm(ck / "checkpoints" / "ssm_seed0.pt", device=device)
    run.log(f"loaded SSM + u_full {u_full.shape}; pool {len(pool)} features "
            f"({pool_stats['n_stable']} stable)")

    meta_df = _load_meta(mcfg["name"])
    prompts = meta_df["prompt"].tolist()

    candidates = _select_candidates(pool, args)
    shortlist = [(c["feature"], c["home_layer"], round(c["purity"], 2)) for c in candidates]
    run.log(f"shortlist (feat, layer, purity): {shortlist}")

    # --- model ---
    model = load_model(mcfg["name"], device=device, dtype=mcfg.get("dtype", "float32"))
    run.log(f"model {mcfg['name']} loaded (L={model.cfg.n_layers}, D={model.cfg.d_model})")

    n_cand = len(candidates)
    results = []
    for ci, cand in enumerate(candidates):
        m, ell = cand["feature"], cand["home_layer"]
        a = u_full[:, ell, m]
        top_idx, ctrl_idx, ctrl_rule = top_and_control_examples(a, n=args.n_examples)
        run.log(f"=== candidate {ci + 1}/{n_cand}: feat {m}@L{ell} "
                f"(purity {cand['purity']:.2f}, class {cand['top_class']}, "
                f"control rule {ctrl_rule}) ===")

        # clean run on top + control: answer = clean argmax at query position
        sel = np.concatenate([top_idx, ctrl_idx])
        clean_logits = run_prompts_with_resid_edit(
            model, [prompts[i] for i in sel], edit=None, batch_size=args.model_batch)
        answer_ids = clean_logits.argmax(dim=-1).numpy()
        clean_p = answer_probs(clean_logits, answer_ids)
        nt = len(top_idx)
        clean_top, clean_ctrl = clean_p[:nt], clean_p[nt:]

        # ablate the candidate on its top-100 and on the controls
        d_top = feature_delta_contribution(ssm, torch.from_numpy(u_full[top_idx]), m, ell)
        d_ctrl = feature_delta_contribution(ssm, torch.from_numpy(u_full[ctrl_idx]), m, ell)
        abl_top = answer_probs(run_prompts_with_resid_edit(
            model, [prompts[i] for i in top_idx], edit=d_top, sign=-1.0,
            batch_size=args.model_batch), answer_ids[:nt])
        abl_ctrl = answer_probs(run_prompts_with_resid_edit(
            model, [prompts[i] for i in ctrl_idx], edit=d_ctrl, sign=-1.0,
            batch_size=args.model_batch), answer_ids[nt:])
        drop_top = clean_top - abl_top
        drop_ctrl = clean_ctrl - abl_ctrl
        t_stat, p_val = stats.ttest_rel(clean_top, abl_top)

        # propagation-only ablation: zero δ at the home layer, keep the downstream (ℓ>ℓ*)
        # state-path contribution. This isolates exactly what no per-layer method has — the
        # cross-layer propagation (§4.2). A real drop here = the state path is causally
        # load-bearing, not just a reconstruction term.
        d_prop = d_top.clone()
        d_prop[:, : ell + 1] = 0.0
        abl_prop = answer_probs(run_prompts_with_resid_edit(
            model, [prompts[i] for i in top_idx], edit=d_prop, sign=-1.0,
            batch_size=args.model_batch), answer_ids[:nt])
        drop_prop = clean_top - abl_prop
        home_norm = float(d_top[:, ell].norm())
        down_norm = float(d_top[:, ell + 1:].norm()) if ell + 1 < d_top.shape[1] else 0.0
        downstream_frac = down_norm / (home_norm + down_norm + 1e-9)

        # magnitude-matched random-feature null on the same top-100
        rand_feats, tol_used = magnitude_matched_features(
            u_full, ell, m, top_idx, n_rand=args.n_rand,
            rng=np.random.default_rng(args.seed + ci))
        rand_drops = []
        for rl, rm in rand_feats:
            d_r = feature_delta_contribution(ssm, torch.from_numpy(u_full[top_idx]), rm, rl)
            abl_r = answer_probs(run_prompts_with_resid_edit(
                model, [prompts[i] for i in top_idx], edit=d_r, sign=-1.0,
                batch_size=args.model_batch), answer_ids[:nt])
            rand_drops.append(float((clean_top - abl_r).mean()))
        rand_applicable = len(rand_feats) >= args.n_rand
        rand_mean = float(np.mean(rand_drops)) if rand_drops else float("nan")
        rand_p95 = float(np.percentile(rand_drops, 95)) if rand_drops else float("nan")

        rec = {
            "feature": m, "home_layer": ell, "top_class": cand["top_class"],
            "purity": cand["purity"], "seed_min_cos": cand["seed_min_cos"],
            "control_rule": ctrl_rule, "n_top": int(nt), "n_control": int(len(ctrl_idx)),
            "clean_prob_top_mean": float(clean_top.mean()),
            "ablate_drop_top_mean": float(drop_top.mean()),
            "ablate_drop_top_median": float(np.median(drop_top)),
            "ablate_drop_prop_mean": float(drop_prop.mean()),
            "downstream_frac": downstream_frac,
            "ttest_t": float(t_stat), "ttest_p": float(p_val),
            "ttest_p_bonferroni": float(min(1.0, p_val * n_cand)),
            "control_drop_mean": float(drop_ctrl.mean()),
            "rand_n": len(rand_feats), "rand_tol": tol_used,
            "rand_applicable": rand_applicable,
            "rand_drop_mean": rand_mean, "rand_drop_p95": rand_p95,
        }
        # pre-registered thresholds (item 6 activation handled separately)
        rec["pass_effect"] = bool(rec["ablate_drop_top_mean"] >= 0.20
                                  and rec["ttest_p_bonferroni"] < 0.01)
        rec["pass_control"] = bool(abs(rec["control_drop_mean"]) <= 0.05)
        rec["pass_random"] = bool((not rand_applicable) or rand_mean <= 0.05)
        results.append(rec)
        run.log(f"  clean p={rec['clean_prob_top_mean']:.3f}  ablate drop="
                f"{rec['ablate_drop_top_mean']:+.3f} (p_bonf={rec['ttest_p_bonferroni']:.1e})  "
                f"control drop={rec['control_drop_mean']:+.3f}  "
                f"random mean={rand_mean:+.3f} (tol {tol_used}, n={len(rand_feats)})")
        run.log(f"  propagation-only drop={rec['ablate_drop_prop_mean']:+.3f}  "
                f"downstream write frac={downstream_frac:.2f}")
        run.log(f"  thresholds: effect {'PASS' if rec['pass_effect'] else 'no'} | "
                f"control {'PASS' if rec['pass_control'] else 'no'} | "
                f"random {'PASS' if rec['pass_random'] else 'no'}")

    run.save_json("tables/causal_screen.json", {
        "model": mcfg.get("name"), "checkpoint_dir": str(ck),
        "n_candidates": n_cand, "results": results,
    })
    import pandas as pd

    pd.DataFrame(results).to_csv(run.path("tables/causal_screen.csv"), index=False)

    print(f"\n=== Phase 4 causal screen ({mcfg.get('name')}) ===")
    print(f"  {'feat@L':>9} {'class':>12} {'clean':>6} {'abl_drop':>9} {'ctrl':>7} "
          f"{'rand':>7} {'verdict':>8}")
    for r in results:
        verdict = "PASS" if (r["pass_effect"] and r["pass_control"] and r["pass_random"]) else "—"
        print(f"  {str(r['feature']) + '@' + str(r['home_layer']):>9} {str(r['top_class']):>12} "
              f"{r['clean_prob_top_mean']:>6.3f} {r['ablate_drop_top_mean']:>+9.3f} "
              f"{r['control_drop_mean']:>+7.3f} {r['rand_drop_mean']:>+7.3f} {verdict:>8}")
    return 0


def _load_meta(model_name: str):
    import pandas as pd

    return pd.read_parquet(Path("data") / model_name / "filtered_triples.parquet").reset_index(
        drop=True)


if __name__ == "__main__":
    raise SystemExit(main())
