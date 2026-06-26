#!/usr/bin/env python3
"""Phase 4 foundation — persist the canonical Branch B decomposition + candidate pool.

Phase 3 fit the 410m decomposition at every Pareto point but never *saved* one (Gate-3
caveat 7). Every Phase 4 deliverable — the baseline comparison (Step 1), the stability
metrics (Step 2), and above all the causal existence proof (Step 3) — reads a single
canonical trained SSM. This script produces it, once, at the Gate-3 knee
(``λ_L1 = 1.0``, ``r = d_z = 32``, seed 0), and writes:

  * ``checkpoints/ssm_seed0.pt``    — operators (reload with ``checkpoint.load_ssm``)
  * ``checkpoints/u_train_fit.npy`` — the fit train inputs ``u`` (canonical train recon)
  * ``checkpoints/u_infer_full.npy``— operators-frozen LASSO inputs for **all** N examples
       (the honest per-example encoder the Step-3 ablation re-solves against; selecting
       top-firing / zero-activation examples and computing feature magnitudes all read this)
  * ``checkpoints/split.npz``       — train/test indices (seed 0, matches Gate-3)
  * ``tables/candidate_pool.json``  — per top-feature record: home layer, |u| mass,
       cross-seed decoder-direction stability (min/mean pairwise cosine, ``stable`` =
       min ≥ 0.7), and the relation-class theme of its top-activating examples.

The candidate pool **is** the Phase-4 Step-2 "seed-stable subset" and the Step-3 candidate
enumeration in one table: the causal proof draws candidates from ``stable == true`` features
(per the Gate-3 carry-over "restrict the candidate pool to seed-stable features"). Seed
stability is identity-matched on the fixed SAE dictionary (same convention as Step 4): a
feature ``j`` with home layer ``ℓ*`` (from full-dataset mass) is scored by the cosine of its
feedthrough column ``D_{ℓ*}[:, j]`` across the seed-0/1/2 refits.

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/phase4_checkpoint.py \
        --config configs/pythia-410m-deduped.yaml --desc ph4-checkpoint-410md \
        --sources-npz data/pythia-410m-deduped/sources_mlsae.npz \
        --rank 32 --lambda-l1 1.0 --outer-iters 10 --device cuda

Three full 410m fits (canonical + 2 stability seeds), GPU freed between them. The 3 GB GTX
1050 fragments across sequential fits, so the **expandable_segments** allocator config is
required (without it the third fit OOMs); ``--resume`` recovers from a crash by reusing the
already-saved canonical checkpoint and recomputing only the seed-stability pool.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from layer_dynamics.checkpoint import load_ssm, save_ssm
from layer_dynamics.cli import add_common_args, resolve_out
from layer_dynamics.decompose import AltMinConfig, alternating_minimize, evaluate, infer_inputs
from layer_dynamics.device import describe, resolve_device
from layer_dynamics.diagnostics.dataset import deltas, load_residual_stream, train_test_split_idx
from layer_dynamics.diagnostics.decompose_sanity import (
    decoder_directions,
    direction_cosines,
    feature_top_examples,
    top_features,
)
from layer_dynamics.runlog import init_run
from layer_dynamics.saes_mlsae import load_cached_u
from layer_dynamics.ssm import LowRankSSM, SSMDims


def _free_cuda(*objs) -> None:
    for o in objs:
        del o
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _fit_train(u_tr, d_tr, dims, cfg, *, seed, device, learn_z0, log):
    """One full alternating-minimisation fit on the train split."""
    model = LowRankSSM(dims, learn_z0=learn_z0, seed=seed).to(device)
    result = alternating_minimize(model, u_tr, d_tr, cfg, log=log)
    return model, result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_args(parser)
    parser.add_argument("--sources-npz", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--d-state", type=int, default=None)
    parser.add_argument("--lambda-l1", type=float, default=1.0)
    parser.add_argument("--lambda-reg", type=float, default=1e-4)
    parser.add_argument("--outer-iters", type=int, default=10)
    parser.add_argument("--operator-steps", type=int, default=40)
    parser.add_argument("--operator-lr", type=float, default=1e-2)
    parser.add_argument("--input-steps", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--n-seeds", type=int, default=3, help="Seeds for stability (>=2).")
    parser.add_argument("--top-n", type=int, default=200,
                        help="Top features (by full-dataset |u| mass) in the candidate pool.")
    parser.add_argument("--no-learn-z0", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Reuse a saved canonical fit in --out's checkpoints/ (skip the "
                             "seed-0 fit + frozen inference); only rebuild the seed-stability "
                             "pool. For recovering from an OOM mid-stability-refit.")
    args = parser.parse_args(argv)

    run = init_run(resolve_out(args), config_path=args.config, seed=args.seed)
    mcfg = run.config.get("model", {})
    device = resolve_device(args.device or mcfg.get("device", "auto"))
    run.log(f"device: {describe(device)}")
    learn_z0 = not args.no_learn_z0

    # --- data + cached sources ---
    h5 = Path("data") / mcfg["name"] / "residual_stream.h5"
    rs, rel_class, _ = load_residual_stream(h5)
    delta_all = deltas(rs)
    n, n_layers, d_model = delta_all.shape
    u_all = load_cached_u(args.sources_npz)
    d_input = u_all.shape[2]
    sae_l0 = float((u_all != 0).mean())
    run.log(f"Δ {delta_all.shape}; sources u {u_all.shape} M={d_input} init L0={sae_l0:.4f}")

    meta_df = _load_meta(mcfg["name"], rel_class)

    train_idx, test_idx = train_test_split_idx(n, test_frac=args.test_frac, seed=args.seed)
    u_tr = torch.from_numpy(u_all[train_idx])
    d_tr = torch.from_numpy(delta_all[train_idx])
    d_te = torch.from_numpy(delta_all[test_idx])
    run.log(f"split: {len(train_idx)} train / {len(test_idx)} test (seed {args.seed})")

    d_state = args.d_state if args.d_state is not None else args.rank
    dims = SSMDims(n_layers=n_layers, d_model=d_model, d_input=d_input,
                   d_state=d_state, rank=args.rank)
    cfg = AltMinConfig(
        lambda_l1=args.lambda_l1, lambda_reg=args.lambda_reg, outer_iters=args.outer_iters,
        operator_steps=args.operator_steps, operator_lr=args.operator_lr,
        input_steps=args.input_steps, batch_size=args.batch_size,
    )
    run.log(f"alt-min: λ_L1={cfg.lambda_l1:g} λ_reg={cfg.lambda_reg:g} outer={cfg.outer_iters} "
            f"r={args.rank} d_z={d_state}")

    ckpt_dir = Path(run.path("checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if args.resume:
        # Reuse a saved canonical fit (e.g. after an OOM mid-stability-refit) — only the
        # seed-stability pool below is recomputed. Operators + frozen inputs are byte-identical
        # to a fresh run (seed 0, deterministic), so this is exact, not an approximation.
        model0 = load_ssm(ckpt_dir / "ssm_seed0.pt", device=device)
        result0 = None
        u_full_np = np.load(ckpt_dir / "u_infer_full.npy")
        u_tr_fit = np.load(ckpt_dir / "u_train_fit.npy")
        train_ev = evaluate(model0, torch.from_numpy(u_tr_fit), d_tr, lambda_l1=cfg.lambda_l1,
                            lambda_reg=cfg.lambda_reg, batch_size=cfg.batch_size)
        _free_cuda()
        test_ev = evaluate(model0, torch.from_numpy(u_full_np[test_idx]), d_te,
                           lambda_l1=cfg.lambda_l1, lambda_reg=cfg.lambda_reg,
                           batch_size=cfg.batch_size)
        run.log(f"RESUMED canonical checkpoint: train R²={train_ev['r2']:+.3f} "
                f"test R²={test_ev['r2']:+.3f} L0={test_ev['l0']:.4f}")
    else:
        # --- canonical fit (seed 0) ---
        run.log(f"=== canonical fit (seed {args.seed}) ===")
        model0, result0 = _fit_train(u_tr, d_tr, dims, cfg, seed=args.seed, device=device,
                                     learn_z0=learn_z0, log=run.log)
        train_ev = evaluate(model0, result0.u, d_tr, lambda_l1=cfg.lambda_l1,
                            lambda_reg=cfg.lambda_reg, batch_size=cfg.batch_size)
        run.log(f"canonical train R²={train_ev['r2']:+.3f} L0={train_ev['l0']:.4f} "
                f"|D|={train_ev['op_D']:.1f} |C|={train_ev['op_C']:.1f}")

        save_ssm(model0, ckpt_dir / "ssm_seed0.pt",
                 meta={"model": mcfg.get("name"), "seed": args.seed, "lambda_l1": args.lambda_l1,
                       "rank": args.rank, "d_state": d_state, "train_r2": train_ev["r2"]})
        u_tr_fit = result0.u.detach().cpu().numpy().astype(np.float32)
        np.save(ckpt_dir / "u_train_fit.npy", u_tr_fit)
        np.savez(ckpt_dir / "split.npz", train_idx=train_idx, test_idx=test_idx)
        run.log(f"saved checkpoint + u_train_fit {u_tr_fit.shape} + split")

        # --- operators-frozen LASSO inputs for ALL examples (Step-3 operative encoder) ---
        run.log("=== frozen-inference over full dataset (operators fixed) ===")
        u_full_init = torch.from_numpy(u_all)
        d_full = torch.from_numpy(delta_all)
        u_full = infer_inputs(model0, u_full_init, d_full, lambda_l1=cfg.lambda_l1,
                              n_steps=max(cfg.input_steps * 4, 80), lipschitz=result0.lipschitz,
                              batch_size=cfg.batch_size)
        u_full_np = u_full.detach().cpu().numpy().astype(np.float32)
        np.save(ckpt_dir / "u_infer_full.npy", u_full_np)
        test_ev = evaluate(model0, u_full[test_idx], d_te, lambda_l1=cfg.lambda_l1,
                           lambda_reg=cfg.lambda_reg, batch_size=cfg.batch_size)
        run.log(f"frozen full inference u {u_full_np.shape}; held-out test R²={test_ev['r2']:+.3f} "
                f"L0={test_ev['l0']:.4f}")
        _free_cuda(u_full, u_full_init, d_full)

    # --- candidate pool: rank top-N by full-dataset |u| mass, score seed stability ---
    u_full_t = torch.from_numpy(u_full_np)
    feats = top_features(u_full_t, top_n=args.top_n)
    fidx = [f["feature"] for f in feats]
    flayer = [f["home_layer"] for f in feats]
    run.log(f"candidate pool: top-{len(fidx)} features by full-dataset |u| mass")

    dirs = {args.seed: decoder_directions(model0, fidx, flayer)}
    _free_cuda(model0, result0)

    extra_seeds = [s for s in range(args.seed + 1, args.seed + args.n_seeds)][: args.n_seeds - 1]
    for s in extra_seeds:
        run.log(f"  --- stability seed {s} ---")
        m, res = _fit_train(u_tr, d_tr, dims, cfg, seed=s, device=device,
                            learn_z0=learn_z0, log=run.log)
        ev = evaluate(m, res.u, d_tr, lambda_l1=cfg.lambda_l1, lambda_reg=cfg.lambda_reg,
                      batch_size=cfg.batch_size)
        run.log(f"  seed {s}: train R²={ev['r2']:+.3f} L0={ev['l0']:.4f}")
        dirs[s] = decoder_directions(m, fidx, flayer)
        _free_cuda(m, res)

    seeds = [args.seed, *extra_seeds]
    pairs = [(a, b) for i, a in enumerate(seeds) for b in seeds[i + 1:]]
    # per-feature pairwise cosines across seed pairs → min (conservative) + mean
    cos_by_pair = {f"{a}-{b}": direction_cosines(dirs[a], dirs[b]) for a, b in pairs}
    cos_stack = np.stack(list(cos_by_pair.values()), axis=1)  # [k, n_pairs]
    min_cos = cos_stack.min(axis=1)
    mean_cos = cos_stack.mean(axis=1)
    stable_mask = min_cos >= 0.7
    run.log(f"[seed stability] {int(stable_mask.sum())}/{len(fidx)} features stable "
            f"(min pairwise cos ≥ 0.7); pool match-rate "
            f"{float((cos_stack >= 0.7).mean()):.2f} over {cos_stack.shape[1]} pair(s)")

    # --- themes: relation-class of each feature's top-activating examples (full dataset) ---
    pool = []
    for k, f in enumerate(feats):
        ex = feature_top_examples(u_full_t, f["feature"], f["home_layer"], top_e=10)
        rows = meta_df.iloc[ex]
        classes = rows["relation_class"].tolist()
        top_class = max(set(classes), key=classes.count) if classes else None
        purity = classes.count(top_class) / len(classes) if classes else 0.0
        pool.append({
            "feature": int(f["feature"]), "home_layer": int(f["home_layer"]),
            "mass": float(f["mass"]),
            "seed_min_cos": float(min_cos[k]), "seed_mean_cos": float(mean_cos[k]),
            "stable": bool(stable_mask[k]),
            "top_class": top_class, "purity": float(purity),
            "top_example_idx": [int(i) for i in ex],
            "top_subjects": rows["subject"].tolist(),
            "top_relations": classes,
        })

    stats = {
        "model": mcfg.get("name"), "device": device, "dims": vars(dims),
        "hyperparams": cfg.__dict__, "seeds": seeds,
        "n_train": len(train_idx), "n_test": len(test_idx),
        "train_r2": train_ev["r2"], "test_r2": test_ev["r2"],
        "train_l0": train_ev["l0"], "test_l0": test_ev["l0"],
        "top_n": args.top_n, "n_stable": int(stable_mask.sum()),
        "stable_fraction": float(stable_mask.mean()),
        "pool_match_rate_0.7": float((cos_stack >= 0.7).mean()),
        "candidate_pool": pool,
    }
    run.save_json("tables/candidate_pool.json", stats)

    import pandas as pd

    pd.DataFrame([{k: v for k, v in p.items()
                   if k not in ("top_example_idx", "top_subjects", "top_relations")}
                  for p in pool]).to_csv(run.path("tables/candidate_pool.csv"), index=False)

    print(f"\n=== Phase 4 checkpoint ({mcfg.get('name')}) ===")
    print(f"  canonical fit: train R²={train_ev['r2']:+.3f}  test R²={test_ev['r2']:+.3f}  "
          f"L0={test_ev['l0']:.4f}")
    print(f"  candidate pool: {len(pool)} features; "
          f"{int(stable_mask.sum())} seed-stable (min pairwise cos ≥ 0.7, "
          f"{100 * stable_mask.mean():.0f}%)")
    print("  top-8 stable features (feature@layer  min_cos  purity  class):")
    shown = 0
    for p in pool:
        if not p["stable"]:
            continue
        print(f"    {p['feature']:>6}@{p['home_layer']:<2}  cos={p['seed_min_cos']:.2f}  "
              f"purity={p['purity']:.2f}  {p['top_class']}")
        shown += 1
        if shown >= 8:
            break
    print(f"  checkpoint dir: {ckpt_dir}")
    return 0


def _load_meta(model_name: str, rel_class: np.ndarray):
    """Subject/relation metadata aligned by position to the residual store."""
    import pandas as pd

    df = pd.read_parquet(Path("data") / model_name / "filtered_triples.parquet")
    df = df.reset_index(drop=True)
    if len(df) != len(rel_class):
        raise ValueError(f"metadata/store length mismatch: {len(df)} vs {len(rel_class)}")
    if int((df["relation_class"].to_numpy() != rel_class).sum()):
        raise ValueError("metadata not aligned to store (relation_class mismatch)")
    return df


if __name__ == "__main__":
    raise SystemExit(main())
