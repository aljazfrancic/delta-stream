#!/usr/bin/env python3
"""Phase 3 Step 4 — pre-Gate-3 sanity checks for the Branch B decomposition (70m-deduped).

Runs the five Branch-B-applicable checks from the plan
([phase-3](../docs/implementation-plan/phase-3-decomposition.md#step-4--sanity-checks-before-gate-3);
item 6, the MA-stationarity recheck, is Branch A only) on top of the M3 alternating
optimiser:

  1. **Collapse** — operators must not vanish into ``b`` absorbing the per-layer mean.
  2. **Reconstruction eyeball** — predicted vs actual ``Δ_ℓ`` for 10 random examples.
  3. **Feature structure** — top-20 input features by ``|u|`` mass, with the subject /
     relation theme of the 10 examples each activates most.
  4. **Seed stability** — re-fit under 3 seeds; cosine stability of the top-100 features'
     decoder directions (``D_ℓ[:, j]``); fraction ≥ 0.7 (the Gate-3 threshold).
  5. **Dataset stability** — re-fit on two disjoint halves; top-feature overlap (Jaccard)
     and decoder-direction cosines on the shared features.

    python scripts/step4_sanity.py --config configs/pythia-70m.yaml --desc ph3-step4-70md \
        --rank 32 --lambda-l1 1.0 --outer-iters 10 --device cuda

Each fit is one full alternating-minimisation run (operators trained, then inputs refined),
so this is heavier than M3 (5 fits: 3 seeds + 2 halves). The checks write tables + figures
under the run dir and print a consolidated verdict.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from layer_dynamics.cli import add_common_args, resolve_out
from layer_dynamics.decompose import AltMinConfig, alternating_minimize, evaluate
from layer_dynamics.device import describe, resolve_device
from layer_dynamics.diagnostics.dataset import deltas, load_residual_stream, train_test_split_idx
from layer_dynamics.diagnostics.decompose_sanity import (
    collapse_flags,
    decoder_directions,
    direction_cosines,
    feature_overlap,
    feature_top_examples,
    match_rate,
    reconstruction_pairs,
    top_features,
)
from layer_dynamics.runlog import init_run
from layer_dynamics.saes import DEFAULT_SAE_REPO, build_sources
from layer_dynamics.ssm import LowRankSSM, SSMDims


def _fit(u_init, delta, dims, cfg, *, seed, device, learn_z0, log):
    """One full alternating-minimisation fit → (model, refined u, train metrics)."""
    model = LowRankSSM(dims, learn_z0=learn_z0, seed=seed).to(device)
    result = alternating_minimize(model, u_init, delta, cfg, log=log)
    ev = evaluate(model, result.u, delta, lambda_l1=cfg.lambda_l1,
                  lambda_reg=cfg.lambda_reg, batch_size=cfg.batch_size)
    return model, result.u, ev


def _free_cuda(*objs) -> None:
    """Drop references and release the caching allocator — fits run one at a time on 3 GB."""
    for o in objs:
        del o
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--sae-repo", type=str, default=DEFAULT_SAE_REPO)
    parser.add_argument("--sae-cache", type=Path, default=None)
    parser.add_argument("--sources-npz", type=Path, default=None,
                        help="Load cached compressed sources (410m mlsae) instead of "
                             "building from --sae-repo (see scripts/build_410m_sources.py).")
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
    parser.add_argument("--top-n", type=int, default=100, help="Top features for stability.")
    parser.add_argument("--top-interp", type=int, default=20, help="Top features to theme.")
    parser.add_argument("--no-learn-z0", action="store_true")
    args = parser.parse_args(argv)

    run = init_run(resolve_out(args), config_path=args.config, seed=args.seed)
    mcfg = run.config.get("model", {})
    device = resolve_device(args.device or mcfg.get("device", "auto"))
    run.log(f"device: {describe(device)}")
    learn_z0 = not args.no_learn_z0

    # --- data ---
    h5 = Path("data") / mcfg["name"] / "residual_stream.h5"
    rs, rel_class, relation = load_residual_stream(h5)
    delta_all = deltas(rs)
    n, n_layers, d_model = delta_all.shape
    run.log(f"loaded {h5}: Δ {delta_all.shape}")
    if args.sources_npz:
        from layer_dynamics.saes_mlsae import load_cached_u

        u_all = load_cached_u(args.sources_npz)
        run.log(f"loaded cached compressed sources {u_all.shape} from {args.sources_npz}")
    else:
        u_all = build_sources(rs, args.sae_repo, cache_dir=args.sae_cache)
    d_input = u_all.shape[2]
    sae_l0 = float((u_all != 0).mean())
    run.log(f"sources u {u_all.shape} M={d_input} init L0={sae_l0:.4f}")

    # subject/relation metadata for the feature-theme read (aligned by position to the h5)
    meta_df = _load_meta(mcfg["name"], rel_class)
    run.log(f"metadata rows: {len(meta_df)} (aligned to residual store)")

    train_idx, _ = train_test_split_idx(n, test_frac=args.test_frac, seed=args.seed)
    u_tr = torch.from_numpy(u_all[train_idx])
    d_tr = torch.from_numpy(delta_all[train_idx])
    run.log(f"train split: {len(train_idx)} examples (seed {args.seed})")

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

    # --- main fit (seed = args.seed) for checks 1-3 ---
    run.log(f"=== main fit (seed {args.seed}) ===")
    model0, u0, ev0 = _fit(u_tr, d_tr, dims, cfg, seed=args.seed, device=device,
                           learn_z0=learn_z0, log=run.log)
    u0 = u0.cpu()  # free the 600 MB GPU input tensor; the checks below read it on CPU
    _free_cuda()
    run.log(f"main: train R²={ev0['r2']:+.3f} L0={ev0['l0']:.4f}")

    # check 1 — collapse
    collapse = collapse_flags(ev0)
    run.log(f"[1] collapse: {collapse}")

    # check 2 — reconstruction eyeball
    rng = np.random.default_rng(args.seed)
    eye_idx = sorted(rng.choice(len(train_idx), size=min(10, len(train_idx)),
                                replace=False).tolist())
    pred, actual = reconstruction_pairs(model0, u0, d_tr, eye_idx)
    per_ex_r2 = _per_example_r2(pred, actual)
    run.log(f"[2] eyeball examples {eye_idx}: per-example R² "
            f"{np.round(per_ex_r2, 2).tolist()}")
    _plot_reconstruction(pred, actual, eye_idx, per_ex_r2, run.path("figures/reconstruction.png"))

    # check 3 — feature structure / theme
    feats = top_features(u0, top_n=args.top_interp)
    feature_themes = _feature_themes(u0, feats, train_idx, meta_df)
    run.save_json("tables/feature_themes.json", feature_themes)
    purity = float(np.mean([f["purity"] for f in feature_themes])) if feature_themes else 0.0
    run.log(f"[3] top-{args.top_interp} features: mean relation-class purity {purity:.2f} "
            f"(1.0 = every top example shares the home class)")

    # --- check 4 — seed stability (re-fit under n_seeds seeds, compare top-100 dirs) ---
    run.log(f"=== seed stability ({args.n_seeds} seeds) ===")
    stab_feats = top_features(u0, top_n=args.top_n)
    fidx = [f["feature"] for f in stab_feats]
    flayer = [f["home_layer"] for f in stab_feats]
    dirs = {args.seed: decoder_directions(model0, fidx, flayer)}
    _free_cuda(model0)  # seed-0 directions extracted; free GPU for the re-fits
    extra_seeds = [s for s in range(args.seed, args.seed + args.n_seeds) if s != args.seed][
        : args.n_seeds - 1
    ]
    for s in extra_seeds:
        run.log(f"  --- seed {s} ---")
        m, u_s, ev = _fit(u_tr, d_tr, dims, cfg, seed=s, device=device,
                          learn_z0=learn_z0, log=run.log)
        run.log(f"  seed {s}: train R²={ev['r2']:+.3f} L0={ev['l0']:.4f}")
        dirs[s] = decoder_directions(m, fidx, flayer)
        _free_cuda(m, u_s)
    seed_pairs = []
    seeds = [args.seed, *extra_seeds]
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            cos = direction_cosines(dirs[seeds[i]], dirs[seeds[j]])
            seed_pairs.append({
                "seed_a": seeds[i], "seed_b": seeds[j],
                "mean_cos": float(np.mean(cos)), "median_cos": float(np.median(cos)),
                "match_rate_0.7": match_rate(cos, 0.7),
            })
    seed_match = float(np.mean([p["match_rate_0.7"] for p in seed_pairs])) if seed_pairs else 0.0
    run.log(f"[4] seed stability: mean match-rate@0.7 = {seed_match:.2f} over {len(seed_pairs)} "
            f"pair(s)")

    # --- check 5 — dataset stability (two disjoint halves) ---
    run.log("=== dataset stability (disjoint halves) ===")
    half = len(train_idx) // 2
    halves = {"A": slice(0, half), "B": slice(half, 2 * half)}
    half_top: dict[str, list] = {}
    # decoder directions of each half's top features, keyed by feature (so the two halves'
    # models never need to be resident at once — one 3 GB fit at a time).
    half_dirs: dict[str, dict[int, np.ndarray]] = {}
    for name, sl in halves.items():
        run.log(f"  --- half {name} ({sl.stop - sl.start} ex) ---")
        m, uh, ev = _fit(u_tr[sl], d_tr[sl], dims, cfg, seed=args.seed, device=device,
                         learn_z0=learn_z0, log=run.log)
        run.log(f"  half {name}: train R²={ev['r2']:+.3f} L0={ev['l0']:.4f}")
        top = top_features(uh, top_n=args.top_n)
        half_top[name] = top
        feats_h = [f["feature"] for f in top]
        layers_h = [f["home_layer"] for f in top]
        d_h = decoder_directions(m, feats_h, layers_h)
        half_dirs[name] = dict(zip(feats_h, d_h, strict=True))
        _free_cuda(m, uh)
    fa = [f["feature"] for f in half_top["A"]]
    fb = [f["feature"] for f in half_top["B"]]
    overlap = feature_overlap(fa, fb)
    shared = [f for f in fa if f in set(fb)]
    if shared:
        dcos = direction_cosines(
            np.stack([half_dirs["A"][f] for f in shared]),
            np.stack([half_dirs["B"][f] for f in shared]),
        )
        data_match = match_rate(dcos, 0.7)
        data_mean_cos = float(np.mean(dcos))
    else:
        data_match = data_mean_cos = 0.0
    run.log(f"[5] dataset stability: top-{args.top_n} Jaccard {overlap['jaccard']:.2f} "
            f"({overlap['intersection']} shared); shared-dir match@0.7 {data_match:.2f}")

    # --- consolidated verdict ---
    flags = {
        "collapse_ok": not collapse["collapsed"],
        "reconstruction_ok": bool(np.median(per_ex_r2) > 0.0),
        "seed_stability_ok": bool(seed_match >= 0.7),
        "dataset_overlap": overlap["jaccard"],
    }
    stats = {
        "model": mcfg.get("name"), "device": device, "dims": vars(dims),
        "hyperparams": cfg.__dict__, "n_train": len(train_idx),
        "main_train": ev0, "collapse": collapse,
        "eyeball_idx": eye_idx, "eyeball_per_example_r2": per_ex_r2.tolist(),
        "feature_theme_purity": purity,
        "seed_stability": {"pairs": seed_pairs, "mean_match_rate_0.7": seed_match},
        "dataset_stability": {"overlap": overlap, "shared_dir_match_0.7": data_match,
                              "shared_dir_mean_cos": data_mean_cos},
        "flags": flags,
    }
    run.save_json("tables/sanity_stats.json", stats)

    import pandas as pd

    pd.DataFrame(seed_pairs).to_csv(run.path("tables/seed_pairs.csv"), index=False)

    print(f"\n=== Phase 3 Step 4 sanity ({mcfg.get('name')}) ===")
    print(f"  [1] collapse-to-bias: {'YES — investigate' if collapse['collapsed'] else 'no'} "
          f"(|ops|={collapse['operator_total_fro']:.1f} vs |b|={collapse['b_norm']:.1f})")
    print(f"  [2] reconstruction: median per-example R² {np.median(per_ex_r2):+.2f} "
          f"(10 eyeballed)")
    print(f"  [3] feature themes: mean relation-class purity {purity:.2f}")
    print(f"  [4] seed stability: match-rate@0.7 {seed_match:.2f} "
          f"({'PASS' if seed_match >= 0.7 else 'below 0.7'})")
    print(f"  [5] dataset stability: Jaccard {overlap['jaccard']:.2f}, "
          f"shared-dir match@0.7 {data_match:.2f}")
    ok = flags["collapse_ok"] and flags["reconstruction_ok"]
    print("sanity:", "OK (no red flags)" if ok else "RED FLAG — see run.log")
    return 0 if ok else 1


def _load_meta(model_name: str, rel_class: np.ndarray):
    """Load subject/relation metadata aligned by position to the residual store."""
    import pandas as pd

    df = pd.read_parquet(Path("data") / model_name / "filtered_triples.parquet")
    df = df.reset_index(drop=True)
    if len(df) != len(rel_class):
        raise ValueError(f"metadata/store length mismatch: {len(df)} vs {len(rel_class)}")
    mism = int((df["relation_class"].to_numpy() != rel_class).sum())
    if mism:
        raise ValueError(f"metadata not aligned to store: {mism} relation_class mismatches")
    return df


def _per_example_r2(pred: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Per-example reconstruction R² vs that example's own zero baseline ‖Δ‖²."""
    ss_res = ((actual - pred) ** 2).reshape(len(pred), -1).sum(axis=1)
    ss_tot = (actual ** 2).reshape(len(pred), -1).sum(axis=1)
    return 1.0 - ss_res / np.clip(ss_tot, 1e-12, None)


def _feature_themes(u, feats, train_idx, meta_df) -> list[dict]:
    """For each top feature, the relation-class theme of its strongest-activating examples."""
    out = []
    for f in feats:
        ex_local = feature_top_examples(u, f["feature"], f["home_layer"], top_e=10)
        rows = meta_df.iloc[[int(train_idx[i]) for i in ex_local]]
        classes = rows["relation_class"].tolist()
        top_class = max(set(classes), key=classes.count) if classes else None
        purity = classes.count(top_class) / len(classes) if classes else 0.0
        out.append({
            "feature": f["feature"], "home_layer": f["home_layer"], "mass": f["mass"],
            "top_class": top_class, "purity": purity,
            "examples": rows["subject"].tolist(), "relations": classes,
        })
    return out


def _plot_reconstruction(pred, actual, idx, per_ex_r2, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(idx)
    ncol = 5
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3 * ncol, 2.6 * nrow), squeeze=False)
    lim = float(np.abs(actual).max())
    for k in range(nrow * ncol):
        ax = axes[k // ncol][k % ncol]
        if k >= n:
            ax.axis("off")
            continue
        ax.scatter(actual[k].ravel(), pred[k].ravel(), s=2, alpha=0.3)
        ax.plot([-lim, lim], [-lim, lim], "r:", lw=0.8)
        ax.set_title(f"ex {idx[k]}  R²={per_ex_r2[k]:+.2f}", fontsize=8)
        ax.set_xlabel("actual Δ", fontsize=7)
        ax.set_ylabel("pred Δ", fontsize=7)
    fig.suptitle("Branch B reconstruction: predicted vs actual Δ (all layers)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
