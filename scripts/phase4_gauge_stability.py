#!/usr/bin/env python3
"""Phase 4 S1 — gauge-invariant seed-stability of the Branch B decomposition.

The recorded Gate-3 limitation is a seed instability of the **raw decoder directions**:
the feedthrough column ``D_{ℓ*}[:, j]`` matches across seeds at only **0.45 < 0.7**, even
though every seed reaches the *same* reconstruction (R² ≈ 0.880). Diagnosed as a structural
gauge degeneracy (the SSM's state↔feedthrough trade-off, Symmetry B; isotropic λ_reg can't
break it). If the instability is a property of the *parameterisation* and not the learned
*function*, then quantities that are invariant under the SSM's gauge freedoms should be
seed-stable. This script measures exactly that (the plan's Fix 5):

  1. **Per-feature data-averaged Δ-contribution** — the gauge-invariant analogue of the raw
     decoder column. Because the SSM is linear in ``u``, this is each feature's *impulse
     response* ``g_j ∈ R^{L×D}`` (``feature_impulse_responses``): the feedthrough at the home
     layer **plus** the downstream cross-layer state propagation. Its home-layer slice equals
     the raw ``D`` column (the 0.45 metric); the full ``g_j`` is what the feature actually does
     to ``Δ``. Reported as the per-feature cosine across seeds (match@0.7, mean), head-to-head
     with the raw-``D`` cosine on the *same* features.
  2. **Top-k decoder subspace overlap** — Grassmann overlap (``subspace_overlap``) of the span
     of the top-k ``D`` columns: are the directions individually shuffled but the subspace
     shared?
  3. **Eigenvalue spectrum of the transitions ``A_ℓ``** — invariant under the state similarity
     gauge (Symmetry A); per-layer sorted-moduli cosine across seeds.

It reloads the canonical seed-0 fit (``phase4_checkpoint.py``'s ``ssm_seed0.pt`` + split +
``u_infer_full.npy`` for the feature ranking) and **refits** the extra seeds with the exact
Gate-3 knee config, so the raw-``D`` number it recomputes reproduces the recorded 0.45 as the
control. One reload + ``n_seeds-1`` full fits; GPU freed between fits (expandable_segments).

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/phase4_gauge_stability.py \
        --config configs/pythia-410m-deduped.yaml --desc ph4-gauge-410md \
        --sources-npz data/pythia-410m-deduped/sources_mlsae.npz \
        --checkpoint runs/2026-06-26-ph4-checkpoint-410md/checkpoints \
        --rank 32 --lambda-l1 1.0 --outer-iters 10 --n-seeds 3 --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from layer_dynamics.checkpoint import load_ssm, save_ssm
from layer_dynamics.cli import add_common_args, resolve_out
from layer_dynamics.decompose import AltMinConfig, alternating_minimize, evaluate
from layer_dynamics.device import describe, resolve_device
from layer_dynamics.diagnostics.dataset import deltas, load_residual_stream, train_test_split_idx
from layer_dynamics.diagnostics.decompose_sanity import (
    decoder_directions,
    direction_cosines,
    feature_impulse_responses,
    match_rate,
    state_eigenvalue_moduli,
    subspace_overlap,
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
    model = LowRankSSM(dims, learn_z0=learn_z0, seed=seed).to(device)
    result = alternating_minimize(model, u_tr, d_tr, cfg, log=log)
    return model, result


def _seed_invariants(model, fidx, flayer):
    """Extract the per-seed quantities we compare across seeds (all on CPU numpy)."""
    raw = decoder_directions(model, fidx, flayer)  # [k, D] gauge-VARIANT (the 0.45 metric)
    g = feature_impulse_responses(model, fidx, flayer)  # [k, L, D] gauge-invariant
    eig = state_eigenvalue_moduli(model)  # [L-1, d_z] gauge-invariant (Symmetry A)
    return {"raw": raw, "impulse": g.reshape(g.shape[0], -1), "impulse_LD": g, "eig": eig}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_args(parser)
    parser.add_argument("--sources-npz", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Dir holding ssm_seed0.pt + split.npz + u_infer_full.npy "
                             "(phase4_checkpoint.py output).")
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
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--top-n", type=int, default=200,
                        help="Top features by full-dataset |u| mass (match phase4_checkpoint).")
    parser.add_argument("--top-k-subspace", type=int, default=100,
                        help="Top-k decoder directions for the subspace-overlap metric.")
    parser.add_argument("--no-learn-z0", action="store_true")
    args = parser.parse_args(argv)

    run = init_run(resolve_out(args), config_path=args.config, seed=args.seed)
    mcfg = run.config.get("model", {})
    device = resolve_device(args.device or mcfg.get("device", "auto"))
    run.log(f"device: {describe(device)}")
    learn_z0 = not args.no_learn_z0

    # --- data + cached sources + the canonical split ---
    h5 = Path("data") / mcfg["name"] / "residual_stream.h5"
    rs, _, _ = load_residual_stream(h5)
    delta_all = deltas(rs)
    n, n_layers, d_model = delta_all.shape
    u_all = load_cached_u(args.sources_npz)
    d_input = u_all.shape[2]
    run.log(f"Δ {delta_all.shape}; sources u {u_all.shape} M={d_input}")

    split_path = args.checkpoint / "split.npz"
    if split_path.exists():
        sp = np.load(split_path)
        train_idx = sp["train_idx"]
        run.log(f"loaded canonical split: {len(train_idx)} train (from {split_path})")
    else:
        train_idx, _ = train_test_split_idx(n, test_frac=args.test_frac, seed=args.seed)
        run.log(f"split.npz absent → recomputed train split: {len(train_idx)} (seed {args.seed})")
    u_tr = torch.from_numpy(u_all[train_idx])
    d_tr = torch.from_numpy(delta_all[train_idx])

    # rank features the same way phase4_checkpoint does (full-dataset frozen-inference |u| mass)
    u_full_path = args.checkpoint / "u_infer_full.npy"
    rank_u = torch.from_numpy(np.load(u_full_path)) if u_full_path.exists() \
        else torch.from_numpy(u_all)
    feats = top_features(rank_u, top_n=args.top_n)
    fidx = [f["feature"] for f in feats]
    flayer = [f["home_layer"] for f in feats]
    run.log(f"candidate features: top-{len(fidx)} by full-dataset |u| mass "
            f"({'frozen-inference' if u_full_path.exists() else 'SAE-init'} ranking)")
    _free_cuda(rank_u)

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

    # --- per-seed invariants: seed 0 from the checkpoint, the rest refit ---
    inv: dict[int, dict] = {}
    ckpt0 = args.checkpoint / "ssm_seed0.pt"
    run.log(f"=== seed {args.seed} (canonical checkpoint {ckpt0}) ===")
    model0 = load_ssm(ckpt0, device=device)
    inv[args.seed] = _seed_invariants(model0, fidx, flayer)
    _free_cuda(model0)

    extra_seeds = list(range(args.seed + 1, args.seed + args.n_seeds))[: args.n_seeds - 1]
    for s in extra_seeds:
        run.log(f"=== seed {s} (refit) ===")
        m, res = _fit_train(u_tr, d_tr, dims, cfg, seed=s, device=device,
                            learn_z0=learn_z0, log=run.log)
        ev = evaluate(m, res.u, d_tr, lambda_l1=cfg.lambda_l1, lambda_reg=cfg.lambda_reg,
                      batch_size=cfg.batch_size)
        run.log(f"  seed {s}: train R²={ev['r2']:+.3f} L0={ev['l0']:.4f}")
        save_ssm(m, run.path(f"checkpoints/ssm_seed{s}.pt"),
                 meta={"model": mcfg.get("name"), "seed": s, "train_r2": ev["r2"]})
        inv[s] = _seed_invariants(m, fidx, flayer)
        _free_cuda(m, res)

    # --- cross-seed comparison ---
    seeds = [args.seed, *extra_seeds]
    pairs = [(a, b) for i, a in enumerate(seeds) for b in seeds[i + 1:]]
    k_sub = min(args.top_k_subspace, len(fidx))

    raw_cos, imp_cos, raw_sub, imp_sub, eig_cos_mean = [], [], [], [], []
    per_pair = []
    for a, b in pairs:
        rc = direction_cosines(inv[a]["raw"], inv[b]["raw"])          # gauge-variant
        ic = direction_cosines(inv[a]["impulse"], inv[b]["impulse"])  # gauge-invariant
        rs_ = subspace_overlap(inv[a]["raw"][:k_sub], inv[b]["raw"][:k_sub])
        is_ = subspace_overlap(inv[a]["impulse"][:k_sub], inv[b]["impulse"][:k_sub])
        ec = direction_cosines(inv[a]["eig"], inv[b]["eig"])  # per-layer spectrum cosine
        raw_cos.append(rc); imp_cos.append(ic)
        raw_sub.append(rs_); imp_sub.append(is_); eig_cos_mean.append(float(ec.mean()))
        per_pair.append({
            "seed_a": a, "seed_b": b,
            "raw_D_match@0.7": match_rate(rc, 0.7), "raw_D_mean_cos": float(rc.mean()),
            "impulse_match@0.7": match_rate(ic, 0.7), "impulse_mean_cos": float(ic.mean()),
            "raw_D_subspace_overlap": rs_, "impulse_subspace_overlap": is_,
            "eig_spectrum_mean_cos": float(ec.mean()),
        })
        run.log(f"  pair {a}-{b}: raw-D match@0.7={match_rate(rc, 0.7):.2f} "
                f"(mean cos {rc.mean():+.2f}) | impulse match@0.7={match_rate(ic, 0.7):.2f} "
                f"(mean cos {ic.mean():+.2f}) | subspace raw={rs_:.2f} imp={is_:.2f} | "
                f"A-eig spectrum cos={ec.mean():+.2f}")

    raw_cos_all = np.concatenate(raw_cos) if raw_cos else np.array([])
    imp_cos_all = np.concatenate(imp_cos) if imp_cos else np.array([])

    # how much of each feature's contribution lives BEYOND the home layer (why g != D)
    downstream_frac = []
    g0 = inv[args.seed]["impulse_LD"]  # [k, L, D]
    for t, ell in enumerate(flayer):
        total = float(np.abs(g0[t]).sum())
        home = float(np.abs(g0[t, ell]).sum())
        downstream_frac.append(1.0 - home / total if total > 0 else 0.0)
    downstream_frac = float(np.mean(downstream_frac))

    summary = {
        "raw_D_match@0.7": float(match_rate(raw_cos_all, 0.7)),
        "raw_D_mean_cos": float(raw_cos_all.mean()) if len(raw_cos_all) else 0.0,
        "impulse_match@0.7": float(match_rate(imp_cos_all, 0.7)),
        "impulse_mean_cos": float(imp_cos_all.mean()) if len(imp_cos_all) else 0.0,
        "raw_D_subspace_overlap_mean": float(np.mean(raw_sub)) if raw_sub else 0.0,
        "impulse_subspace_overlap_mean": float(np.mean(imp_sub)) if imp_sub else 0.0,
        "eig_spectrum_mean_cos": float(np.mean(eig_cos_mean)) if eig_cos_mean else 0.0,
        "mean_downstream_contribution_fraction": downstream_frac,
    }
    stats = {
        "model": mcfg.get("name"), "device": device, "dims": vars(dims),
        "hyperparams": cfg.__dict__, "seeds": seeds, "n_features": len(fidx),
        "top_k_subspace": k_sub, "summary": summary, "pairs": per_pair,
    }
    run.save_json("tables/gauge_stability.json", stats)

    import pandas as pd

    pd.DataFrame(per_pair).to_csv(run.path("tables/gauge_pairs.csv"), index=False)
    _plot(raw_cos_all, imp_cos_all, summary, run.path("figures/gauge_stability.svg"))

    print(f"\n=== Phase 4 S1 — gauge-invariant seed stability ({mcfg.get('name')}) ===")
    print(f"  features compared: {len(fidx)} (top by |u| mass); seeds {seeds}")
    print(f"  raw decoder column D[:,j]   match@0.7 = {summary['raw_D_match@0.7']:.2f}  "
          f"(mean cos {summary['raw_D_mean_cos']:+.2f})   ← the recorded limitation")
    print(f"  gauge-invariant Δ-contrib   match@0.7 = {summary['impulse_match@0.7']:.2f}  "
          f"(mean cos {summary['impulse_mean_cos']:+.2f})   ← impulse response g_j")
    print(f"  top-{k_sub} subspace overlap   raw-D = {summary['raw_D_subspace_overlap_mean']:.2f}  "
          f"impulse = {summary['impulse_subspace_overlap_mean']:.2f}")
    print(f"  A_ℓ eigenvalue-spectrum cos = {summary['eig_spectrum_mean_cos']:+.2f}  "
          f"(per-layer, sorted moduli)")
    print(f"  mean contribution beyond home layer = {downstream_frac:.0%} "
          f"(why g_j differs from the raw D column)")
    verdict = "COSMETIC (gauge artefact)" if summary["impulse_match@0.7"] >= 0.7 \
        else "partly genuine (function differs across seeds)"
    print(f"  verdict: the raw-D instability is {verdict}")
    return 0


def _plot(raw_cos, imp_cos, summary, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    bins = np.linspace(-1, 1, 41)
    ax.hist(raw_cos, bins=bins, alpha=0.6, label=f"raw D[:,j]  (match@0.7 "
            f"{summary['raw_D_match@0.7']:.2f})", color="tab:red")
    ax.hist(imp_cos, bins=bins, alpha=0.6, label=f"gauge-inv Δ-contrib  (match@0.7 "
            f"{summary['impulse_match@0.7']:.2f})", color="tab:blue")
    ax.axvline(0.7, color="k", ls=":", lw=1, label="0.7 threshold")
    ax.set_xlabel("cross-seed cosine")
    ax.set_ylabel("# features")
    ax.set_title("Per-feature seed stability:\nraw decoder column vs gauge-invariant contribution")
    ax.legend(fontsize=8)

    ax = axes[1]
    labels = ["raw-D\nper-dir", "Δ-contrib\nper-dir", "raw-D\nsubspace",
              "Δ-contrib\nsubspace", "A-eig\nspectrum"]
    vals = [summary["raw_D_match@0.7"], summary["impulse_match@0.7"],
            summary["raw_D_subspace_overlap_mean"], summary["impulse_subspace_overlap_mean"],
            summary["eig_spectrum_mean_cos"]]
    colors = ["tab:red", "tab:blue", "tab:red", "tab:blue", "tab:green"]
    ax.bar(labels, vals, color=colors, alpha=0.75)
    ax.axhline(0.7, color="k", ls=":", lw=1)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("stability (match@0.7 / overlap / cos)")
    ax.set_title("Gauge-invariant quantities are seed-stable")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
