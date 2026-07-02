#!/usr/bin/env python3
"""Phase 3 Step 5 — Gate-3 reconstruction-vs-sparsity Pareto (70m-deduped debug lane).

Sweeps ``λ_L1`` to trace the held-out reconstruction-R²-vs-L0 Pareto curve for the
Branch B SSM, and against the matched **per-layer sparse-dictionary baseline** — the
same model with the cross-layer state ablated (``use_state=False``: ``Δ_ℓ ≈ b_ℓ +
D_ℓ u_ℓ``, a per-layer SAE on ``Δ_ℓ`` fit with the identical optimiser, source init and
L1 sweep). The only thing the full SSM adds over the baseline is the cross-layer state
term ``C_ℓ z_ℓ``, so the vertical gap between the two curves at matched L0 *is* the
reconstruction value of structured cross-layer propagation — the Gate-3 go criterion
(plan Step 5): "reconstruction on or above the per-layer SAE Pareto at comparable
sparsity" (seed stability ≥ 0.7 already cleared at Step 4, match-rate 0.99).

    python scripts/gate3_pareto.py --config configs/pythia-70m.yaml --desc ph3-gate3-70md \
        --rank 32 --lambdas 0.1,0.3,1,3,10 --outer-iters 10 --device cuda

Each (λ, variant) point is a full alternating-minimisation fit on train + held-out
inference with operators frozen, so the loop is heavy (2·len(λ) fits). The GPU is freed
between fits (3 GB GTX 1050 OOMs otherwise — same as Step 4).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from layer_dynamics.cli import add_common_args, resolve_out
from layer_dynamics.decompose import AltMinConfig, alternating_minimize, evaluate, infer_inputs
from layer_dynamics.device import describe, resolve_device
from layer_dynamics.diagnostics.dataset import deltas, load_residual_stream, train_test_split_idx
from layer_dynamics.runlog import init_run
from layer_dynamics.saes import DEFAULT_SAE_REPO, build_sources
from layer_dynamics.ssm import LowRankSSM, SSMDims


def _free_cuda(*objs) -> None:
    for o in objs:
        del o
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _fit_eval(u_tr, d_tr, u_te, d_te, dims, cfg, *, use_state, seed, device, learn_z0, log):
    """One full fit (train) + held-out inference (operators frozen) → train/test metrics."""
    model = LowRankSSM(dims, learn_z0=learn_z0, seed=seed, use_state=use_state).to(device)
    result = alternating_minimize(model, u_tr, d_tr, cfg, log=log)
    u_te_fit = infer_inputs(model, u_te, d_te, lambda_l1=cfg.lambda_l1,
                            n_steps=max(cfg.input_steps * 4, 80),
                            lipschitz=result.lipschitz, batch_size=cfg.batch_size)
    train_ev = evaluate(model, result.u, d_tr, lambda_l1=cfg.lambda_l1,
                        lambda_reg=cfg.lambda_reg, batch_size=cfg.batch_size)
    test_ev = evaluate(model, u_te_fit, d_te, lambda_l1=cfg.lambda_l1,
                       lambda_reg=cfg.lambda_reg, batch_size=cfg.batch_size)
    _free_cuda(model, result, u_te_fit)
    return train_ev, test_ev


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
    parser.add_argument("--lambdas", type=str, default="0.1,0.3,1,3,10",
                        help="Comma-separated λ_L1 grid (memory: knee ≈ 1 on 70m).")
    parser.add_argument("--lambda-reg", type=float, default=1e-4)
    parser.add_argument("--outer-iters", type=int, default=10)
    parser.add_argument("--operator-steps", type=int, default=40)
    parser.add_argument("--operator-lr", type=float, default=1e-2)
    parser.add_argument("--input-steps", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--no-learn-z0", action="store_true")
    parser.add_argument("--h5", type=Path, default=None,
                        help="Override residual-stream HDF5 (e.g. the untrained-model "
                             "store for the T4 reconstruction control; pair with the "
                             "matching --sources-npz).")
    parser.add_argument("--split-by", choices=["random", "object", "subject"],
                        default="random",
                        help="R5: entity-disjoint train/test split keyed on the "
                             "filtered_triples.parquet column (objects recur heavily; "
                             "random shares answers across the boundary).")
    args = parser.parse_args(argv)

    lambdas = [float(x) for x in args.lambdas.split(",") if x.strip()]

    run = init_run(resolve_out(args), config_path=args.config, seed=args.seed)
    mcfg = run.config.get("model", {})
    device = resolve_device(args.device or mcfg.get("device", "auto"))
    run.log(f"device: {describe(device)}")
    learn_z0 = not args.no_learn_z0

    # --- data ---
    h5 = args.h5 if args.h5 is not None else Path("data") / mcfg["name"] / "residual_stream.h5"
    rs, _, _ = load_residual_stream(h5)
    delta = deltas(rs)
    n, n_layers, d_model = delta.shape
    run.log(f"loaded {h5}: Δ {delta.shape}")
    if args.sources_npz:
        from layer_dynamics.saes_mlsae import load_cached_u

        u = load_cached_u(args.sources_npz)
        run.log(f"loaded cached compressed sources {u.shape} from {args.sources_npz}")
    else:
        u = build_sources(rs, args.sae_repo, cache_dir=args.sae_cache)
    if u.shape[1] < n_layers:  # partial-coverage SAE suite (e.g. Qwen blocks 0..11 of 24)
        n_layers = u.shape[1]
        delta = delta[:, :n_layers]
        run.log(f"partial SAE coverage -> truncating to first {n_layers} layers (Δ {delta.shape})")
    d_input = u.shape[2]
    sae_l0 = float((u != 0).mean())
    run.log(f"sources u {u.shape} M={d_input} init L0={sae_l0:.4f}")

    if args.split_by != "random":
        from layer_dynamics.diagnostics.dataset import entity_split_idx

        pq = Path("data") / mcfg["name"] / "filtered_triples.parquet"
        train_idx, test_idx = entity_split_idx(n, pq, args.split_by,
                                               test_frac=args.test_frac, seed=args.seed)
    else:
        train_idx, test_idx = train_test_split_idx(n, test_frac=args.test_frac, seed=args.seed)
    u_tr = torch.from_numpy(u[train_idx])
    u_te = torch.from_numpy(u[test_idx])
    d_tr = torch.from_numpy(delta[train_idx])
    d_te = torch.from_numpy(delta[test_idx])
    run.log(f"split: {len(train_idx)} train / {len(test_idx)} test "
            f"(by={args.split_by}, seed {args.seed})")

    d_state = args.d_state if args.d_state is not None else args.rank
    dims = SSMDims(n_layers=n_layers, d_model=d_model, d_input=d_input,
                   d_state=d_state, rank=args.rank)
    run.log(f"dims: L={n_layers} D={d_model} M={d_input} d_z={d_state} r={args.rank}; "
            f"λ_L1 grid={lambdas}")

    variants = {"ssm": True, "baseline": False}  # full Branch B vs no-state per-layer dict
    rows: list[dict] = []
    for lam in lambdas:
        cfg = AltMinConfig(
            lambda_l1=lam, lambda_reg=args.lambda_reg, outer_iters=args.outer_iters,
            operator_steps=args.operator_steps, operator_lr=args.operator_lr,
            input_steps=args.input_steps, batch_size=args.batch_size,
        )
        for vname, use_state in variants.items():
            run.log(f"=== {vname}  λ_L1={lam:g} ===")
            tr_ev, te_ev = _fit_eval(u_tr, d_tr, u_te, d_te, dims, cfg, use_state=use_state,
                                     seed=args.seed, device=device, learn_z0=learn_z0,
                                     log=run.log)
            row = {
                "variant": vname, "lambda_l1": lam,
                "train_r2": tr_ev["r2"], "test_r2": te_ev["r2"],
                "train_l0": tr_ev["l0"], "test_l0": te_ev["l0"],
                "test_recon": te_ev["recon"],
                "op_C": te_ev["op_C"], "op_D": te_ev["op_D"], "b_norm": te_ev["b_norm"],
            }
            rows.append(row)
            run.log(f"  {vname} λ={lam:g}: test R²={te_ev['r2']:+.3f} test L0={te_ev['l0']:.4f} "
                    f"(train R²={tr_ev['r2']:+.3f}) |C|={te_ev['op_C']:.1f} "
                    f"|D|={te_ev['op_D']:.1f}")

    # --- Pareto comparison: SSM vs baseline at matched L0 (interp baseline onto SSM L0) ---
    ssm_rows = sorted((r for r in rows if r["variant"] == "ssm"), key=lambda r: r["test_l0"])
    base_rows = sorted((r for r in rows if r["variant"] == "baseline"), key=lambda r: r["test_l0"])
    gaps = _matched_l0_gaps(ssm_rows, base_rows)
    on_or_above = all(g["gap"] >= -1e-3 for g in gaps) if gaps else False

    stats = {
        "model": mcfg.get("name"), "device": device, "sae_repo": args.sae_repo,
        "dims": vars(dims), "lambdas": lambdas, "lambda_reg": args.lambda_reg,
        "outer_iters": args.outer_iters, "sae_init_l0": sae_l0,
        "split_by": args.split_by,
        "n_train": len(train_idx), "n_test": len(test_idx),
        "rows": rows, "matched_l0_gaps": gaps,
        "ssm_on_or_above_baseline": on_or_above,
    }
    run.save_json("tables/gate3_pareto.json", stats)

    import pandas as pd

    pd.DataFrame(rows).to_csv(run.path("tables/pareto.csv"), index=False)
    _plot_pareto(rows, run.path("figures/pareto.svg"))

    print(f"\n=== Phase 3 Gate-3 Pareto ({mcfg.get('name')}) ===")
    print(f"  {'λ_L1':>6} {'variant':>9} {'test L0':>9} {'test R²':>9}")
    for r in sorted(rows, key=lambda r: (r["lambda_l1"], r["variant"])):
        print(f"  {r['lambda_l1']:>6g} {r['variant']:>9} {r['test_l0']:>9.4f} "
              f"{r['test_r2']:>+9.3f}")
    print("  --- SSM vs per-layer baseline at matched L0 (SSM L0 → interp baseline R²) ---")
    for g in gaps:
        verdict = "SSM≥base" if g["gap"] >= -1e-3 else "SSM<base"
        print(f"  L0={g['l0']:.4f}: SSM R²={g['ssm_r2']:+.3f} vs base {g['baseline_r2']:+.3f} "
              f"(gap {g['gap']:+.3f}) {verdict}")
    print(f"go criterion (recon on/above baseline Pareto): "
          f"{'MET' if on_or_above else 'NOT met'}")
    return 0


def _matched_l0_gaps(ssm_rows: list[dict], base_rows: list[dict]) -> list[dict]:
    """At each SSM operating point, the baseline test-R² linearly interpolated to the same
    L0, and the SSM−baseline gap. Points outside the baseline L0 range are skipped (no
    honest interpolation)."""
    if not base_rows:
        return []
    bl0 = [r["test_l0"] for r in base_rows]
    br2 = [r["test_r2"] for r in base_rows]
    lo, hi = bl0[0], bl0[-1]
    gaps = []
    for s in ssm_rows:
        x = s["test_l0"]
        if x < lo or x > hi:
            continue
        # linear interp of baseline R² at L0 = x
        j = next(k for k in range(1, len(bl0)) if bl0[k] >= x)
        x0, x1, y0, y1 = bl0[j - 1], bl0[j], br2[j - 1], br2[j]
        bi = y0 if x1 == x0 else y0 + (y1 - y0) * (x - x0) / (x1 - x0)
        gaps.append({"l0": x, "ssm_r2": s["test_r2"], "baseline_r2": bi,
                     "gap": s["test_r2"] - bi, "lambda_l1": s["lambda_l1"]})
    return gaps


def _plot_pareto(rows: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    styles = {"ssm": ("o-", "C0", "Branch B SSM (with state)"),
              "baseline": ("s--", "C1", "per-layer dict (no state)")}
    for vname, (st, col, label) in styles.items():
        pts = sorted((r for r in rows if r["variant"] == vname), key=lambda r: r["test_l0"])
        if not pts:
            continue
        ax.plot([p["test_l0"] for p in pts], [p["test_r2"] for p in pts], st, color=col,
                label=label)
        for p in pts:
            ax.annotate(f"λ={p['lambda_l1']:g}", (p["test_l0"], p["test_r2"]),
                        fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("held-out input density (L0 fraction)")
    ax.set_ylabel("held-out reconstruction R²")
    ax.set_title("Gate-3 Pareto: Branch B vs per-layer baseline")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
