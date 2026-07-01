#!/usr/bin/env python3
"""Debug the Branch B alternating-minimisation optimiser on 70m-deduped (Phase 3 Step 3, M3).

End-to-end shake-out of the decomposition optimiser the plan calls for before the primary
run: load the collected residual streams, build SAE-initialised inputs ``u`` (Phase 3
Step 2), fit the low-rank SSM operators on a train split by alternating
Adam-over-operators / FISTA-over-inputs (:mod:`layer_dynamics.decompose`), then infer ``u``
on a held-out test split with the operators **frozen** and report reconstruction.

This is a single ``(rank, λ_L1)`` point, not the full Gate-3 Pareto sweep — its job is to
confirm the optimiser descends, sparsifies with *exact* zeros, generalises to held-out
examples, and does **not** collapse to "biases absorb the per-layer mean" (Step 4 sanity #1,
watched via the per-family operator norms vs ``‖b‖``).

    python scripts/step3_decompose.py --config configs/pythia-70m.yaml --desc ph3-decomp-70md \
        --rank 32 --lambda-l1 1e-2 --device cpu

The held-out R² is the honest number: operators are shared parameters (must generalise),
the inputs are a per-example code (re-inferred on test). A large train–test R² gap would
mean the operators are memorising.
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--sae-repo", type=str, default=DEFAULT_SAE_REPO)
    parser.add_argument("--sae-cache", type=Path, default=None)
    parser.add_argument("--sources-npz", type=Path, default=None,
                        help="Load cached compressed sources (410m mlsae) instead of "
                             "building from --sae-repo (see scripts/build_410m_sources.py).")
    parser.add_argument("--device", type=str, default=None,
                        help="Override the config device (cpu | cuda | auto).")
    parser.add_argument("--rank", type=int, default=32, help="Low-rank r for A/B/C/D.")
    parser.add_argument("--d-state", type=int, default=None,
                        help="SSM latent dim d_z (default: = rank).")
    parser.add_argument("--lambda-l1", type=float, default=1e-2)
    parser.add_argument("--lambda-reg", type=float, default=1e-4)
    parser.add_argument("--outer-iters", type=int, default=15)
    parser.add_argument("--operator-steps", type=int, default=40)
    parser.add_argument("--operator-lr", type=float, default=1e-2)
    parser.add_argument("--input-steps", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--no-learn-z0", action="store_true",
                        help="Fix z_0 = 0 instead of learning it.")
    args = parser.parse_args(argv)

    run = init_run(resolve_out(args), config_path=args.config, seed=args.seed)
    mcfg = run.config.get("model", {})
    device = resolve_device(args.device or mcfg.get("device", "auto"))
    run.log(f"device: {describe(device)}")

    # --- data: residual streams -> Δ targets, SAE-initialised inputs u ---
    h5 = Path("data") / mcfg["name"] / "residual_stream.h5"
    rs, _, _ = load_residual_stream(h5)  # [N, L+1, D]
    delta = deltas(rs)  # [N, L, D]
    n, n_layers, d_model = delta.shape
    run.log(f"loaded {h5}: residual_stream {rs.shape} -> Δ {delta.shape}")
    if args.sources_npz:
        from layer_dynamics.saes_mlsae import load_cached_u

        u = load_cached_u(args.sources_npz)  # [N, L, M*] compressed mlsae sources
        run.log(f"loaded cached compressed sources {u.shape} from {args.sources_npz}")
    else:
        run.log(f"building SAE sources from {args.sae_repo} ...")
        u = build_sources(rs, args.sae_repo, cache_dir=args.sae_cache)  # [N, L, M]
    d_input = u.shape[2]
    sae_l0 = float((u != 0).mean())
    run.log(f"sources u {u.shape}  M={d_input}  init L0={sae_l0:.4f} "
            f"(u_0 zero-init; v_0 has no embedding SAE)")

    # --- train/test split (operators generalise; inputs are a per-example code) ---
    train_idx, test_idx = train_test_split_idx(n, test_frac=args.test_frac, seed=args.seed)
    u_tr = torch.from_numpy(u[train_idx])
    u_te = torch.from_numpy(u[test_idx])
    d_tr = torch.from_numpy(delta[train_idx])
    d_te = torch.from_numpy(delta[test_idx])
    run.log(f"split: {len(train_idx)} train / {len(test_idx)} test (seed {args.seed})")

    # --- model: first-order, fully depth-indexed low-rank SSM (Branch B) ---
    from layer_dynamics.ssm import LowRankSSM, SSMDims

    d_state = args.d_state if args.d_state is not None else args.rank
    dims = SSMDims(n_layers=n_layers, d_model=d_model, d_input=d_input,
                   d_state=d_state, rank=args.rank)
    model = LowRankSSM(dims, learn_z0=not args.no_learn_z0, seed=args.seed).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    run.log(f"LowRankSSM L={n_layers} D={d_model} M={d_input} d_z={d_state} r={args.rank} "
            f"learn_z0={not args.no_learn_z0}  params={n_params:,}")

    cfg = AltMinConfig(
        lambda_l1=args.lambda_l1, lambda_reg=args.lambda_reg, outer_iters=args.outer_iters,
        operator_steps=args.operator_steps, operator_lr=args.operator_lr,
        input_steps=args.input_steps, batch_size=args.batch_size,
    )
    run.log(f"alt-min: λ_L1={cfg.lambda_l1:g} λ_reg={cfg.lambda_reg:g} "
            f"outer={cfg.outer_iters} op_steps={cfg.operator_steps} in_steps={cfg.input_steps} "
            f"batch={cfg.batch_size}")
    result = alternating_minimize(model, u_tr, d_tr, cfg, log=run.log)

    # --- held-out: infer u on test with operators frozen, then evaluate ---
    run.log("inferring held-out inputs (operators frozen) ...")
    u_te_fit = infer_inputs(model, u_te, d_te, lambda_l1=cfg.lambda_l1,
                            n_steps=max(cfg.input_steps * 4, 80),
                            lipschitz=result.lipschitz, batch_size=cfg.batch_size)
    train_ev = evaluate(model, result.u, d_tr, lambda_l1=cfg.lambda_l1,
                        lambda_reg=cfg.lambda_reg, batch_size=cfg.batch_size)
    test_ev = evaluate(model, u_te_fit, d_te, lambda_l1=cfg.lambda_l1,
                       lambda_reg=cfg.lambda_reg, batch_size=cfg.batch_size)

    # --- collapse check (Step 4 sanity #1): bias should NOT dwarf the operators ---
    op_total = (train_ev["op_A"] ** 2 + train_ev["op_B"] ** 2
                + train_ev["op_C"] ** 2 + train_ev["op_D"] ** 2) ** 0.5
    collapsed = bool(train_ev["l0"] < 1e-4 and op_total < 1e-3 * train_ev["b_norm"])

    stats = {
        "model": mcfg.get("name"), "device": device, "sae_repo": args.sae_repo,
        "n_train": len(train_idx), "n_test": len(test_idx),
        "dims": {"L": n_layers, "D": d_model, "M": d_input, "d_z": d_state, "r": args.rank},
        "hyperparams": vars(cfg) if hasattr(cfg, "__dict__") else cfg.__dict__,
        "sae_init_l0": sae_l0,
        "train": train_ev, "test": test_ev,
        "init_loss": result.history[0]["loss"], "final_loss": result.history[-1]["loss"],
        "loss_decreased": bool(result.history[-1]["loss"] < result.history[0]["loss"]),
        "operator_total_fro": op_total, "collapsed": collapsed,
    }
    run.save_json("tables/decompose_stats.json", stats)

    import pandas as pd

    pd.DataFrame(result.history).to_csv(run.path("tables/altmin_history.csv"), index=False)
    _plot_curves(result.history, run.path("figures/altmin_curves.svg"))

    print(f"\n=== Phase 3 Branch B decomposition — debug ({mcfg.get('name')}) ===")
    print(f"  train R²={train_ev['r2']:+.3f}  test R²={test_ev['r2']:+.3f}  "
          f"(gap {train_ev['r2'] - test_ev['r2']:+.3f})")
    print(f"  L0: SAE-init {sae_l0:.4f} -> train {train_ev['l0']:.4f} / test {test_ev['l0']:.4f}")
    print(f"  loss {result.history[0]['loss']:.3e} -> {result.history[-1]['loss']:.3e} "
          f"({'decreased' if stats['loss_decreased'] else 'NOT decreased'})")
    print(f"  norms: |A|={train_ev['op_A']:.2f} |B|={train_ev['op_B']:.2f} "
          f"|C|={train_ev['op_C']:.2f} |D|={train_ev['op_D']:.2f} |b|={train_ev['b_norm']:.2f}")
    print(f"  collapse-to-bias: {'YES — investigate' if collapsed else 'no'}")
    ok = stats["loss_decreased"] and test_ev["r2"] > 0.0 and not collapsed
    print("optimiser:", "OK" if ok else "SUSPECT — see run.log")
    return 0 if ok else 1


def _plot_curves(history: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    its = [h["iter"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    axes[0].plot(its, [h["loss"] for h in history], "o-"); axes[0].set_yscale("log")
    axes[0].set_title("loss (recon + λ·L1 + λ_reg·F²)"); axes[0].set_xlabel("outer iter")
    axes[1].plot(its, [h["r2"] for h in history], "o-", color="C2")
    axes[1].axhline(0, ls=":", c="grey"); axes[1].set_title("train reconstruction R²")
    axes[1].set_xlabel("outer iter")
    axes[2].plot(its, [h["l0"] for h in history], "o-", color="C3")
    axes[2].set_title("input density (L0 fraction)"); axes[2].set_xlabel("outer iter")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
