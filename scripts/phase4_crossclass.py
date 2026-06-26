#!/usr/bin/env python3
"""Phase 4 Step 2 — cross-class generalisation of the decomposition.

Seed stability (0.45, recorded limitation) and dataset stability (Jaccard 0.89) were both
measured at Gate 3. The third Step-2 metric is cross-class generalisation (the plan's
"features trained on geography relations: how well do they reconstruct date-relation
updates?"). With three relation classes (organization / geography / language), we fit the
Branch B SSM on one class's examples and measure held-out Δ reconstruction R² on **each
other class**, against the within-class held-out R² — a full transfer matrix.

A within-class-comparable off-diagonal says the depth dynamics are circuit-*general*
(the same cross-layer propagation reconstructs unseen relation types); a collapse says
they are circuit-*specific*. Either is a legitimate result — this directly answers the
Gate-2 "circuit-general or -specific?" question that Step 4 left blocked on dataset size
(reconstruction transfer needs far fewer examples than a per-class Granger test).

    python scripts/phase4_crossclass.py --config configs/pythia-410m-deduped.yaml \
        --desc ph4-crossclass-410md --sources-npz data/pythia-410m-deduped/sources_mlsae.npz \
        --rank 32 --lambda-l1 1.0 --outer-iters 10 --device cuda

One fit per train class (3 fits), GPU freed between them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from layer_dynamics.cli import add_common_args, resolve_out
from layer_dynamics.decompose import AltMinConfig, alternating_minimize, evaluate, infer_inputs
from layer_dynamics.device import describe, resolve_device
from layer_dynamics.diagnostics.dataset import deltas, load_residual_stream, train_test_split_idx
from layer_dynamics.runlog import init_run
from layer_dynamics.saes_mlsae import load_cached_u
from layer_dynamics.ssm import LowRankSSM, SSMDims


def _free_cuda(*objs) -> None:
    for o in objs:
        del o
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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
    parser.add_argument("--no-learn-z0", action="store_true")
    args = parser.parse_args(argv)

    run = init_run(resolve_out(args), config_path=args.config, seed=args.seed)
    mcfg = run.config.get("model", {})
    device = resolve_device(args.device or mcfg.get("device", "auto"))
    run.log(f"device: {describe(device)}")
    learn_z0 = not args.no_learn_z0

    h5 = Path("data") / mcfg["name"] / "residual_stream.h5"
    rs, rel_class, _ = load_residual_stream(h5)
    delta_all = deltas(rs)
    n, n_layers, d_model = delta_all.shape
    u_all = load_cached_u(args.sources_npz)
    d_input = u_all.shape[2]
    run.log(f"Δ {delta_all.shape}; u {u_all.shape}; classes {sorted(set(rel_class))}")

    # per-class index pools, each split into train/test (held-out within class)
    classes = sorted(set(rel_class.tolist()))
    cls_idx, cls_train, cls_test = {}, {}, {}
    for c in classes:
        idx = np.flatnonzero(rel_class == c)
        tr, te = train_test_split_idx(len(idx), test_frac=args.test_frac, seed=args.seed)
        cls_idx[c] = idx
        cls_train[c] = idx[tr]
        cls_test[c] = idx[te]
        run.log(f"  class {c}: {len(idx)} ({len(tr)} train / {len(te)} test)")

    d_state = args.d_state if args.d_state is not None else args.rank
    dims = SSMDims(n_layers=n_layers, d_model=d_model, d_input=d_input,
                   d_state=d_state, rank=args.rank)
    cfg = AltMinConfig(
        lambda_l1=args.lambda_l1, lambda_reg=args.lambda_reg, outer_iters=args.outer_iters,
        operator_steps=args.operator_steps, operator_lr=args.operator_lr,
        input_steps=args.input_steps, batch_size=args.batch_size,
    )

    # transfer matrix: fit on train class c, frozen-infer + R² on every class's test split
    matrix = {c: {} for c in classes}
    for c in classes:
        run.log(f"=== fit on '{c}' ({len(cls_train[c])} train) ===")
        u_tr = torch.from_numpy(u_all[cls_train[c]])
        d_tr = torch.from_numpy(delta_all[cls_train[c]])
        model = LowRankSSM(dims, learn_z0=learn_z0, seed=args.seed).to(device)
        result = alternating_minimize(model, u_tr, d_tr, cfg, log=run.log)
        for c2 in classes:
            u_te = torch.from_numpy(u_all[cls_test[c2]])
            d_te = torch.from_numpy(delta_all[cls_test[c2]])
            u_fit = infer_inputs(model, u_te, d_te, lambda_l1=cfg.lambda_l1,
                                 n_steps=max(cfg.input_steps * 4, 80),
                                 lipschitz=result.lipschitz, batch_size=cfg.batch_size)
            ev = evaluate(model, u_fit, d_te, lambda_l1=cfg.lambda_l1,
                          lambda_reg=cfg.lambda_reg, batch_size=cfg.batch_size)
            matrix[c][c2] = ev["r2"]
            tag = "within" if c == c2 else "cross "
            run.log(f"  {tag} {c} -> {c2}: test R²={ev['r2']:+.3f} L0={ev['l0']:.4f}")
            _free_cuda(u_te, d_te, u_fit)
        _free_cuda(model, result, u_tr, d_tr)

    within = [matrix[c][c] for c in classes]
    cross = [matrix[c][c2] for c in classes for c2 in classes if c != c2]
    retention = float(np.mean(cross) / np.mean(within)) if np.mean(within) else 0.0
    run.save_json("tables/crossclass.json", {
        "model": mcfg.get("name"), "classes": classes, "matrix": matrix,
        "within_mean": float(np.mean(within)), "cross_mean": float(np.mean(cross)),
        "retention": retention, "dims": vars(dims), "hyperparams": cfg.__dict__,
    })

    print(f"\n=== Phase 4 cross-class transfer ({mcfg.get('name')}) ===")
    print("  rows = train class, cols = eval class (held-out R²)")
    print("  " + " " * 14 + "".join(f"{c[:6]:>9}" for c in classes))
    for c in classes:
        print(f"  {c:>12}  " + "".join(f"{matrix[c][c2]:>+9.3f}" for c2 in classes))
    print(f"  within-class mean R² {np.mean(within):+.3f} | cross-class mean {np.mean(cross):+.3f} "
          f"| retention {retention:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
