#!/usr/bin/env python3
"""R0 — paired bootstrap CIs over test examples for the headline gaps.

Two parts, both reading saved artefacts (review item R0: every headline gap is a point
estimate; the E3 −0.0035 "tie" especially needs a CI before it can be written up):

**A · Gate-3 SSM-vs-baseline gap** (``--checkpoint-dir``). Loads the canonical trained
SSM (``phase4_checkpoint.py`` output: operators + operators-frozen LASSO inputs for all
N + the split), computes per-example test SS for the SSM, then fits the matched
per-layer baseline (``use_state=False``) with the *identical* Gate-3 protocol
(alternating minimisation on train, frozen-operator LASSO inference on test) and
paired-bootstraps the pooled-R² gap over the 311 test examples. The baseline's inputs
are re-inferred through its own LASSO — the same treatment the SSM's inputs got — so
the arms differ only in the cross-layer state term.

**B · Distributed-code curve** (``--distributed-json``). The E1 pooled top-k ablation
curves are stored per candidate (8 seed-stable features × 100 examples, means only), so
the resampling unit is the **candidate** (the natural cluster; per-example drops were
not persisted). Reports candidate-bootstrap CIs for the top-k drop and its paired
difference vs the norm-matched random-direction controls at each k.

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python scripts/r0_bootstrap_ci.py \
        --config configs/pythia-410m-deduped.yaml --desc r0-gate3-ci-mlsae \
        --checkpoint-dir runs/2026-06-26-ph4-checkpoint-410md \
        --sources-npz data/pythia-410m-deduped/sources_mlsae.npz

    uv run python scripts/r0_bootstrap_ci.py --config configs/pythia-410m-deduped.yaml \
        --desc r0-distributed-ci \
        --distributed-json runs/2026-06-29-ph4-e1-distributed-410md/tables/distributed_control.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from layer_dynamics.checkpoint import load_ssm
from layer_dynamics.cli import add_common_args, resolve_out
from layer_dynamics.decompose import AltMinConfig, alternating_minimize, evaluate, infer_inputs
from layer_dynamics.device import describe, resolve_device
from layer_dynamics.diagnostics.bootstrap import paired_gap_ci, per_example_ss, pooled_r2
from layer_dynamics.diagnostics.dataset import deltas, load_residual_stream
from layer_dynamics.runlog import init_run
from layer_dynamics.saes_mlsae import load_cached_u
from layer_dynamics.ssm import LowRankSSM, SSMDims


def _free_cuda(*objs) -> None:
    for o in objs:
        del o
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _predict_per_example(model, u, delta, *, batch_size: int) -> np.ndarray:
    """Model predictions on CPU, minibatched (u/delta stay on CPU)."""
    device = next(model.parameters()).device
    preds = []
    with torch.no_grad():
        for i in range(0, u.shape[0], batch_size):
            sl = slice(i, i + batch_size)
            preds.append(model(u[sl].to(device)).cpu())
    return torch.cat(preds).numpy()


def gate3_ci(args, run) -> dict:
    ckpt_dir = args.checkpoint_dir
    device = resolve_device(args.device or "auto")
    run.log(f"device: {describe(device)}")

    split = np.load(ckpt_dir / "checkpoints" / "split.npz")
    train_idx, test_idx = split["train_idx"], split["test_idx"]
    u_full = torch.from_numpy(np.load(ckpt_dir / "checkpoints" / "u_infer_full.npy"))

    mcfg = run.config["model"]
    h5 = Path("data") / mcfg["name"] / "residual_stream.h5"
    rs, _, _ = load_residual_stream(h5)
    delta = torch.from_numpy(deltas(rs))
    if delta.shape[1] > u_full.shape[1]:
        delta = delta[:, : u_full.shape[1]]
    d_te = delta[test_idx]
    mean_te = d_te.mean(dim=0, keepdim=True).numpy()

    # --- arm A: the canonical SSM (already fit; per-example SS from saved inputs) ---
    ssm = load_ssm(ckpt_dir / "checkpoints" / "ssm_seed0.pt", device=device)
    pred_ssm = _predict_per_example(ssm, u_full[test_idx], d_te, batch_size=args.batch_size)
    ss_ssm, ss_tot = per_example_ss(pred_ssm, d_te.numpy(), mean_te)
    l0_ssm = float((u_full[test_idx] != 0).float().mean())
    run.log(f"SSM (checkpoint): test R²={pooled_r2(ss_ssm, ss_tot):+.4f} L0={l0_ssm:.4f}")
    _free_cuda(ssm, pred_ssm)

    # --- arm B: matched per-layer baseline, identical Gate-3 protocol ---
    dims = SSMDims(n_layers=u_full.shape[1], d_model=delta.shape[2],
                   d_input=u_full.shape[2], d_state=args.rank, rank=args.rank)
    cfg = AltMinConfig(lambda_l1=args.lambda_l1, lambda_reg=args.lambda_reg,
                       outer_iters=args.outer_iters, operator_steps=40, operator_lr=1e-2,
                       input_steps=25, batch_size=args.batch_size)
    u0 = torch.from_numpy(load_cached_u(args.sources_npz))  # SAE init, same as Gate-3
    run.log(f"fitting per-layer baseline (use_state=False, λ_L1={args.lambda_l1:g}, "
            f"outer={args.outer_iters}) — the Gate-3 protocol")
    base = LowRankSSM(dims, learn_z0=not args.no_learn_z0, seed=args.seed,
                      use_state=False).to(device)
    result = alternating_minimize(base, u0[train_idx], delta[train_idx], cfg, log=run.log)
    u_te_fit = infer_inputs(base, u0[test_idx], d_te, lambda_l1=cfg.lambda_l1,
                            n_steps=max(cfg.input_steps * 4, 80),
                            lipschitz=result.lipschitz, batch_size=cfg.batch_size)
    te_ev = evaluate(base, u_te_fit, d_te, lambda_l1=cfg.lambda_l1,
                     lambda_reg=cfg.lambda_reg, batch_size=cfg.batch_size)
    pred_base = _predict_per_example(base, u_te_fit.cpu(), d_te, batch_size=args.batch_size)
    ss_base, _ = per_example_ss(pred_base, d_te.numpy(), mean_te)
    l0_base = float((u_te_fit != 0).float().mean())
    run.log(f"baseline: test R²={te_ev['r2']:+.4f} L0={l0_base:.4f}")
    _free_cuda(base, result, u_te_fit, pred_base)

    ci = paired_gap_ci(ss_ssm, ss_base, ss_tot, n_boot=args.n_boot, seed=args.seed)
    out = {
        "part": "gate3", "checkpoint_dir": str(ckpt_dir),
        "sources": args.sources_npz.name if args.sources_npz else None,
        "lambda_l1": args.lambda_l1, "rank": args.rank, "outer_iters": args.outer_iters,
        "n_test": int(len(test_idx)),
        "ssm_test_r2": pooled_r2(ss_ssm, ss_tot), "ssm_test_l0": l0_ssm,
        "baseline_test_r2": pooled_r2(ss_base, ss_tot), "baseline_test_l0": l0_base,
        "gap_ci": ci,
        "note": ("Paired bootstrap over test examples at the canonical operating point "
                 "(λ_L1 knee). L0 reported for both arms — the gap claim is at "
                 "comparable sparsity, so check l0_base ≈ l0_ssm."),
    }
    np.savez_compressed(run.path("tables/gate3_per_example_ss.npz"),
                        test_idx=test_idx, ss_tot=ss_tot,
                        ss_res_ssm=ss_ssm, ss_res_baseline=ss_base)
    return out


def distributed_ci(args, run) -> dict:
    data = json.loads(Path(args.distributed_json).read_text())
    ks = data["ks"]
    cands = data["per_candidate"]
    n_c = len(cands)
    top = np.array([c["drop_top"] for c in cands])            # [n_c, len(ks)]
    iso = np.array([c["drop_normrand_iso"] for c in cands])
    shp = np.array([c["drop_normrand_shaped"] for c in cands])
    rnd = np.array([c["drop_random"] for c in cands])
    rng = np.random.default_rng(args.seed)
    idx = rng.integers(0, n_c, size=(args.n_boot, n_c))

    def ci_of(mat):  # candidate-cluster bootstrap of the mean curve
        boot = mat[idx].mean(axis=1)                          # [n_boot, len(ks)]
        lo, hi = np.quantile(boot, [0.025, 0.975], axis=0)
        return {"mean": mat.mean(0).tolist(), "ci_lo": lo.tolist(), "ci_hi": hi.tolist()}

    def ci_diff(a, b):  # paired difference (same candidates in both arms)
        boot = (a - b)[idx].mean(axis=1)
        lo, hi = np.quantile(boot, [0.025, 0.975], axis=0)
        return {"mean": (a - b).mean(0).tolist(), "ci_lo": lo.tolist(), "ci_hi": hi.tolist()}

    out = {
        "part": "distributed", "source_json": str(args.distributed_json),
        "ks": ks, "n_candidates": n_c, "n_boot": args.n_boot,
        "resampling_unit": "candidate (per-example drops were not persisted; examples "
                           "within a candidate share the feature, so the candidate is "
                           "the honest cluster)",
        "top": ci_of(top), "normrand_iso": ci_of(iso), "normrand_shaped": ci_of(shp),
        "count_matched_random": ci_of(rnd),
        "top_minus_iso": ci_diff(top, iso), "top_minus_shaped": ci_diff(top, shp),
        "top_minus_count_random": ci_diff(top, rnd),
    }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--checkpoint-dir", type=Path, default=None,
                        help="phase4_checkpoint run dir (part A: Gate-3 gap CI)")
    parser.add_argument("--sources-npz", type=Path, default=None,
                        help="SAE-init sources for the baseline refit (part A)")
    parser.add_argument("--distributed-json", type=Path, default=None,
                        help="distributed_control.json (part B: candidate-level CIs)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--lambda-l1", type=float, default=1.0)
    parser.add_argument("--lambda-reg", type=float, default=1e-4)
    parser.add_argument("--outer-iters", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--no-learn-z0", action="store_true")
    parser.add_argument("--n-boot", type=int, default=10_000)
    args = parser.parse_args(argv)
    if args.checkpoint_dir is None and args.distributed_json is None:
        raise SystemExit("need --checkpoint-dir (part A) and/or --distributed-json (part B)")
    if args.checkpoint_dir is not None and args.sources_npz is None:
        raise SystemExit("part A needs --sources-npz (baseline refit init)")

    run = init_run(resolve_out(args), config_path=args.config, seed=args.seed)
    results = {}
    if args.checkpoint_dir is not None:
        results["gate3"] = gate3_ci(args, run)
    if args.distributed_json is not None:
        results["distributed"] = distributed_ci(args, run)
    run.save_json("tables/r0_bootstrap_ci.json", results)

    print("\n=== R0 paired bootstrap CIs ===")
    if "gate3" in results:
        g = results["gate3"]
        ci = g["gap_ci"]
        print(f"Gate-3 gap (SSM − baseline) @ λ_L1={g['lambda_l1']:g}: "
              f"{ci['gap']:+.4f}  95% [{ci['ci_lo']:+.4f}, {ci['ci_hi']:+.4f}]"
              f"{'  *significant*' if ci['significant'] else ''}")
        print(f"  SSM R²={g['ssm_test_r2']:+.4f} (L0 {g['ssm_test_l0']:.4f})  "
              f"baseline R²={g['baseline_test_r2']:+.4f} (L0 {g['baseline_test_l0']:.4f})")
    if "distributed" in results:
        d = results["distributed"]
        for k_i, k in enumerate(d["ks"]):
            t = d["top"]
            df = d["top_minus_iso"]
            print(f"top-k drop @k={k}: {t['mean'][k_i]:+.3f} "
                  f"[{t['ci_lo'][k_i]:+.3f}, {t['ci_hi'][k_i]:+.3f}]  "
                  f"top−iso {df['mean'][k_i]:+.3f} [{df['ci_lo'][k_i]:+.3f}, "
                  f"{df['ci_hi'][k_i]:+.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
