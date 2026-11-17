#!/usr/bin/env python3
"""v1.1 check A, part 1 — codes inferred from the layers below ℓ only.

The recorded state probe reads codes fit to each example's whole update trajectory, so
the state at depth ℓ can carry information about ``Δ_ℓ`` and later updates (paper §6.7).
For every depth ℓ = 1 … L-1 this script re-solves the recorded held-out encoder
(operators frozen, same λ, same initial SAE codes, same step count, same Lipschitz
hand-over) against ``Δ_0 … Δ_{ℓ-1}`` only, and keeps three depth-causal summaries of the
result, each of the state's dimension:

  * ``z_causal[:, ℓ]``      — the state entering observation ℓ built from those codes;
  * ``prefix_pca[:, ℓ]``    — train-fit PCA of the same prefix codes ``u_0 … u_{ℓ-1}``
                              (the variance-optimal summary of exactly the state's inputs);
  * ``feed_prev[:, ℓ]``     — the feedthrough's rank-r image of the last past code
                              ``u_{ℓ-1}`` (a per-layer, reconstruction-trained summary).

``scripts/v11_probe_causal.py`` probes them next to the recorded arms.

Hard stop: at full length the truncated encoder must reproduce the checkpoint's
``u_infer_full``. If it does not, the difference between the causal and the recorded
arms would not be the truncation alone, and nothing is written.

Resumable: each depth is cached under ``checkpoints/`` and skipped if present.

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python \\
        scripts/v11_causal_codes.py --config configs/pythia-410m-deduped.yaml \\
        --out runs/<date>-v11c-causal-codes-mlsae-410md \\
        --checkpoint-dir runs/<date>-v11c-checkpoint-mlsae-410md \\
        --sources-npz data/pythia-410m-deduped/sources_mlsae.npz
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from layer_dynamics.checkpoint import load_ssm
from layer_dynamics.cli import add_common_args, resolve_out
from layer_dynamics.device import describe, resolve_device
from layer_dynamics.diagnostics.causal_codes import (
    PrefixSSM,
    feedthrough_image,
    infer_prefix_inputs,
    prefix_state,
)
from layer_dynamics.diagnostics.dataset import deltas, load_residual_stream
from layer_dynamics.runlog import init_run
from layer_dynamics.saes_mlsae import load_cached_u

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase4_state_probe import _pca_fit  # noqa: E402  (the recorded probe's PCA, verbatim)


def _prefix_r2(model, u_p: torch.Tensor, delta_p: torch.Tensor, *, batch_size: int) -> float:
    """Pooled R² of the prefix reconstruction vs the per-depth mean of ``delta_p``."""
    device = next(model.parameters()).device
    fwd = PrefixSSM(model, u_p.shape[1])
    mean = delta_p.mean(dim=0, keepdim=True)
    ss_res = ss_tot = 0.0
    with torch.no_grad():
        for i in range(0, u_p.shape[0], batch_size):
            db = delta_p[i:i + batch_size].to(device)
            ss_res += ((db - fwd(u_p[i:i + batch_size].to(device))) ** 2).sum().item()
            ss_tot += ((db - mean.to(device)) ** 2).sum().item()
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--checkpoint-dir", type=Path, required=True,
                        help="phase4_checkpoint run dir (operators, u_infer_full, split, "
                             "infer_meta.json).")
    parser.add_argument("--sources-npz", type=Path, required=True,
                        help="The SAE codes the checkpoint was fit over (the encoder's "
                             "warm start).")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--depths", type=int, nargs="+", default=None,
                        help="Depths ℓ to compute (default: all of 1 … L-1).")
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Smoke only: use the first N examples (whole batches of the "
                             "recorded order, so the full-length check still applies).")
    parser.add_argument("--tol", type=float, default=1e-5,
                        help="Max abs difference allowed between the full-length truncated "
                             "encoder and the checkpoint's u_infer_full.")
    args = parser.parse_args(argv)

    run = init_run(resolve_out(args), config_path=args.config, seed=args.seed)
    mcfg = run.config["model"]
    device = resolve_device(args.device or mcfg.get("device", "auto"))
    run.log(f"device: {describe(device)}")

    ckpt = args.checkpoint_dir / "checkpoints"
    meta = json.loads((ckpt / "infer_meta.json").read_text(encoding="utf-8"))
    lam, n_steps = float(meta["lambda_l1"]), int(meta["n_steps"])
    lip0, bs = float(meta["lipschitz"]), int(meta["batch_size"])
    model = load_ssm(ckpt / "ssm_seed0.pt", device=device)
    split = np.load(ckpt / "split.npz")
    train_idx, test_idx = split["train_idx"], split["test_idx"]
    u_rec = np.load(ckpt / "u_infer_full.npy", mmap_mode="r")
    rs, _, _ = load_residual_stream(Path("data") / mcfg["name"] / "residual_stream.h5")
    delta_all = torch.from_numpy(deltas(rs))
    u_init = torch.from_numpy(load_cached_u(args.sources_npz))
    n, n_layers, d_input = u_init.shape
    if u_rec.shape != (n, n_layers, d_input):
        raise SystemExit(f"sources {tuple(u_init.shape)} do not match the checkpoint's codes "
                         f"{u_rec.shape}")
    if args.max_examples is not None:
        n = min(n, args.max_examples)
        u_init, delta_all = u_init[:n], delta_all[:n]
        train_idx, test_idx = train_idx[train_idx < n], test_idx[test_idx < n]
    d_z, rank = model.dims.d_state, model.dims.rank
    run.log(f"codes {tuple(u_init.shape)}, λ_L1={lam:g}, {n_steps} FISTA steps, "
            f"Lipschitz hand-over {lip0:g}, batch {bs}; split {len(train_idx)}/{len(test_idx)}")

    cache = Path(run.path("checkpoints"))
    cache.mkdir(parents=True, exist_ok=True)
    kw = {"lambda_l1": lam, "n_steps": n_steps, "lipschitz": lip0, "batch_size": bs}

    # --- hard stop: full-length truncated encoder == the recorded encoder --------------
    gate_path = cache / "full_length_check.json"
    if gate_path.exists():
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        run.log(f"full-length check already done: max |Δu| = {gate['max_abs_diff']:.3e}")
    else:
        t0 = time.time()
        u_full = infer_prefix_inputs(model, u_init, delta_all, n_layers, **kw).numpy()
        ref = np.asarray(u_rec[:n])
        gate = {
            "max_abs_diff": float(np.abs(u_full - ref).max()),
            "nonzero_pattern_identical": bool(((u_full != 0) == (ref != 0)).all()),
            "n_examples": int(n), "seconds": round(time.time() - t0, 1), "tol": args.tol,
        }
        gate["ok"] = bool(gate["max_abs_diff"] <= args.tol)
        gate_path.write_text(json.dumps(gate, indent=2), encoding="utf-8")
        run.log(f"full-length check: max |Δu| = {gate['max_abs_diff']:.3e}, nonzero pattern "
                f"identical: {gate['nonzero_pattern_identical']} ({gate['seconds']} s)")
        del u_full, ref
    if not gate["ok"]:
        run.save_json("tables/causal_codes.json", {"full_length_check": gate, "ok": False})
        run.log("ABORT: the truncated encoder does not reproduce u_infer_full at full length")
        return 1

    # --- per-depth causal summaries ----------------------------------------------------
    depths = sorted(set(args.depths)) if args.depths else list(range(1, n_layers))
    bad = [ell for ell in depths if not 1 <= ell <= n_layers - 1]
    if bad:
        raise SystemExit(f"depths {bad} outside 1..{n_layers - 1}")
    rows = []
    for ell in depths:
        f = cache / f"causal_depth_{ell:02d}.npz"
        if f.exists():
            with np.load(f) as zf:
                rows.append(json.loads(str(zf["diag"])))
            run.log(f"  ℓ={ell:2d}: cached")
            continue
        t0 = time.time()
        u_p = infer_prefix_inputs(model, u_init, delta_all, ell, **kw)       # [n, ℓ, M] cpu
        with torch.no_grad():
            parts_z, parts_f = [], []
            for i in range(0, n, bs):
                ub = u_p[i:i + bs].to(device)
                parts_z.append(prefix_state(model, ub).cpu())
                parts_f.append(feedthrough_image(model, ub[:, ell - 1], ell - 1).cpu())
            z = torch.cat(parts_z).numpy()
            feed = torch.cat(parts_f).numpy()
        x = u_p.numpy().reshape(n, -1)
        mean, comp = _pca_fit(x[train_idx].astype(np.float32, copy=False), d_z)
        pca = ((x - mean) @ comp).astype(np.float64)
        diag = {
            "depth": ell,
            "test_prefix_r2": _prefix_r2(model, u_p[test_idx], delta_all[test_idx, :ell],
                                         batch_size=bs),
            "l0": float((u_p != 0).float().mean()),
            "seconds": round(time.time() - t0, 1),
        }
        tmp = f.with_name(f.name + ".tmp.npz")
        np.savez(tmp, z=z, feed=feed, pca=pca, diag=json.dumps(diag))
        tmp.replace(f)
        rows.append(diag)
        run.log(f"  ℓ={ell:2d}: prefix test R²={diag['test_prefix_r2']:+.3f} "
                f"L0={diag['l0']:.4f} ({diag['seconds']} s)")
        del u_p, x, pca

    # --- assemble (zeros at ℓ = 0 and at any depth not computed) -----------------------
    z_causal = np.zeros((n, n_layers, d_z), dtype=np.float64)
    feed_prev = np.zeros((n, n_layers, rank), dtype=np.float64)
    prefix_pca = np.zeros((n, n_layers, d_z), dtype=np.float64)
    with torch.no_grad():
        z_causal[:, 0] = model._z0(n).cpu().numpy()
    for ell in depths:
        with np.load(cache / f"causal_depth_{ell:02d}.npz") as zf:
            z_causal[:, ell], feed_prev[:, ell], prefix_pca[:, ell] = zf["z"], zf["feed"], zf["pca"]
    np.savez_compressed(cache / "features_causal.npz", z_causal=z_causal, feed_prev=feed_prev,
                        prefix_pca=prefix_pca, depths=np.array(depths), n_examples=n)
    run.save_json("tables/causal_codes.json", {
        "ok": True, "checkpoint_dir": str(args.checkpoint_dir),
        "sources_npz": str(args.sources_npz), "lambda_l1": lam, "n_steps": n_steps,
        "lipschitz": lip0, "batch_size": bs, "n_examples": int(n), "depths": depths,
        "full_length_check": gate, "per_depth": rows,
        "note": ("For each depth ℓ the recorded held-out encoder is re-solved against "
                 "Δ_0..Δ_{ℓ-1} only (operators frozen). test_prefix_r2 is the pooled R² of "
                 "that prefix reconstruction on the test split; l0 the fraction of nonzero "
                 "codes over all examples and prefix layers."),
    })
    print(f"\n=== v1.1 causal codes ({mcfg.get('name')}) ===")
    print(f"  full-length check: max |Δu| = {gate['max_abs_diff']:.3e} (tol {args.tol:g})")
    print(f"  {len(depths)} depth(s) written to {cache / 'features_causal.npz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
