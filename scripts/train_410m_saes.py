#!/usr/bin/env python3
"""Train matched per-layer L1 / TopK residual SAEs on the cached 410m activations (E2).

Closes the missing **within-Pythia-410m** L1-vs-TopK seed-stability control (``progress.md``
Fix 6 / E2): trains, on the *same* cached query-position residuals, two SAE dictionaries
that differ only in the sparsity mechanism, then caches each as a drop-in
``sources_trained_{mode}.npz`` consumed by ``gate3_pareto`` / ``step4_sanity`` /
``phase4_checkpoint`` via ``--sources-npz`` (identical mlsae ``CompressedSources`` format,
same ``v_0`` zero-init — so the only thing that changes vs the recorded Gate-3 run is the
dictionary).

    # TopK (matches mlsae k=32; isolates trained-here vs pretrained)
    python scripts/train_410m_saes.py --config configs/pythia-410m-deduped.yaml \
        --mode topk --k 32 --desc ph4-e2-train-topk-410md
    # L1 (the seed-stable hypothesis; tune --l1-coeff to match the TopK L0)
    python scripts/train_410m_saes.py --config configs/pythia-410m-deduped.yaml \
        --mode l1 --l1-coeff 3e-2 --desc ph4-e2-train-l1-410md

The build-step recon-R² gate (mean > 0.4, the same on-data smoke-test the mlsae/qwen
builders use) prints OK/SUSPECT and sets the exit code. SAE weights are saved next to the
npz so the dictionaries can be reloaded for interpretation / the causal re-run.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from layer_dynamics.cli import add_common_args, resolve_out
from layer_dynamics.device import resolve_device
from layer_dynamics.diagnostics.dataset import load_residual_stream
from layer_dynamics.runlog import init_run
from layer_dynamics.sae_train import build_compressed_sources_trained


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--mode", choices=["topk", "l1"], required=True)
    parser.add_argument("--n-latents", type=int, default=8192,
                        help="dictionary width M (×8 expansion of D=1024 by default).")
    parser.add_argument("--k", type=int, default=32, help="TopK width (matches mlsae k=32).")
    parser.add_argument("--l1-coeff", type=float, default=3e-2, help="L1 penalty (l1 mode).")
    parser.add_argument("--no-standardize", action="store_true",
                        help="disable per-token standardization (mlsae uses it; default on).")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-layers", type=int, default=None,
                        help="train only the first K source layers (smoke test).")
    parser.add_argument("--out-npz", type=Path, default=None,
                        help="default: data/<model>/sources_trained_<mode>.npz")
    args = parser.parse_args(argv)

    run = init_run(resolve_out(args), config_path=args.config, seed=args.seed)
    mcfg = run.config["model"]
    device = args.device or resolve_device(mcfg.get("device", "auto"))
    run.log(f"device={device}  mode={args.mode}  M={args.n_latents}  "
            f"k={args.k}  l1={args.l1_coeff}  standardize={not args.no_standardize}")

    h5 = Path("data") / mcfg["name"] / "residual_stream.h5"
    rs, _, _ = load_residual_stream(h5)  # [N, L+1, D]
    if args.max_layers is not None and args.max_layers + 1 < rs.shape[1]:
        rs = rs[:, : args.max_layers + 1]
        run.log(f"--max-layers {args.max_layers}: truncated residual stream to {rs.shape}")
    n, n_states, d = rs.shape
    run.log(f"loaded {h5}: residual_stream {rs.shape}; training sources v_1..v_{n_states - 2}")

    cs, state_dicts = build_compressed_sources_trained(
        rs, mode=args.mode, n_latents=args.n_latents, k=args.k, l1_coeff=args.l1_coeff,
        standardize=not args.no_standardize, epochs=args.epochs, lr=args.lr,
        seed=args.seed, device=device, log=run.log,
    )
    m_star = cs.u.shape[2]
    init_l0 = float((cs.u != 0).mean())
    matched = ~np.isnan(cs.recon_r2)
    run.log(f"compressed sources u {cs.u.shape}  M*={m_star}  "
            f"({cs.u.nbytes / 1e6:.0f} MB)  init L0={init_l0:.5f}")
    run.log(f"alive per layer: min={int(cs.alive_count[matched].min())} "
            f"max={int(cs.alive_count.max())} mean={cs.alive_count[matched].mean():.0f}")
    run.log(f"matched recon R²: mean={np.nanmean(cs.recon_r2):+.3f} "
            f"min={np.nanmin(cs.recon_r2):+.3f} max={np.nanmax(cs.recon_r2):+.3f}")

    out_npz = args.out_npz or (Path("data") / mcfg["name"] / f"sources_trained_{args.mode}.npz")
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, u=cs.u, feat_index=cs.feat_index, alive_count=cs.alive_count,
             recon_r2=cs.recon_r2, n_latents_full=cs.n_latents_full,
             sae_template=f"trained-{args.mode}")
    run.log(f"cached sources -> {out_npz}")
    weights_path = out_npz.with_suffix(".weights.pkl")
    with open(weights_path, "wb") as f:
        pickle.dump({"mode": args.mode, "k": args.k, "n_latents": args.n_latents,
                     "standardize": not args.no_standardize, "state_dicts": state_dicts}, f)
    run.log(f"saved SAE weights -> {weights_path}")

    stats = {
        "model": mcfg["name"], "mode": args.mode, "n_latents": int(args.n_latents),
        "k": int(args.k), "l1_coeff": float(args.l1_coeff),
        "standardize": not args.no_standardize, "epochs": int(args.epochs),
        "n_examples": int(n), "n_source_layers": int(n_states - 1), "d_model": int(d),
        "M_star": int(m_star), "init_l0": init_l0, "out_npz": str(out_npz),
        "alive_count": cs.alive_count.tolist(),
        "recon_r2": [None if np.isnan(x) else float(x) for x in cs.recon_r2],
        "matched_recon_r2_mean": float(np.nanmean(cs.recon_r2)),
        "matched_recon_r2_min": float(np.nanmin(cs.recon_r2)),
    }
    run.save_json(f"tables/sources_trained_{args.mode}.json", stats)

    print(f"\n=== Phase 4 E2 SAE training ({mcfg['name']}, {args.mode}) ===")
    print(f"compressed u {cs.u.shape}  M*={m_star}  init L0={init_l0:.5f}")
    print(f"matched recon R²: mean {stats['matched_recon_r2_mean']:+.3f}, "
          f"min {stats['matched_recon_r2_min']:+.3f}")
    ok = stats["matched_recon_r2_mean"] > 0.4
    print("loader path:", "OK" if ok else "SUSPECT — recon below gate; tune epochs/M/l1")
    print(f"cached -> {out_npz}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
