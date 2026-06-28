#!/usr/bin/env python3
"""Build + cache the compressed mlsae sources for the 410m primary lane (Phase 3 Step 2).

The 410m-deduped primary model has no EleutherAI residual SAE; this uses the public
per-layer MLSAE suite (``tim-lawson/sae-pythia-410m-deduped-x64-k32-layers-{N}``,
Lawson et al. 2024) via :mod:`layer_dynamics.saes_mlsae`. A dense ``u[N, L, 65536]`` is
~10 GB, so each layer is pruned to its **alive** latents (active on ≥1 example) — a
standard dead-latent removal that Branch B's per-layer width permits — and padded to a
uniform ``M*`` for the SSM. The compressed sources + the slot→global-latent map are
cached to ``data/<model>/sources_mlsae.npz`` so the fit scripts (``step3_decompose``,
``gate3_pareto``, ``step4_sanity``) load them with ``--sources-npz`` instead of
re-downloading 13 GB of SAE weights per fit.

    python scripts/build_410m_sources.py --config configs/pythia-410m-deduped.yaml \
        --desc ph3-sources-410md

The per-layer reconstruction R² (un-standardized) is logged as the loader sanity — a
healthy public TopK SAE reconstructs resid at R² ~ 0.5–0.9 (one weak layer is the SAE's
property, not a loader bug; it is flagged, not hidden).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from layer_dynamics.cli import add_common_args, resolve_out
from layer_dynamics.diagnostics.dataset import load_residual_stream
from layer_dynamics.runlog import init_run
from layer_dynamics.saes_mlsae import REPO_TEMPLATE, build_compressed_sources
from layer_dynamics.saes_qwen import REPO as QWEN_REPO
from layer_dynamics.saes_qwen import build_compressed_sources_qwen
from layer_dynamics.saes_resjb import REPO as RESJB_REPO
from layer_dynamics.saes_resjb import build_compressed_sources_resjb


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--suite", choices=["mlsae", "resjb", "qwen"], default="mlsae",
                        help="SAE suite: 'mlsae' (Pythia TopK resid_post), "
                             "'resjb' (GPT-2-small L1 resid_pre), or "
                             "'qwen' (Qwen2-0.5B L1 resid_post).")
    parser.add_argument("--sae-template", type=str, default=REPO_TEMPLATE,
                        help="mlsae HF repo template; '{layer}' is the block index.")
    parser.add_argument("--sae-repo", type=str, default=None,
                        help="resjb/qwen HF repo id (default: suite's canonical repo).")
    parser.add_argument("--sae-cache", type=Path, default=None)
    parser.add_argument("--max-layers", type=int, default=None,
                        help="Build sources for only the first K source layers (v_0..v_{K-1}); "
                             "use for partial-coverage suites (e.g. Qwen chanind covers blocks "
                             "0..11 of 24). The fit scripts auto-truncate Δ to match.")
    parser.add_argument("--out-npz", type=Path, default=None,
                        help="Where to cache sources (default: data/<model>/sources_<suite>.npz).")
    args = parser.parse_args(argv)

    run = init_run(resolve_out(args), config_path=args.config, seed=args.seed)
    mcfg = run.config["model"]

    h5 = Path("data") / mcfg["name"] / "residual_stream.h5"
    rs, _, _ = load_residual_stream(h5)  # [N, L+1, D]
    if args.max_layers is not None and args.max_layers + 1 < rs.shape[1]:
        rs = rs[:, : args.max_layers + 1]  # keep v_0..v_{K} → K source layers
        run.log(f"--max-layers {args.max_layers}: truncated residual stream to {rs.shape}")
    n, n_states, d = rs.shape
    run.log(f"loaded {h5}: residual_stream {rs.shape}; sources v_0..v_{n_states - 2}")
    default_repo = {"resjb": RESJB_REPO, "qwen": QWEN_REPO}.get(args.suite)
    repo = args.sae_repo or default_repo
    sae_source = repo if args.suite in ("resjb", "qwen") else args.sae_template
    run.log(f"SAE suite: {args.suite} ({sae_source})")

    if args.suite == "resjb":
        cs = build_compressed_sources_resjb(rs, repo=repo,
                                            cache_dir=args.sae_cache, log=run.log)
    elif args.suite == "qwen":
        cs = build_compressed_sources_qwen(rs, repo=repo,
                                           cache_dir=args.sae_cache, log=run.log)
    else:
        cs = build_compressed_sources(rs, template=args.sae_template,
                                      cache_dir=args.sae_cache, log=run.log)
    m_star = cs.u.shape[2]
    init_l0 = float((cs.u != 0).mean())
    matched = ~np.isnan(cs.recon_r2)
    run.log(f"compressed sources u {cs.u.shape}  M*={m_star}  "
            f"({cs.u.nbytes / 1e6:.0f} MB)  init L0={init_l0:.5f}")
    run.log(f"alive per layer: min={int(cs.alive_count[matched].min())} "
            f"max={int(cs.alive_count.max())} mean={cs.alive_count[matched].mean():.0f}")
    run.log(f"matched recon R²: mean={np.nanmean(cs.recon_r2):+.3f} "
            f"min={np.nanmin(cs.recon_r2):+.3f} max={np.nanmax(cs.recon_r2):+.3f}")

    out_npz = args.out_npz or (Path("data") / mcfg["name"] / f"sources_{args.suite}.npz")
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_npz,
        u=cs.u,
        feat_index=cs.feat_index,
        alive_count=cs.alive_count,
        recon_r2=cs.recon_r2,
        n_latents_full=cs.n_latents_full,
        sae_template=sae_source,
    )
    run.log(f"cached sources -> {out_npz}")

    stats = {
        "model": mcfg["name"], "suite": args.suite, "sae_template": sae_source,
        "n_examples": int(n), "n_source_layers": int(n_states - 1), "d_model": int(d),
        "M_star": int(m_star), "n_latents_full": int(cs.n_latents_full),
        "init_l0": init_l0, "out_npz": str(out_npz),
        "alive_count": cs.alive_count.tolist(),
        "recon_r2": [None if np.isnan(x) else float(x) for x in cs.recon_r2],
        "matched_recon_r2_mean": float(np.nanmean(cs.recon_r2)),
        "matched_recon_r2_min": float(np.nanmin(cs.recon_r2)),
    }
    run.save_json("tables/sources_mlsae.json", stats)

    print(f"\n=== Phase 3 410m source build ({mcfg['name']}) ===")
    print(f"compressed u {cs.u.shape}  M*={m_star}  ({cs.u.nbytes / 1e6:.0f} MB)")
    print(f"alive/layer: {cs.alive_count.tolist()}")
    print(f"matched recon R²: mean {stats['matched_recon_r2_mean']:+.3f}, "
          f"min {stats['matched_recon_r2_min']:+.3f}")
    ok = stats["matched_recon_r2_mean"] > 0.4
    print("loader path:", "OK" if ok else "SUSPECT — investigate encode/decode convention")
    print(f"cached -> {out_npz}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
