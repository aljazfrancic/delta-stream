#!/usr/bin/env python3
"""Verify the per-layer SAE source path on collected activations (Phase 3 Step 2).

Loads each per-layer SAE from the suite, encodes the matched residual state
``v_{ℓ'}`` (TopK), decodes, and reports reconstruction R² and the active-latent
fraction. This is the milestone-1 sanity for Branch B sources: it confirms the
minimal loader (no ``sparsify`` dependency) reproduces the published SAE quality
and pins the sparse-source statistics the optimiser will consume.

    python scripts/verify_saes.py --config configs/pythia-70m.yaml --desc ph3-saes-70md \
        --sae-repo EleutherAI/sae-pythia-70m-deduped-32k

Matched-state R² is reported per source layer ``ℓ' = 1 … L-1`` (``layers.{ℓ'-1}``);
``v_0`` uses a flagged proxy SAE (no embedding SAE exists in the suite) and is
reported separately so its lower quality is not mistaken for a loader bug.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from layer_dynamics.cli import add_common_args, resolve_out
from layer_dynamics.diagnostics.dataset import load_residual_stream
from layer_dynamics.runlog import init_run
from layer_dynamics.saes import (
    DEFAULT_SAE_REPO,
    encode,
    load_sae,
    reconstruction_r2,
    source_hookpoint,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--sae-repo", type=str, default=DEFAULT_SAE_REPO,
                        help="HF repo of the per-layer SAE suite.")
    parser.add_argument("--sae-cache", type=Path, default=None,
                        help="Optional cache dir for SAE downloads (default: HF cache).")
    args = parser.parse_args(argv)

    run = init_run(resolve_out(args), config_path=args.config, seed=args.seed)
    mcfg = run.config["model"]

    h5 = Path("data") / mcfg["name"] / "residual_stream.h5"
    rs, _, _ = load_residual_stream(h5)  # [N, L+1, D]
    n, n_states, d = rs.shape
    n_layers = n_states - 1
    run.log(f"loaded {h5}: residual_stream {rs.shape}; sources v_0..v_{n_layers - 1}")
    run.log(f"SAE suite: {args.sae_repo}")

    rows = []
    for ellp in range(n_layers):  # source states v_0 .. v_{L-1}
        hookpoint, matched = source_hookpoint(ellp, n_layers)
        sae = load_sae(args.sae_repo, hookpoint, cache_dir=args.sae_cache)
        v = rs[:, ellp, :]  # [N, D]
        z = encode(sae, v)
        r2 = reconstruction_r2(sae, v)
        active = float((z != 0).sum(axis=1).mean())
        rows.append({
            "source_layer": ellp, "state": f"v_{ellp}", "hookpoint": hookpoint,
            "matched": matched, "recon_r2": r2, "n_latents": sae.n_latents,
            "k_cfg": sae.k, "active_mean": active,
        })
        flag = "" if matched else "  (PROXY — no embedding SAE)"
        run.log(f"  v_{ellp:<2d} <- {hookpoint:<10s} R²={r2:+.3f} active={active:.1f}/"
                f"{sae.n_latents}{flag}")

    matched_r2 = [r["recon_r2"] for r in rows if r["matched"]]
    stats = {
        "model": mcfg["name"], "sae_repo": args.sae_repo, "n_examples": n,
        "n_source_layers": n_layers, "d_model": d,
        "matched_recon_r2_mean": float(np.mean(matched_r2)),
        "matched_recon_r2_min": float(np.min(matched_r2)),
        "proxy_v0_recon_r2": next(r["recon_r2"] for r in rows if not r["matched"]),
        "per_layer": rows,
    }
    run.save_json("tables/sae_verify.json", stats)

    import pandas as pd

    pd.DataFrame(rows).to_csv(run.path("tables/sae_verify.csv"), index=False)

    print(f"\n=== Phase 3 SAE source verification ({mcfg['name']}) ===")
    print(f"matched-state reconstruction R²: mean {stats['matched_recon_r2_mean']:.3f}, "
          f"min {stats['matched_recon_r2_min']:.3f} (over v_1..v_{n_layers - 1})")
    print(f"v_0 proxy R²: {stats['proxy_v0_recon_r2']:.3f} (expected lower — no embedding SAE)")
    # A healthy public TopK-SAE suite reconstructs resid_post at R² ~ 0.6-0.9.
    ok = stats["matched_recon_r2_mean"] > 0.4
    print("loader path:", "OK" if ok else "SUSPECT — investigate encode/decode convention")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
