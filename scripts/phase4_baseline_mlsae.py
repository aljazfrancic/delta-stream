#!/usr/bin/env python3
"""Phase 4 Step 1 — Baseline B: the per-layer SAE's *implied* Δ reconstruction.

The headline reconstruction claim (Gate 3) compares the Branch B SSM against a per-layer
sparse dictionary fit on Δ with the **same** optimiser (``use_state=False``). That is the
strict, apples-to-apples per-layer baseline. This script adds the complementary,
*pretrained-SAE* baseline the plan's Step 1 calls for: take the public MLSAE suite — the
same dictionary the SSM sources from — reconstruct each residual state ``v̂_ℓ`` with its
own layer's SAE, and difference adjacent reconstructions to predict the update,

    Δ̂_ℓ = v̂_{ℓ+1} − v̂_ℓ ,   v̂_{ℓ'} = SAE_{ℓ'-1}.decode(encode(v_{ℓ'})) ,  v̂_0 = v_0.

This is an *indirect* baseline — the SAE was trained to reconstruct ``v_ℓ``, not ``Δ_ℓ`` —
and the bias runs **against** it: ``Δ̂_ℓ = Δ_ℓ − (ε_{ℓ+1} − ε_ℓ)`` adds the two per-layer
reconstruction residuals ``ε_ℓ = v_ℓ − v̂_ℓ`` onto the scored quantity. We therefore report
the raw ``v_ℓ`` reconstruction R² alongside the implied-Δ R², and are explicit about the
direction of the bias (per the plan). With no public Pythia-410m CLT or crosscoder, this
MLSAE baseline is also the closest feasible proxy for the CLT comparison (Baseline D),
which is recorded as a compute limitation.

R² is vs the per-depth mean of Δ (same intercept-only baseline as the SSM's ``evaluate``),
on the full dataset and on the seed-0 held-out test split, so the test number is directly
comparable to the SSM's held-out R².

    python scripts/phase4_baseline_mlsae.py --config configs/pythia-410m-deduped.yaml \
        --desc ph4-baseline-mlsae-410md

Pure NumPy on the cached SAEs (no GPU) — safe to run alongside a GPU fit.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from layer_dynamics.cli import add_common_args, resolve_out
from layer_dynamics.diagnostics.dataset import deltas, load_residual_stream, train_test_split_idx
from layer_dynamics.runlog import init_run
from layer_dynamics.saes_mlsae import REPO_TEMPLATE, load_mlsae, reconstruct


def _r2_per_layer(delta: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Per-layer R² of ``pred`` vs ``delta`` (intercept = per-depth mean), → ``[L]``."""
    mean = delta.mean(axis=0, keepdims=True)
    ss_res = ((delta - pred) ** 2).sum(axis=(0, 2))
    ss_tot = ((delta - mean) ** 2).sum(axis=(0, 2))
    return np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, 0.0)


def _r2_pooled(delta: np.ndarray, pred: np.ndarray) -> float:
    """Pooled R² over examples × depths (matches the SSM ``evaluate`` convention)."""
    mean = delta.mean(axis=0, keepdims=True)
    ss_res = float(((delta - pred) ** 2).sum())
    ss_tot = float(((delta - mean) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_args(parser)
    parser.add_argument("--template", type=str, default=REPO_TEMPLATE)
    parser.add_argument("--sae-cache", type=Path, default=None)
    parser.add_argument("--test-frac", type=float, default=0.2)
    args = parser.parse_args(argv)

    run = init_run(resolve_out(args), config_path=args.config, seed=args.seed)
    mcfg = run.config.get("model", {})

    h5 = Path("data") / mcfg["name"] / "residual_stream.h5"
    rs, _, _ = load_residual_stream(h5)               # [N, L+1, D]
    delta = deltas(rs)                                 # [N, L, D]
    n, n_states, d_model = rs.shape
    n_layers = n_states - 1
    run.log(f"loaded {h5}: v {rs.shape}, Δ {delta.shape}")

    # --- reconstruct each residual state with its layer's SAE ---
    # SAE ``layers-N`` = output of block N = v_{N+1}, so state v_{ℓ'} (ℓ'=1..L) uses
    # ``layers-{ℓ'-1}``. v_0 (embedding) has no SAE → v̂_0 = v_0. (Unlike the *source*
    # builder, which stops at v_{L-1}, the implied-Δ needs v̂_L too — for Δ̂_{L-1}.)
    v_hat = rs.copy()                                  # v̂_0 = v_0 (no embedding SAE)
    v_recon_r2 = np.full(n_layers + 1, np.nan)         # raw v_ℓ recon R² (NaN for v_0)
    for ellp in range(1, n_layers + 1):
        sae = load_mlsae(ellp - 1, template=args.template, cache_dir=args.sae_cache)
        x = rs[:, ellp, :].astype(np.float32)
        rec = reconstruct(sae, x)
        v_hat[:, ellp, :] = rec
        ss_res = float(np.sum((x - rec) ** 2))
        ss_tot = float(np.sum((x - x.mean(axis=0)) ** 2))
        v_recon_r2[ellp] = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        run.log(f"  v_{ellp:<2d} <- layers-{ellp - 1:<2d}  recon R²={v_recon_r2[ellp]:+.3f}")

    delta_hat = v_hat[:, 1:, :] - v_hat[:, :-1, :]     # [N, L, D] implied Δ̂

    # --- R² of the implied Δ, full + held-out test split ---
    train_idx, test_idx = train_test_split_idx(n, test_frac=args.test_frac, seed=args.seed)
    r2_layer_full = _r2_per_layer(delta, delta_hat)
    r2_full = _r2_pooled(delta, delta_hat)
    r2_test = _r2_pooled(delta[test_idx], delta_hat[test_idx])
    r2_layer_test = _r2_per_layer(delta[test_idx], delta_hat[test_idx])

    run.log(f"implied-Δ R²: pooled full {r2_full:+.3f}  |  held-out test {r2_test:+.3f}")
    run.log(f"raw v_ℓ recon R²: mean {np.nanmean(v_recon_r2):+.3f} "
            f"(min {np.nanmin(v_recon_r2):+.3f} @v_{int(np.nanargmin(v_recon_r2))})")

    stats = {
        "model": mcfg.get("name"), "baseline": "mlsae_implied_delta",
        "template": args.template, "n": n, "n_layers": n_layers,
        "delta_r2_pooled_full": r2_full, "delta_r2_pooled_test": r2_test,
        "delta_r2_per_layer_full": r2_layer_full.tolist(),
        "delta_r2_per_layer_test": r2_layer_test.tolist(),
        "v_recon_r2_per_state": v_recon_r2.tolist(),
        "v_recon_r2_mean": float(np.nanmean(v_recon_r2)),
        "note": ("indirect baseline: SAE trained for v_ℓ not Δ_ℓ; bias is against the SAE "
                 "(Δ̂_ℓ = Δ_ℓ − (ε_{ℓ+1} − ε_ℓ)). Closest feasible proxy for CLT (Baseline D)."),
        "n_train": len(train_idx), "n_test": len(test_idx),
    }
    run.save_json("tables/baseline_mlsae.json", stats)

    print(f"\n=== Phase 4 Baseline B — MLSAE implied Δ ({mcfg.get('name')}) ===")
    print(f"  implied-Δ R² (held-out test): {r2_test:+.3f}  [SSM @λ=1 ≈ +0.857]")
    print(f"  implied-Δ R² (pooled full):   {r2_full:+.3f}")
    print(f"  raw v_ℓ recon R² (mean):      {np.nanmean(v_recon_r2):+.3f}")
    print("  per-layer implied-Δ R² (test):")
    for ell in range(n_layers):
        print(f"    Δ_{ell:<2d}: {r2_layer_test[ell]:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
