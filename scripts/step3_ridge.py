#!/usr/bin/env python3
"""Linear depth-local predictability — ridge low-rank fit (Phase 1 Step 3, part A).

For every layer pair (ℓ, ℓ+1), fit Δ_ℓ ≈ v_ℓ Wᵀ with rank-constrained W and
report held-out pooled R² at each rank, against the trivial baselines:
zero-prediction (R² generally < 0), mean-prediction (R² = 0 by construction),
and the identity transparency baseline (v̂_{ℓ+1} = v_ℓ, R² ≈ 1). The Tuned-Lens
gate baseline is added separately (part B).

    python scripts/step3_ridge.py --config configs/pythia-410m-deduped.yaml --desc ph1-step3-410m

Outputs: tables/ridge_r2.csv, figures/r2_heatmap.svg, figures/r2_by_layer.svg,
tables/step3_stats.json (incl. layer-averaged best-rank ridge R² — the quantity
the Step 5 gate compares against the Tuned Lens).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from layer_dynamics.cli import add_common_args, resolve_out
from layer_dynamics.diagnostics.dataset import deltas, load_residual_stream, train_test_split_idx
from layer_dynamics.diagnostics.lowrank_ridge import fit_lowrank_ridge, identity_r2, pooled_r2
from layer_dynamics.runlog import init_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args(argv)

    run = init_run(resolve_out(args), config_path=args.config, seed=args.seed)
    cfg = run.config
    mcfg = cfg["model"]
    dgcfg = cfg.get("diagnostics", {})
    ranks = tuple(dgcfg.get("ranks", [16, 32, 64, 128]))
    test_frac = float(dgcfg.get("test_frac", 0.2))

    h5 = Path("data") / mcfg["name"] / "residual_stream.h5"
    rs, _, _ = load_residual_stream(h5)
    n, n_states, d = rs.shape
    n_layers = n_states - 1
    dlt = deltas(rs)  # [N, L, D]
    run.log(f"loaded {h5}: residual_stream {rs.shape}, deltas {dlt.shape}")

    tr, te = train_test_split_idx(n, test_frac=test_frac, seed=args.seed)
    run.log(f"split: {len(tr)} train / {len(te)} test")

    rows = []
    for ell in range(n_layers):
        x_tr, x_te = rs[tr, ell], rs[te, ell]            # v_ℓ
        y_tr, y_te = dlt[tr, ell], dlt[te, ell]          # Δ_ℓ
        res = fit_lowrank_ridge(x_tr, y_tr, x_te, y_te, ranks=ranks)

        ref = np.broadcast_to(y_tr.mean(axis=0), y_te.shape)
        zero_r2 = pooled_r2(y_te, np.zeros_like(y_te), ref)
        ident = identity_r2(x_tr, rs[tr, ell + 1], x_te, rs[te, ell + 1])

        best_rank = max(ranks, key=lambda r: res[r])
        rows.append({
            "layer": ell,
            **{f"r{r}": res[r] for r in ranks},
            "full": res["full"],
            "zero": zero_r2,
            "identity": ident,
            "alpha": res["alpha"],
            "best_rank": best_rank,
            "best_r2": res[best_rank],
        })

    import pandas as pd

    df = pd.DataFrame(rows)
    layer_avg_best = float(df["best_r2"].mean())
    layer_avg_full = float(df["full"].mean())

    # Part B — Tuned-Lens-implied update (the Step-5 gate baseline).
    tlcfg = cfg.get("tuned_lens")
    gate = None
    if tlcfg:
        from layer_dynamics.diagnostics.tuned_lens import load_lens, tl_implied_deltas

        run.log(f"loading Tuned Lens {tlcfg['resource_id']} ...")
        lens = load_lens(tlcfg["resource_id"])
        tl_dlt = tl_implied_deltas(lens, rs)  # [N, L, D]
        tl_r2 = []
        for ell in range(n_layers):
            ref = np.broadcast_to(dlt[tr, ell].mean(axis=0), dlt[te, ell].shape)
            tl_r2.append(pooled_r2(dlt[te, ell], tl_dlt[te, ell], ref))
        df["tuned_lens"] = tl_r2
        layer_avg_tl = float(np.mean(tl_r2))
        gate = {
            "ridge_layer_avg_best_rank_r2": layer_avg_best,
            "tuned_lens_layer_avg_r2": layer_avg_tl,
            "ridge_beats_tuned_lens": layer_avg_best > layer_avg_tl,
        }
        beats = gate["ridge_beats_tuned_lens"]
        run.log(f"GATE: ridge R²={layer_avg_best:.4f} vs TL R²={layer_avg_tl:.4f} "
                f"-> ridge {'BEATS' if beats else 'DOES NOT BEAT'} "
                f"TL (Step 5 AR test {'proceeds' if beats else 'skipped'})")

    df.to_csv(run.path("tables/ridge_r2.csv"), index=False)
    stats = {
        "n_train": len(tr), "n_test": len(te), "n_layers": n_layers, "d_model": d,
        "ranks": list(ranks),
        "layer_avg_best_rank_ridge_r2": layer_avg_best,
        "layer_avg_full_rank_r2": layer_avg_full,
        "mean_zero_r2": float(df["zero"].mean()),
        "mean_identity_r2": float(df["identity"].mean()),
        "best_rank_counts": df["best_rank"].value_counts().to_dict(),
        "per_layer_best_r2": df["best_r2"].round(4).tolist(),
        "tuned_lens_gate": gate,
    }
    run.save_json("tables/step3_stats.json", stats)
    run.log(f"layer-avg best-rank ridge R² = {layer_avg_best:.4f} (full-rank {layer_avg_full:.4f})")

    _plots(run, df, ranks)
    print("\n=== per-layer R² (best rank) ===")
    for r in rows:
        print(f"  ℓ={r['layer']:2d}  best_r2={r['best_r2']:.3f} (rank {r['best_rank']:3d})  "
              f"zero={r['zero']:.3f}  identity={r['identity']:.3f}")
    return 0


def _plots(run, df, ranks) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Heatmap R²(ℓ, rank) + full.
    cols = [f"r{r}" for r in ranks] + ["full"]
    mat = df[cols].to_numpy().T
    fig, ax = plt.subplots(figsize=(10, 3.2))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", origin="lower", vmin=0)
    ax.set_yticks(range(len(cols)), [*[str(r) for r in ranks], "full"])
    ax.set_xlabel("layer pair ℓ"); ax.set_ylabel("rank")
    ax.set_title("Δ_ℓ ridge held-out R²(ℓ, rank)")
    fig.colorbar(im, ax=ax, label="pooled R²")
    fig.tight_layout()
    fig.savefig(run.path("figures/r2_heatmap.svg"), bbox_inches="tight")
    plt.close(fig)

    # Per-layer best ridge vs zero baseline.
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(df["layer"], df["best_r2"], "-o", ms=3, label="ridge (best rank)")
    if "tuned_lens" in df:
        ax.plot(df["layer"], df["tuned_lens"], "-s", ms=3, color="tab:red",
                label="Tuned-Lens implied (gate baseline)")
    ax.plot(df["layer"], df["zero"], "--", color="grey", label="zero-pred")
    ax.axhline(0, color="k", lw=0.6, label="mean-pred (R²=0)")
    ax.set_xlabel("layer pair ℓ"); ax.set_ylabel("pooled R² on Δ_ℓ")
    ax.set_title("Depth-local predictability of Δ_ℓ"); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(run.path("figures/r2_by_layer.svg"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
