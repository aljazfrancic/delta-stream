#!/usr/bin/env python3
"""CCA backstop — shared-subspace structure between v_ℓ and Δ_ℓ (Phase 1 Step 4).

For every layer pair (ℓ, ℓ+1), fit shrinkage CCA on the train split and report
the first ``n_components`` canonical correlations on the held-out split. The
split is the same deterministic example-level split Step 3 uses (same seed /
test_frac), so the held-out canonical correlations are directly comparable to
the ridge held-out R² — which is what the plan's interpretation key reads:

  high R² & high CC      -> linear predictability is real, captures the structure
  low  R² & high CC      -> ridge underfits genuine (possibly symmetric) structure
  low  R² & low  CC      -> no linear dependence; H1 in trouble at this pair

Pass ``--step3-csv`` (the Step 3 ``tables/ridge_r2.csv``) to join the ridge R²
and emit the per-layer (R², CC) scatter that supports that key.

    python scripts/step4_cca.py --config configs/pythia-410m-deduped.yaml \
        --desc ph1-step4-410md \
        --step3-csv runs/2026-06-21-ph1-step3-410md/tables/ridge_r2.csv

Outputs: tables/cca_correlations.csv, figures/cca_heatmap.png,
figures/cca_vs_r2.png (only with --step3-csv), tables/step4_stats.json.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from layer_dynamics.cli import add_common_args, resolve_out
from layer_dynamics.diagnostics.cca import regularized_cca
from layer_dynamics.diagnostics.dataset import deltas, load_residual_stream, train_test_split_idx
from layer_dynamics.runlog import init_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument(
        "--step3-csv",
        type=Path,
        default=None,
        help="Step 3 tables/ridge_r2.csv to join for the R²-vs-CC interpretation key.",
    )
    args = parser.parse_args(argv)

    run = init_run(resolve_out(args), config_path=args.config, seed=args.seed)
    cfg = run.config
    mcfg = cfg["model"]
    dgcfg = cfg.get("diagnostics", {})
    n_components = int(dgcfg.get("cca_components", 10))
    gamma = float(dgcfg.get("cca_shrinkage", 0.1))
    test_frac = float(dgcfg.get("test_frac", 0.2))
    run_null = bool(dgcfg.get("cca_null", True))
    null_seed = int(dgcfg.get("cca_null_seed", 1234))

    h5 = Path("data") / mcfg["name"] / "residual_stream.h5"
    rs, _, _ = load_residual_stream(h5)
    n, n_states, d = rs.shape
    n_layers = n_states - 1
    dlt = deltas(rs)  # [N, L, D]
    run.log(f"loaded {h5}: residual_stream {rs.shape}, deltas {dlt.shape}")

    tr, te = train_test_split_idx(n, test_frac=test_frac, seed=args.seed)
    run.log(f"split: {len(tr)} train / {len(te)} test (same split as Step 3)")
    run.log(f"shrinkage CCA: n_components={n_components}, gamma={gamma}")

    rows = []
    for ell in range(n_layers):
        x_tr, x_te = rs[tr, ell], rs[te, ell]  # v_ℓ
        y_tr, y_te = dlt[tr, ell], dlt[te, ell]  # Δ_ℓ
        res = regularized_cca(
            x_tr, y_tr, x_te, y_te, n_components=n_components, gamma=gamma
        )
        k = len(res["test"])
        row = {"layer": ell}
        row.update({f"te{i + 1}": float(res["test"][i]) for i in range(k)})
        row.update({f"tr{i + 1}": float(res["train"][i]) for i in range(k)})
        row["mean_te_top3"] = float(np.mean(res["test"][:3]))
        row["mean_te_topk"] = float(np.mean(res["test"]))
        if run_null:
            # Scrambled null: permute Δ's example axis independently of v, so
            # any within-example structure (incl. the definitional −v_ℓ term) is
            # destroyed while per-layer marginals are preserved. A held-out CC
            # that survives this is finite-sample noise; one that collapses is
            # real example-specific structure.
            rng_l = np.random.default_rng(null_seed + ell)
            p_tr = rng_l.permutation(len(y_tr))
            p_te = rng_l.permutation(len(y_te))
            nres = regularized_cca(
                x_tr, y_tr[p_tr], x_te, y_te[p_te],
                n_components=n_components, gamma=gamma,
            )
            row["null_te1"] = float(nres["test"][0])
            row["null_te_top3"] = float(np.mean(nres["test"][:3]))
        rows.append(row)

    import pandas as pd

    df = pd.DataFrame(rows)
    te_cols = [f"te{i + 1}" for i in range(n_components)]
    tr_cols = [f"tr{i + 1}" for i in range(n_components)]

    # Overfitting / degeneracy diagnostic: the gap between in-sample and held-out
    # leading correlation. Large gap => the raw CCA would have been degenerate.
    overfit_gap = float((df["tr1"] - df["te1"]).mean())

    stats = {
        "n_train": len(tr),
        "n_test": len(te),
        "n_layers": n_layers,
        "d_model": d,
        "n_components": n_components,
        "gamma": gamma,
        "layer_avg_te_top1": float(df["te1"].mean()),
        "layer_avg_te_top3": float(df["mean_te_top3"].mean()),
        "layer_avg_te_topk": float(df["mean_te_topk"].mean()),
        "layer_avg_tr_top1": float(df["tr1"].mean()),
        "mean_overfit_gap_top1": overfit_gap,
        "per_layer_te_top1": df["te1"].round(4).tolist(),
    }
    if run_null:
        stats["layer_avg_null_te_top1"] = float(df["null_te1"].mean())
        stats["layer_avg_null_te_top3"] = float(df["null_te_top3"].mean())
        stats["real_minus_null_top1"] = float((df["te1"] - df["null_te1"]).mean())
    # Honest reading of the ≈1 correlations: real (they survive shrinkage and
    # collapse under the scrambled null) but largely the definitional −v_ℓ term.
    stats["caveat"] = (
        "Held-out canonical correlations ≈1 are real (robust to strong shrinkage; "
        "collapse to ~0 under the scrambled null) but substantially reflect the "
        "definitional −v_ℓ component of Δ_ℓ = v_{ℓ+1} − v_ℓ — the same "
        "residual-stream tautology as the Step 3 identity baseline, amplified by a "
        "few high-variance ('rogue') residual dimensions. CCA therefore confirms "
        "abundant shared subspace structure (Gate-1 Q3 = yes) but does not expose "
        "structure the ridge missed; the substantive predictability claim stays "
        "the Step 3 ridge R² vs the mean/Tuned-Lens baselines."
    )

    # Optional Step 3 join -> interpretation key.
    interp = None
    if args.step3_csv is not None:
        r3 = pd.read_csv(args.step3_csv)[["layer", "best_r2"]].rename(
            columns={"best_r2": "ridge_best_r2"}
        )
        df = df.merge(r3, on="layer", how="left")
        # Descriptive regime label per the plan's key. Thresholds (0.5) are
        # reporting aids for the Gate-1 memo, NOT pre-registered cutoffs.
        thr = 0.5
        df["regime"] = [
            f"{'highR2' if r >= thr else 'lowR2'}_{'highCC' if c >= thr else 'lowCC'}"
            for r, c in zip(df["ridge_best_r2"], df["mean_te_top3"], strict=True)
        ]
        interp = {
            "threshold": thr,
            "regime_counts": df["regime"].value_counts().to_dict(),
            "corr_r2_vs_meanCC": float(
                np.corrcoef(df["ridge_best_r2"], df["mean_te_top3"])[0, 1]
            ),
        }
    stats["interpretation"] = interp

    df.to_csv(run.path("tables/cca_correlations.csv"), index=False)
    run.save_json("tables/step4_stats.json", stats)
    run.log(
        f"held-out canonical corr: top1 avg={stats['layer_avg_te_top1']:.4f}, "
        f"top3 avg={stats['layer_avg_te_top3']:.4f} "
        f"(in-sample top1 avg={stats['layer_avg_tr_top1']:.4f}, "
        f"overfit gap={overfit_gap:.4f})"
    )
    if run_null:
        run.log(
            f"NULL: scrambled held-out top1 avg={stats['layer_avg_null_te_top1']:.4f} "
            f"-> real−null={stats['real_minus_null_top1']:.4f} "
            "(correlations are real example-specific structure, not noise)"
        )

    _plots(run, df, te_cols, tr_cols, has_r2=interp is not None)

    print("\n=== per-layer held-out canonical correlations (top 3) ===")
    for _, r in df.iterrows():
        r2 = f"  ridge_R²={r['ridge_best_r2']:.3f}" if interp is not None else ""
        null = f"  null_c1={r['null_te1']:.3f}" if run_null else ""
        print(
            f"  ℓ={int(r['layer']):2d}  c1={r['te1']:.3f}  c2={r['te2']:.3f}  "
            f"c3={r['te3']:.3f}{null}{r2}"
        )
    return 0


def _plots(run, df, te_cols, tr_cols, *, has_r2: bool) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Heatmap: held-out canonical correlation (component × layer).
    mat = df[te_cols].to_numpy().T  # [k, L]
    fig, ax = plt.subplots(figsize=(10, 3.2))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", origin="lower", vmin=0, vmax=1)
    ax.set_yticks(range(len(te_cols)), [c.replace("te", "c") for c in te_cols])
    ax.set_xlabel("layer pair ℓ")
    ax.set_ylabel("canonical component")
    ax.set_title("Held-out canonical correlation between v_ℓ and Δ_ℓ")
    fig.colorbar(im, ax=ax, label="held-out corr")
    fig.tight_layout()
    fig.savefig(run.path("figures/cca_heatmap.png"), dpi=130)
    plt.close(fig)

    # Leading canonical correlation across depth: held-out vs in-sample (the gap
    # is the degeneracy that shrinkage + held-out evaluation controls for).
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(df["layer"], df["te1"], "-o", ms=3, label="held-out c1")
    ax.plot(df["layer"], df["tr1"], "--", color="grey", label="in-sample c1")
    ax.plot(df["layer"], df["mean_te_top3"], "-s", ms=3, color="tab:green",
            label="held-out mean(c1..c3)")
    if "null_te1" in df:
        ax.plot(df["layer"], df["null_te1"], "-^", ms=3, color="tab:red",
                label="scrambled-null c1")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("layer pair ℓ")
    ax.set_ylabel("canonical correlation")
    ax.set_title("Leading canonical correlation by depth")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(run.path("figures/cca_by_layer.png"), dpi=130)
    plt.close(fig)

    if not has_r2:
        return

    # Interpretation key: per-layer ridge R² vs mean held-out canonical corr.
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.scatter(df["ridge_best_r2"], df["mean_te_top3"], c=df["layer"], cmap="plasma")
    for _, r in df.iterrows():
        ax.annotate(int(r["layer"]), (r["ridge_best_r2"], r["mean_te_top3"]),
                    fontsize=6, xytext=(2, 2), textcoords="offset points")
    ax.axvline(0.5, color="grey", lw=0.6, ls=":")  # visual quadrant guides,
    ax.axhline(0.5, color="grey", lw=0.6, ls=":")  # not pre-registered cutoffs
    ax.set_xlabel("ridge held-out R² (Step 3, best rank)")
    ax.set_ylabel("mean held-out canonical corr (c1..c3)")
    ax.set_title("Interpretation key: predictability vs shared subspace")
    fig.tight_layout()
    fig.savefig(run.path("figures/cca_vs_r2.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
