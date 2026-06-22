#!/usr/bin/env python3
"""Null 2 — context perturbation (the critical control), Phase 1 Step 5.

The bare s+r prompts carry no tokens outside the subject-relation span, so to
perturb context we add a fixed-length neutral preamble and vary it:

  wrapped baseline : "In school I learned that " + prompt
  perturbed        : <P random single-token function words> + prompt   (same P tokens)

Both variants place the subject-relation span and the final prompt token at the
same absolute positions (both prepend exactly P preamble tokens after BOS), so any
change in the AR/Granger structure is attributable to the preamble *content*
leaking through attention into the query position — not to a position shift.

We recollect residual streams for both, drop examples where the wrapped baseline no
longer recovers the correct top-1 token (the wrapping itself can shift predictions),
then run the AR(k) fit and the Granger grid on each and compare:

  structure unchanged  -> the cross-layer dynamics at the query token are a property
                          of subject-relation processing — depth-intrinsic.
  structure changes    -> the apparent dynamics reflect context carried by hidden
                          variables — a cross-context, not depth-propagation, effect.

Both outcomes are publishable; they say different things about H1.

    python scripts/step5_context_null.py --config configs/pythia-410m-deduped.yaml \
        --desc ph1-step5ctx-410md

Outputs: tables/context_ar_r2.csv, tables/context_granger.csv,
tables/step5_context_stats.json, figures/context_compare.png.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from layer_dynamics.activations import collect_trajectories_from_ids, recovers_top1
from layer_dynamics.cli import add_common_args, resolve_out
from layer_dynamics.device import describe
from layer_dynamics.diagnostics.autoregress import (
    LAG_SWEEP,
    fit_ar_pooled,
    fit_delta_pca,
    granger_lag_grid,
)
from layer_dynamics.diagnostics.dataset import deltas, train_test_split_idx
from layer_dynamics.modeling import load_model
from layer_dynamics.runlog import init_run

PREAMBLE = "In school I learned that "

# Candidate neutral function words; filtered at runtime to those that tokenise to a
# single token with a leading space (so a P-word preamble is exactly P tokens).
FUNCTION_WORDS = (
    "the a of and to in that it is was for on with as by at an be this which or from "
    "but not are were has had have so if then than when where what who how all some "
    "any no one two into over under about after before also just only very more most "
    "such each both few many much other same new old here there now they we you"
).split()


def _single_token_pool(model) -> list[int]:
    """Function words that encode to exactly one token id (with a leading space)."""
    tok = model.tokenizer
    pool = []
    for w in FUNCTION_WORDS:
        ids = tok(" " + w, add_special_tokens=False)["input_ids"]
        if len(ids) == 1:
            pool.append(ids[0])
    return sorted(set(pool))


def _ar_sweep(dlt_tr, dlt_te, lags, ranks, input_pcs):
    basis = fit_delta_pca(dlt_tr, input_pcs) if input_pcs else None
    rows, best = [], (-1, -1, -np.inf)
    for k in lags:
        fit = fit_ar_pooled(dlt_tr, dlt_te, k, ranks=ranks, input_basis=basis,
                            input_pcs=input_pcs)
        rows.append({"k": k, "best_rank": fit.best_rank, "best_r2": fit.best_r2,
                     "full": fit.r2["full"], "alpha": fit.alpha})
        if fit.best_r2 > best[2]:
            best = (k, fit.best_rank, fit.best_r2)
    return rows, best


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--n-context", type=int, default=1000,
                        help="Number of examples to sample for the perturbation test.")
    args = parser.parse_args(argv)

    run = init_run(resolve_out(args), config_path=args.config, seed=args.seed)
    cfg = run.config
    mcfg, ccfg, dgcfg = cfg["model"], cfg.get("collection", {}), cfg.get("diagnostics", {})
    ranks = tuple(dgcfg.get("ranks", [16, 32, 64, 128]))
    lags = tuple(dgcfg.get("ar_lags", LAG_SWEEP))
    q_pca = int(dgcfg.get("granger_pca", 32))
    input_pcs = int(dgcfg.get("ar_input_pcs", 256))
    test_frac = float(dgcfg.get("test_frac", 0.2))
    batch = int(ccfg.get("batch_size", 4))

    import pandas as pd

    parquet = Path("data") / mcfg["name"] / "filtered_triples.parquet"
    df = pd.read_parquet(parquet).reset_index(drop=True)
    rng = np.random.default_rng(args.seed)
    n_ctx = min(args.n_context, len(df))
    idx = rng.choice(len(df), size=n_ctx, replace=False)
    sub = df.iloc[idx].reset_index(drop=True)
    run.log(f"sampled {n_ctx}/{len(df)} examples for the context-perturbation test")

    model = load_model(mcfg["name"], device=mcfg.get("device", "auto"), dtype=mcfg["dtype"])
    run.log(f"model {mcfg['name']} on {describe(next(model.parameters()).device.type)}")

    pool = _single_token_pool(model)
    preamble_ids = model.tokenizer(PREAMBLE, add_special_tokens=False)["input_ids"]
    p_len = len(preamble_ids)
    boundary = preamble_ids[-1]  # trailing-space token; shared so the subject is
    # preceded by the same whitespace in both variants (only the content words vary).
    run.log(f"preamble {PREAMBLE!r} -> {p_len} tokens; "
            f"single-token function-word pool size {len(pool)}; boundary token {boundary}")

    # Build wrapped + perturbed inputs at the TOKEN level so the subject-relation span
    # lands at identical absolute positions. Both share the example's own content ids
    # and a length-p_len preamble; the perturbed preamble swaps the first p_len-1
    # content-word tokens for random single-token function words, keeping the boundary.
    wrapped_ids, perturbed_ids = [], []
    for prompt in sub["prompt"].tolist():
        content = model.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        sampled = rng.choice(pool, size=p_len - 1, replace=True).tolist()
        wrapped_ids.append(list(preamble_ids) + content)
        perturbed_ids.append(sampled + [boundary] + content)
    align_ok = sum(
        len(w) == len(p) for w, p in zip(wrapped_ids, perturbed_ids, strict=True)
    )
    run.log(f"token-length alignment wrapped==perturbed: {align_ok}/{n_ctx} "
            "(guaranteed by construction)")

    run.log("collecting wrapped-baseline residual streams ...")
    rs_w = collect_trajectories_from_ids(model, wrapped_ids, batch_size=batch)
    run.log("collecting perturbed residual streams ...")
    rs_p = collect_trajectories_from_ids(model, perturbed_ids, batch_size=batch)

    # Keep only examples where the wrapped baseline still recovers the correct top-1.
    keep = []
    for i in range(n_ctx):
        rec = model.tokenizer.decode([recovers_top1(model, rs_w[i, -1])])
        keep.append(rec == sub.iloc[i]["top1_token"])
    keep = np.array(keep)
    n_keep = int(keep.sum())
    run.log(f"wrapped baseline recovers correct top-1 on {n_keep}/{n_ctx} "
            f"({100 * n_keep / n_ctx:.1f}%); dropping the rest")
    rs_w, rs_p = rs_w[keep], rs_p[keep]

    dlt_w, dlt_p = deltas(rs_w), deltas(rs_p)
    tr, te = train_test_split_idx(n_keep, test_frac=test_frac, seed=args.seed)

    run.log("AR(k) + Granger on wrapped baseline ...")
    ar_w, best_w = _ar_sweep(dlt_w[tr], dlt_w[te], lags, ranks, input_pcs)
    g_w = granger_lag_grid(dlt_w, lags=lags, q_pca=q_pca)
    run.log("AR(k) + Granger on perturbed variant ...")
    ar_p, best_p = _ar_sweep(dlt_p[tr], dlt_p[te], lags, ranks, input_pcs)
    g_p = granger_lag_grid(dlt_p, lags=lags, q_pca=q_pca)

    bonf = 0.05 / g_w["n_tests"]
    thr = -np.log10(bonf)
    sig_w = int(np.nansum(g_w["neg_log10_p"] > thr))
    sig_p = int(np.nansum(g_p["neg_log10_p"] > thr))
    # Heatmap similarity: correlation of the −log10 p maps over shared finite cells.
    a, b = g_w["neg_log10_p"].ravel(), g_p["neg_log10_p"].ravel()
    m = np.isfinite(a) & np.isfinite(b)
    heatmap_corr = float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 2 else np.nan

    rows = ([{"variant": "wrapped", **r} for r in ar_w]
            + [{"variant": "perturbed", **r} for r in ar_p])
    pd.DataFrame(rows).to_csv(run.path("tables/context_ar_r2.csv"), index=False)
    _granger_csv(run, "tables/context_granger.csv", g_w, g_p)

    stats = {
        "n_context": n_ctx, "n_kept": n_keep, "preamble_tokens": p_len,
        "pool_size": len(pool), "align_ok": align_ok,
        "ar_best_wrapped": {"k": best_w[0], "rank": best_w[1], "r2": best_w[2]},
        "ar_best_perturbed": {"k": best_p[0], "rank": best_p[1], "r2": best_p[2]},
        "ar_r2_delta_perturbed_minus_wrapped": best_p[2] - best_w[2],
        "granger_n_tests": g_w["n_tests"], "granger_bonferroni_alpha": bonf,
        "granger_significant_wrapped": sig_w, "granger_significant_perturbed": sig_p,
        "granger_heatmap_corr": heatmap_corr,
        "interpretation": (
            "If the AR R² and the Granger heatmap are ~unchanged (high heatmap_corr, "
            "similar significant-cell counts) the cross-layer dynamics at the query "
            "token are depth-intrinsic (Gate-1 Q6 = survives). A substantial change "
            "means the structure is context-dependent (Q6 = no) -> the Gate-1 'pivot: "
            "context-reconstruction dynamics' route."
        ),
    }
    run.save_json("tables/step5_context_stats.json", stats)
    _plot(run, ar_w, ar_p, g_w, g_p, thr)

    print("\n=== context perturbation: wrapped vs perturbed ===")
    print(f"kept {n_keep}/{n_ctx} (wrapped top-1 recovered)")
    print(f"AR best R²: wrapped={best_w[2]:.4f} (k={best_w[0]})  "
          f"perturbed={best_p[2]:.4f} (k={best_p[0]})  Δ={best_p[2] - best_w[2]:+.4f}")
    print(f"Granger Bonferroni-sig cells: wrapped={sig_w}  perturbed={sig_p}  "
          f"(of {g_w['n_tests']})")
    print(f"Granger −log10 p heatmap correlation: {heatmap_corr:.4f}")
    return 0


def _granger_csv(run, name, g_w, g_p) -> None:
    import pandas as pd

    lags = g_w["lags"]
    rows = []
    for ell in range(g_w["F"].shape[0]):
        for ji, j in enumerate(lags):
            fw = g_w["F"][ell, ji]
            if np.isnan(fw):
                continue
            rows.append({
                "depth": ell, "lag": j,
                "nlp_wrapped": float(g_w["neg_log10_p"][ell, ji]),
                "nlp_perturbed": float(g_p["neg_log10_p"][ell, ji]),
                "F_wrapped": float(fw), "F_perturbed": float(g_p["F"][ell, ji]),
            })
    pd.DataFrame(rows).to_csv(run.path(name), index=False)


def _plot(run, ar_w, ar_p, g_w, g_p, thr) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.4))
    kw = [r["k"] for r in ar_w]
    axes[0].plot(kw, [r["best_r2"] for r in ar_w], "-o", label="wrapped")
    axes[0].plot(kw, [r["best_r2"] for r in ar_p], "--s", label="perturbed")
    axes[0].axhline(0, color="k", lw=0.6)
    axes[0].set_xlabel("AR order k"); axes[0].set_ylabel("held-out R² on Δ")
    axes[0].set_title("AR(k): wrapped vs perturbed"); axes[0].legend(fontsize=8)

    vmax = float(np.nanmax([np.nanmax(g_w["neg_log10_p"]), np.nanmax(g_p["neg_log10_p"])]))
    for ax, grid, tag in ((axes[1], g_w, "wrapped"), (axes[2], g_p, "perturbed")):
        im = ax.imshow(grid["neg_log10_p"].T, aspect="auto", cmap="magma",
                       origin="lower", vmin=0, vmax=vmax)
        ax.set_yticks(range(len(grid["lags"])), [str(j) for j in grid["lags"]])
        ax.set_xlabel("depth ℓ"); ax.set_ylabel("lag j")
        ax.set_title(f"Granger −log10 p ({tag}) thr={thr:.1f}")
        fig.colorbar(im, ax=ax, label="−log10 p")
    fig.tight_layout(); fig.savefig(run.path("figures/context_compare.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
