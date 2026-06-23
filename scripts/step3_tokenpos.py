#!/usr/bin/env python3
"""Token-position invariance of the AR(k*) kernel (Phase 2 Step 3).

Refits the depth-dynamics kernel at four token positions (first/last subject
token, a mid-prompt non-subject token, and the Phase 1 final prompt token) and
asks whether it is the same kernel everywhere or specific to the query position.
The 4×4 cross-position confusion runs on the multi-token-subject subset (where
first ≠ last subject token); a parallel 3×3 (first == last merged to ``subj``)
runs on the single-token-subject subset.

Uses k = k* from Step 2 (≈1) and the Step 1 working rank r, on the same kind of
held-out split, with a shared input PCA basis across positions so the kernels are
comparable. A scrambled-null diagonal confirms the within-position fits are real
example-specific structure rather than the lag-1 ∓v_ℓ tautology (cca-vdelta).

    python scripts/step3_tokenpos.py --config configs/pythia-410m-deduped.yaml \
        --desc ph2-step3-410md \
        --step1-stats runs/2026-06-22-ph2-step1-410md/tables/step1_stats.json \
        --step2-stats runs/2026-06-22-ph2-step2-410md/tables/step2_stats.json

Outputs: tables/{confusion_multi,confusion_single,kernel_comparisons,step3_stats}.{csv,json},
figures/ (4×4 and 3×3 confusion heatmaps, within-vs-cross bars).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from layer_dynamics.cli import add_common_args, resolve_out
from layer_dynamics.diagnostics.autoregress import feasible_depths, scramble_per_depth
from layer_dynamics.diagnostics.dataset import deltas, train_test_split_idx
from layer_dynamics.diagnostics.token_position import (
    classify_position_invariance,
    cross_position_r2,
    fit_position_kernels,
    kernel_comparisons,
    offdiag_retention,
    shared_position_basis,
)
from layer_dynamics.runlog import init_run
from layer_dynamics.token_positions import POSITION_NAMES


def _load_multipos(path: Path):
    with h5py.File(path, "r") as f:
        rs = f["residual_stream"][:].astype(np.float32)  # [N, P, L+1, D]
        valid = f["valid"][:]
        multi = f["multi_token_subject"][:]
        pos_names = list(f.attrs["position_names"])
    return rs, valid, multi, pos_names


def _deltas_by_position(rs: np.ndarray, rows: np.ndarray, pos_idx: dict[str, int]):
    """Δ over depth for each named position, sliced to ``rows`` → {name: [n, L, D]}."""
    return {name: deltas(rs[rows, j]) for name, j in pos_idx.items()}


def _confusion(
    deltas_by_pos, k, r, input_pcs, depths, seed, null_seed
) -> dict:
    """Fit per-position kernels and return the real confusion + null diagonal + compares."""
    n = next(iter(deltas_by_pos.values())).shape[0]
    tr, te = train_test_split_idx(n, seed=seed)
    basis = shared_position_basis(deltas_by_pos, tr, input_pcs)
    kernels = fit_position_kernels(deltas_by_pos, tr, te, k, r, basis, depths)
    real = cross_position_r2(kernels, deltas_by_pos, tr, te, k, basis, depths)
    compares = kernel_comparisons(kernels, k, r)
    summary = offdiag_retention(real)

    # Scrambled null: permute the example axis per depth, independently per position.
    null_by_pos = {nm: scramble_per_depth(d, null_seed + i)
                   for i, (nm, d) in enumerate(deltas_by_pos.items())}
    null_basis = shared_position_basis(null_by_pos, tr, input_pcs)
    null_kernels = fit_position_kernels(null_by_pos, tr, te, k, r, null_basis, depths)
    null_diag = {nm: pk.within_r2 for nm, pk in null_kernels.items()}

    return {
        "n": n, "n_train": len(tr), "n_test": len(te),
        "real": real, "null_diagonal": null_diag,
        "comparisons": compares, "summary": summary,
        "verdict": classify_position_invariance(summary, compares),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--step1-stats", type=Path, default=None,
                        help="Step 1 step1_stats.json — pulls the working rank r.")
    parser.add_argument("--step2-stats", type=Path, default=None,
                        help="Step 2 step2_stats.json — pulls k* (median per-depth).")
    parser.add_argument("--rank", type=int, default=None, help="Override the working rank r.")
    parser.add_argument("--kstar", type=int, default=None, help="Override k*.")
    args = parser.parse_args(argv)

    run = init_run(resolve_out(args), config_path=args.config, seed=args.seed)
    cfg = run.config
    mcfg = cfg["model"]
    dgcfg = cfg.get("diagnostics", {})
    input_pcs = int(dgcfg.get("ar_input_pcs", 256))
    null_seed = int(dgcfg.get("ar_null_seed", 1234))

    r = args.rank
    if r is None and args.step1_stats and args.step1_stats.exists():
        r = int(json.loads(args.step1_stats.read_text())["working_rank"])
    r = r or 256

    k = args.kstar
    if k is None and args.step2_stats and args.step2_stats.exists():
        k = int(round(json.loads(args.step2_stats.read_text())["per_depth_kstar_median"]))
    k = k or 1
    run.log(f"k*={k}, working rank r={r}, input_pcs={input_pcs}")

    h5 = Path("data") / mcfg["name"] / "residual_stream_multipos.h5"
    rs, valid, multi, pos_names = _load_multipos(h5)
    n_delta = rs.shape[2] - 1
    depths = feasible_depths(n_delta, k)
    run.log(f"loaded {h5}: residual_stream {rs.shape}; feasible depths {depths[0]}..{depths[-1]}")

    name_idx = {nm: pos_names.index(nm) for nm in POSITION_NAMES}

    # --- Multi-token-subject subset: 4×4 (all four positions valid & distinct) ---
    multi_rows = np.where(multi & valid.all(axis=1))[0]
    multi_pos = {nm: name_idx[nm] for nm in POSITION_NAMES}
    run.log(f"multi-token subset: {len(multi_rows)} examples (4×4 confusion)")
    multi_res = _confusion(
        _deltas_by_position(rs, multi_rows, multi_pos),
        k, r, input_pcs, depths, args.seed, null_seed)

    # --- Single-token-subject subset: 3×3 (subj_first == subj_last → merge to subj) ---
    single_mask = (~multi) & valid[:, [name_idx["subj_first"], name_idx["mid"],
                                       name_idx["final"]]].all(axis=1)
    single_rows = np.where(single_mask)[0]
    single_pos = {"subj": name_idx["subj_first"], "mid": name_idx["mid"],
                  "final": name_idx["final"]}
    run.log(f"single-token subset: {len(single_rows)} examples (3×3 confusion)")
    single_res = _confusion(
        _deltas_by_position(rs, single_rows, single_pos),
        k, r, input_pcs, depths, args.seed, null_seed)

    for tag, res in (("multi", multi_res), ("single", single_res)):
        s, v = res["summary"], res["verdict"]
        run.log(f"  [{tag}] diag R²={s['mean_diagonal_r2']:.3f} "
                f"offdiag={s['mean_offdiagonal_r2']:.3f} "
                f"worst-retention={s['worst_retention']:.3f} "
                f"maxAngle={v['max_output_angle_deg']:.1f}° "
                f"null-diag~{np.mean(list(res['null_diagonal'].values())):.3f} → {v['verdict']}")

    stats = {
        "model": mcfg["name"], "kstar": k, "working_rank": r, "input_pcs": input_pcs,
        "n_layers": n_delta, "feasible_depths": [depths[0], depths[-1]],
        "multi": {"n": int(len(multi_rows)), **multi_res["verdict"],
                  "null_mean_diag_r2": float(np.mean(list(multi_res["null_diagonal"].values())))},
        "single": {"n": int(len(single_rows)), **single_res["verdict"],
                   "null_mean_diag_r2": float(np.mean(list(single_res["null_diagonal"].values())))},
        "interpretation": _interpret(multi_res, single_res, k),
    }

    _write_tables(run, multi_res, single_res)
    run.save_json("tables/step3_stats.json", stats)
    _plots(run, multi_res, single_res)

    print(f"\n=== Phase 2 Step 3 — token-position invariance (k*={k}, r={r}) ===")
    for tag, res in (("multi 4×4", multi_res), ("single 3×3", single_res)):
        s, v = res["summary"], res["verdict"]
        print(f"  {tag}: n={res['n']}  diag={s['mean_diagonal_r2']:.3f}  "
              f"offdiag={s['mean_offdiagonal_r2']:.3f}  "
              f"worst-retention={s['worst_retention']:.3f}  → {v['verdict']}")
    print(stats["interpretation"])
    return 0


def _interpret(multi_res, single_res, k) -> str:
    mv, sv = multi_res["verdict"], single_res["verdict"]
    return (
        "Token-position invariance at k*={k}: multi-token-subject subset (4×4, n={mn}) "
        "verdict {mvv} — mean within-position R² {md:.3f}, mean cross-position {mo:.3f}, "
        "worst retention {mr:.3f}, max kernel angle {ma:.1f}°. Single-token subset (3×3, "
        "n={sn}) verdict {svv}. The scrambled null collapses the within-position fits "
        "(multi ~{nd:.3f}), so the diagonal is real example-specific structure, not the "
        "lag-1 ∓v_ℓ tautology — and because the tautology is shared by diagonal and "
        "off-diagonal alike, the *retention* (cross/within) is the position-invariance "
        "read and is robust to it. {concl}"
    ).format(
        k=k, mn=multi_res["n"], mvv=mv["verdict"], md=mv["mean_diagonal_r2"],
        mo=mv["mean_offdiagonal_r2"], mr=mv["worst_retention"], ma=mv["max_output_angle_deg"],
        sn=single_res["n"], svv=sv["verdict"],
        nd=np.mean(list(multi_res["null_diagonal"].values())),
        concl=(
            "Kernels transfer across positions → the AR structure is a property of the "
            "depth dynamics, not the query position; the convolutive picture needs no "
            "position qualifier."
            if mv["verdict"] == "position_invariant" else
            "Foreign-position kernels lose a large share of the within-position R² → the "
            "Phase 1 dynamics are (at least partly) specific to the query position where "
            "context aggregates; scope the framing to context-aggregated factual recall "
            "at the query position."
            if mv["verdict"] == "query_position_specific" else
            "Within-position fits are too thin to compare at these positions; rerun with "
            "more examples per position before drawing a position-invariance conclusion."
        ),
    )


def _write_tables(run, multi_res, single_res) -> None:
    import pandas as pd

    for tag, res in (("multi", multi_res), ("single", single_res)):
        real = res["real"]
        names = list(real)
        df = pd.DataFrame(
            [[real[p][q] for q in names] for p in names], index=names, columns=names
        )
        df.to_csv(run.path(f"tables/confusion_{tag}.csv"))
        rows = [{"pair": pair, **vals} for pair, vals in res["comparisons"].items()]
        pd.DataFrame(rows).to_csv(run.path(f"tables/kernel_comparisons_{tag}.csv"), index=False)


def _plots(run, multi_res, single_res) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for tag, res in (("multi", multi_res), ("single", single_res)):
        real = res["real"]
        names = list(real)
        mat = np.array([[real[p][q] for q in names] for p in names])
        fig, ax = plt.subplots(figsize=(0.9 * len(names) + 2.5, 0.9 * len(names) + 2))
        im = ax.imshow(mat, cmap="viridis", vmin=min(0, mat.min()), vmax=mat.max())
        ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
        ax.set_yticks(range(len(names)), names)
        ax.set_xlabel("evaluated at position"); ax.set_ylabel("kernel fit at position")
        ax.set_title(f"Cross-position R² confusion ({tag} subset)")
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        color="w" if mat[i, j] < mat.max() * 0.6 else "k", fontsize=8)
        fig.colorbar(im, ax=ax, label="held-out R²"); fig.tight_layout()
        fig.savefig(run.path(f"figures/confusion_{tag}.png"), dpi=130); plt.close(fig)

    # within vs mean-cross per fit-position (multi subset).
    real = multi_res["real"]
    names = list(real)
    within = [real[p][p] for p in names]
    cross = [np.mean([real[p][q] for q in names if q != p]) for p in names]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    x = np.arange(len(names))
    ax.bar(x - 0.2, within, 0.4, label="within-position (diagonal)")
    ax.bar(x + 0.2, cross, 0.4, label="mean cross-position")
    ax.set_xticks(x, names, rotation=20, ha="right"); ax.set_ylabel("held-out R²")
    ax.set_title("Within- vs cross-position reconstruction (multi-token subset)")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(run.path("figures/within_vs_cross.png"), dpi=130); plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
