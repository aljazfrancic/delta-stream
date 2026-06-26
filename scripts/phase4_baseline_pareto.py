#!/usr/bin/env python3
"""Phase 4 Step 1 — assemble the headline reconstruction-vs-sparsity comparison.

Stitches the baselines already computed into one Pareto figure + table (the plan's "headline
plot: L0 vs Δ reconstruction, one curve per method"):

  * **My method** — Branch B SSM (Gate-3 Pareto, ``ssm`` rows).
  * **Baseline A** — matched per-layer sparse dictionary on Δ (Gate-3 ``baseline`` rows, the
    ``use_state=False`` SSM: identical optimiser/sources/L1 sweep, cross-layer state ablated).
  * **Baseline B** — pretrained MLSAE differenced into Δ̂ (``phase4_baseline_mlsae.py``), a
    single point at the SAE's native sparsity. Also the closest feasible CLT proxy.
  * **Baseline D (CLT)** — recorded as a compute limitation (no public Pythia-410m CLT; training
    even a scoped CLT needs more than the 3 GB dev GPU). Annotated, not plotted.

No fitting here — pure assembly of saved JSON, so it is instant and regenerates the figure
from intermediates (Step-5 reproducibility).

    python scripts/phase4_baseline_pareto.py --gate3 runs/2026-06-25-ph3-gate3-410md \
        --mlsae runs/2026-06-26-ph4-baseline-mlsae-410md --desc ph4-baseline-pareto-410md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from layer_dynamics.cli import add_common_args, resolve_out
from layer_dynamics.runlog import init_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_args(parser)
    parser.add_argument("--gate3", type=Path, required=True, help="Gate-3 Pareto run dir.")
    parser.add_argument("--mlsae", type=Path, required=True, help="Baseline B (MLSAE) run dir.")
    args = parser.parse_args(argv)

    run = init_run(resolve_out(args), config_path=args.config, seed=args.seed)

    g3 = json.loads((args.gate3 / "tables" / "gate3_pareto.json").read_text())
    mb = json.loads((args.mlsae / "tables" / "baseline_mlsae.json").read_text())

    ssm = sorted((r for r in g3["rows"] if r["variant"] == "ssm"), key=lambda r: r["test_l0"])
    base = sorted((r for r in g3["rows"] if r["variant"] == "baseline"), key=lambda r: r["test_l0"])
    mlsae_l0 = g3["sae_init_l0"]          # MLSAE native sparsity (TopK k/M, pre-LASSO)
    mlsae_r2 = mb["delta_r2_pooled_test"]

    summary = {
        "ssm": [{"l0": r["test_l0"], "r2": r["test_r2"], "lambda_l1": r["lambda_l1"]} for r in ssm],
        "baseline_A_perlayer_dict": [
            {"l0": r["test_l0"], "r2": r["test_r2"], "lambda_l1": r["lambda_l1"]} for r in base],
        "baseline_B_mlsae_implied_delta": {"l0": mlsae_l0, "r2": mlsae_r2},
        "baseline_D_clt": "limitation: no public Pythia-410m CLT; scoped training exceeds 3 GB GPU",
        "matched_l0_gaps": g3["matched_l0_gaps"],
        "ssm_best_test_r2": max(r["test_r2"] for r in ssm),
        "baseline_A_best_test_r2": max(r["test_r2"] for r in base),
    }
    run.save_json("tables/baseline_comparison.json", summary)
    _plot(ssm, base, (mlsae_l0, mlsae_r2), run.path("figures/baseline_pareto.png"))

    print("\n=== Phase 4 Step 1 — reconstruction-vs-sparsity comparison (Δ held-out R²) ===")
    print(f"  {'method':>34} {'L0':>9} {'R²':>8}")
    print(f"  {'Branch B SSM (best)':>34} {ssm[0]['test_l0']:>9.4f} "
          f"{summary['ssm_best_test_r2']:>+8.3f}")
    print(f"  {'Baseline A: per-layer dict on Δ':>34} {base[0]['test_l0']:>9.4f} "
          f"{summary['baseline_A_best_test_r2']:>+8.3f}")
    print(f"  {'Baseline B: MLSAE implied Δ':>34} {mlsae_l0:>9.4f} {mlsae_r2:>+8.3f}")
    print(f"  {'Baseline D: CLT':>34} {'—':>9} {'limitation':>8}")
    print(f"  SSM − Baseline A gap @ matched L0: "
          f"{min(g['gap'] for g in g3['matched_l0_gaps']):+.3f} … "
          f"{max(g['gap'] for g in g3['matched_l0_gaps']):+.3f}")
    return 0


def _plot(ssm, base, mlsae_pt, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    ax.plot([r["test_l0"] for r in ssm], [r["test_r2"] for r in ssm], "o-", color="C0",
            label="Branch B SSM (my method)")
    ax.plot([r["test_l0"] for r in base], [r["test_r2"] for r in base], "s--", color="C1",
            label="Baseline A: per-layer dict on Δ")
    ax.scatter([mlsae_pt[0]], [mlsae_pt[1]], marker="*", s=160, color="C3", zorder=5,
               label="Baseline B: MLSAE implied Δ")
    ax.axhline(0.0, color="gray", lw=0.8, ls=":")
    ax.annotate("CLT (Baseline D): compute limitation", (0.5, 0.04),
                xycoords="axes fraction", fontsize=8, color="gray")
    ax.set_xscale("log")
    ax.set_xlabel("held-out input density (L0 fraction)")
    ax.set_ylabel("held-out Δ reconstruction R²")
    ax.set_title("Phase 4 Step 1 — Δ reconstruction vs sparsity (Pythia-410m)")
    ax.legend(fontsize=8, loc="center right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
