#!/usr/bin/env python3
"""Generate the curated headline paper-figure set as clean SVG (Phase-4 D2).

Reads only saved ``tables/`` across run dirs — no model, no GPU — and writes
``figures/figNN_*.svg`` at the repo root. Unlike ``scripts/replot.py`` (which
re-emits each run's own diagnostics in place), this driver *combines* runs into
the figures the paper actually uses, including the cross-store views where the
newer data changes the picture: factual-vs-generic tautology (T1), and the
CLT-parsimony-vs-depth contrast (E3 24L tie vs E8 12L gap).

Usage:  python scripts/make_paper_figures.py [--out figures]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from layer_dynamics import figures
from layer_dynamics.figures import plt

ROOT = Path(__file__).resolve().parent.parent


def find_run(pattern: str, *, general: bool = False) -> Path | None:
    cands = [p for p in ROOT.glob(f"runs/*{pattern}*") if p.is_dir()
             and (("general" in p.name) == general)]
    return sorted(cands, key=lambda p: p.name)[-1] if cands else None


def tbl_csv(pattern: str, name: str, *, general: bool = False, **kw) -> pd.DataFrame | None:
    r = find_run(pattern, general=general)
    p = r / "tables" / name if r else None
    return pd.read_csv(p, **kw) if p and p.exists() else None


def tbl_json(pattern: str, name: str, *, general: bool = False) -> dict | None:
    r = find_run(pattern, general=general)
    p = r / "tables" / name if r else None
    return json.loads(p.read_text()) if p and p.exists() else None


# --- builders: each returns a Figure or None (source missing) ---------------

def h1_predictability():
    df = tbl_csv("ph1-step3-410md", "ridge_r2.csv")
    return figures.ridge_by_layer(df) if df is not None else None


def h1_ar():
    df = tbl_csv("ph1-step5-410md", "ar_r2.csv")
    return figures.ar_r2_vs_k(df) if df is not None else None


def h1_granger():
    df = tbl_csv("ph1-step5-410md", "granger_real.csv")
    if df is None:
        return None
    return figures.granger_heatmap(
        df, title="Granger nested-F (−log10 p): depth structure to lag 20")


def h2_cross_band():
    df = tbl_csv("ph2-step1-410md", "cross_band_r2.csv")
    if df is None:
        return None
    return figures.cross_band(df, title="H2 fails: every cross-band R² negative")


def kstar():
    df = tbl_csv("ph2-step2-410md", "per_depth_kstar.csv")
    return figures.kstar_vs_depth(df, granger_reach=20) if df is not None else None


def depth_scaling():
    js = tbl_json("crossmodel", "depth_scaling_gaps.json")
    return figures.depth_scaling(js if isinstance(js, list) else js["rows"]) if js else None


def baseline_pareto():
    js = tbl_json("baseline-pareto", "baseline_comparison.json")
    return figures.baseline_pareto(js) if js else None


def clt_parsimony_depth():
    """Combined E3 (410m/24L) + E8 (Qwen/12L): the parsimony tie is depth-graded."""
    e3 = tbl_json("e3-clt-410md", "baseline_clt.json")
    e8 = tbl_json("e8-clt-qwen", "baseline_clt.json")
    if not e3 or not e8:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, js, name in ((axes[0], e3, "Pythia-410m (24L): CLT≈SSM tie"),
                         (axes[1], e8, "Qwen2-0.5B (12L): CLT still buys +0.02")):
        for row in js["rows"]:
            p = row.get("n_params", row.get("n_params_M", 0) * 1e6) / 1e6
            ax.scatter(p, row["test_r2"], s=80)
            ax.annotate(row["model"], (p, row["test_r2"]), textcoords="offset points",
                        xytext=(6, 5), fontsize=8)
        ax.set_xscale("log"); ax.grid(alpha=0.3)
        ax.set_xlabel("effective parameters (millions, log)"); ax.set_title(name)
    axes[0].set_ylabel("held-out Δ reconstruction R²")
    fig.suptitle("Residual-write CLT vs SSM parsimony is depth-graded "
                 "(E3 tie at 24L, E8 gap at 12L)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def distributed_code():
    js = tbl_json("e1-distributed", "distributed_control.json")
    return figures.distributed_control(
        js, title="Causal null: group effect is magnitude-driven (E1)") if js else None


def crossclass():
    js = tbl_json("crossclass", "crossclass.json")
    return figures.crossclass(js) if js else None


def seed_stability():
    js = tbl_json("gauge", "gauge_stability.json")
    if not js or "summary" not in js:
        return None
    pf = tbl_csv("gauge", "gauge_perfeature_cos.csv")  # persisted by reproduce_full_figs.py
    if pf is not None:
        return figures.gauge_full(js["summary"], pf["raw_D_cos"].to_numpy(),
                                  pf["impulse_cos"].to_numpy())
    return figures.gauge_bars(js["summary"])


def aL_spectrum():
    spec = tbl_csv("aL-operator", "spectrum_by_depth.csv")
    decay = tbl_csv("aL-operator", "state_path_decay.csv")
    return figures.aL_spectrum(spec, decay) if spec is not None else None


def source_prefix():
    js = tbl_json("e5-prefix", "embed_only.json")
    return figures.source_prefix(js) if js and js.get("rows") else None


def tautology_factual_vs_generic():
    """T1 persistence-stratified AR: factual + generic overlaid (the 'additional data' view)."""
    dfs = {}
    fac = tbl_csv("pressure-410md", "t1_persistence_sweep.csv")
    gen = tbl_csv("pressure-410md", "t1_persistence_sweep.csv", general=True)
    if fac is not None:
        dfs["factual"] = fac
    if gen is not None:
        dfs["generic"] = gen
    return figures.tautology_t1(dfs) if dfs else None


def t5_floor():
    df = tbl_csv("t5-taut-floor", "t5_taut_floor.csv")
    if df is None:
        return None
    stats = tbl_json("t5-taut-floor", "t5_stats.json") or {}
    return figures.t5_floor(df, r2_full=stats.get("r2_full"), label="factual")


PAPER_FIGURES = [
    ("fig01_h1_predictability", h1_predictability),
    ("fig02_h1_ar_r2_vs_k", h1_ar),
    ("fig03_h1_granger_real", h1_granger),
    ("fig04_h2_cross_band", h2_cross_band),
    ("fig05_kstar_vs_depth", kstar),
    ("fig06_depth_scaling", depth_scaling),
    ("fig07_baseline_pareto", baseline_pareto),
    ("fig08_clt_parsimony_depth", clt_parsimony_depth),
    ("fig09_distributed_code", distributed_code),
    ("fig10_crossclass", crossclass),
    ("fig11_seed_stability", seed_stability),
    ("fig12_aL_spectrum", aL_spectrum),
    ("fig13_source_prefix", source_prefix),
    ("fig14_tautology_factual_vs_generic", tautology_factual_vs_generic),
    ("fig15_t5_floor", t5_floor),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "figures")
    args = ap.parse_args()
    figures.apply_style()

    written, skipped = 0, []
    for stem, builder in PAPER_FIGURES:
        try:
            fig = builder()
        except Exception as e:  # noqa: BLE001
            skipped.append(f"{stem}: {type(e).__name__}: {e}")
            continue
        if fig is None:
            skipped.append(f"{stem}: source tables missing")
            continue
        path = figures.save_svg(fig, args.out / f"{stem}.svg")
        written += 1
        print(f"  wrote {path.relative_to(ROOT)}")
    print(f"\n{written}/{len(PAPER_FIGURES)} paper figures written to "
          f"{args.out.relative_to(ROOT)}/")
    for s in skipped:
        print(f"  ! skipped {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
