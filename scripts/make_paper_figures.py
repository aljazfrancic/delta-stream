#!/usr/bin/env python3
"""Generate the curated headline paper-figure set as clean SVG (Phase-4 D2).

Reads only saved ``tables/`` across run dirs — no model, no GPU — and writes
``figures/figNN_*.svg`` at the repo root. Unlike ``scripts/replot.py`` (which
re-emits each run's own diagnostics in place), this driver *combines* runs into
the figures the paper actually uses, including the cross-store views where the
newer data changes the picture: factual-vs-generic tautology (T1), the
three-axis CLT comparison (R1/R2/R3), and the two-dictionary R7 state probe.

Usage:  python scripts/make_paper_figures.py [--out figures]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from layer_dynamics import figures

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
        df, title=r"Granger nested-F ($-\log_{10}\, p$): depth structure to lag 20")


def h2_cross_band():
    df = tbl_csv("ph2-step1-410md", "cross_band_r2.csv")
    if df is None:
        return None
    return figures.cross_band(df, title="H2 fails: every cross-band $R^2$ negative")


def kstar():
    df = tbl_csv("ph2-step2-410md", "per_depth_kstar.csv")
    if df is None:
        return None
    # The one k*>1 bar: Δ_17's lag-1 R² collapses to 0.64 (≈0.86 elsewhere), so the
    # 95%-of-best rule needs a second lag. Store-specific: ℓ=17 is the 410m
    # answer-promotion depth (R7 probe jumps 0.57→0.76 there; the source-prefix
    # sweep dips there too). Cross-figure pointers live in the paper prose, not
    # in the panels (figure numbers differ between the repo and the paper).
    note = (17, "$\\ell$=17: lag-1 $R^2$ dips to 0.64 —\nthe answer-promotion write"
                "\n(the probe accuracy jumps here)")
    return figures.kstar_vs_depth(df, granger_reach=20, spike_note=note)


def depth_scaling():
    js = tbl_json("crossmodel", "depth_scaling_gaps.json")
    return figures.depth_scaling(js if isinstance(js, list) else js["rows"]) if js else None


def baseline_pareto():
    js = tbl_json("baseline-pareto", "baseline_comparison.json")
    return figures.baseline_pareto(js) if js else None


def clt_three_axis():
    """R1/R2/R3 reframe (2026-07-03) — retires the E3/E8 "depth-graded tie" story.

    The E3 tie was an edge-limited reg-grid artifact; with the grid fixed the CLT
    leads at matched per-decoder rank everywhere, but the SSM Pareto-dominates at
    matched TOTAL params and wins outright below ~1k training examples.
    """
    def run_json(suffix):
        """baseline_clt.json of the newest run whose dir name ENDS with suffix
        (tbl_json's substring glob would let ...rank32 match ...rank32-lamprobe)."""
        hits = sorted(p for p in ROOT.glob(f"runs/*{suffix}") if p.is_dir())
        f = hits[-1] / "tables" / "baseline_clt.json" if hits else None
        return json.loads(f.read_text()) if f and f.exists() else None

    def best_rows(*suffixes):
        """Best (max test R²) row per model family across several runs' tables."""
        best: dict[str, dict] = {}
        for suf in suffixes:
            js = run_json(suf)
            if not js:
                continue
            for row in js["rows"]:
                cur = best.get(row["model"])
                if cur is None or row["test_r2"] > cur["test_r2"]:
                    best[row["model"]] = row
        return best

    # Panel A: param-budget Pareto (410m; rank-32 = best across base grid + λ probe;
    # rank-64/128 per_layer points from R9, the 4th-pass param-matched control —
    # per_layer@r64 ≈ 8.4M ≈ ssm@r32, per_layer@r128 ≈ 16.8M ≈ ssm@r64).
    by_rank = {16: best_rows("r2-clt-410md-rank16"),
               32: best_rows("r2-clt-410md-rank32", "r2-clt-410md-rank32-lamprobe"),
               64: best_rows("r2-clt-410md-rank64", "r9-perlayer-rank64"),
               128: best_rows("r9-perlayer-rank128")}
    budget: dict[str, list] = {}
    for rk, fam_rows in by_rank.items():
        for fam, row in fam_rows.items():
            budget.setdefault(fam, []).append((row["n_params"] / 1e6, row["test_r2"], rk))

    # Panel B: SSM−CLT per model at rank 32 (paired CIs; Qwen = pre-review grid, open)
    model_gaps = []
    for label, suf in (("Pythia-70m (6L)", "r1-clt-p70"),
                       ("GPT-2-small (12L)", "r1-clt-gpt2"),
                       ("Pythia-160m (12L)", "r1-clt-p160"),
                       ("Pythia-410m (24L)", "r2-clt-410md-rank32-lamprobe")):
        js = run_json(suf)
        ci = (js or {}).get("paired_bootstrap_cis", {}).get("ssm_minus_clt_full")
        if ci:
            model_gaps.append((label, ci["gap"], ci["ci_lo"], ci["ci_hi"], True))
    e8 = run_json("e8-clt-qwen")
    if e8 and "ssm_vs_clt_test_r2_gap" in e8:
        g = e8["ssm_vs_clt_test_r2_gap"]
        model_gaps.append(("Qwen2-0.5B (12L)\n(coarse $\\lambda$ grid, no CI)", g, g, g, False))

    # Panel C: sample-efficiency (410m rank 32, consistent λ≤1 grid across N)
    n_curve = []
    for n, suf in ((300, "r3-clt-410md-n300"), (600, "r3-clt-410md-n600"),
                   (1244, "r2-clt-410md-rank32")):
        js = run_json(suf)
        ci = (js or {}).get("paired_bootstrap_cis", {}).get("ssm_minus_clt_full")
        if ci:
            n_curve.append((n, ci["gap"], ci["ci_lo"], ci["ci_hi"]))

    if not (budget and model_gaps and n_curve):
        return None
    return figures.clt_three_axis(budget, model_gaps, n_curve)


def distributed_code():
    js = tbl_json("e1-distributed", "distributed_control.json")
    return figures.distributed_control(
        js, title="Causal null: group effect is magnitude-driven") if js else None


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
    return figures.source_prefix(js, notes=True) if js and js.get("rows") else None


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


def state_probe():
    """R7 (2026-07-04): the SSM state is functionally informative, both dictionaries."""
    dfs, notes = {}, {}
    majority = None
    for suffix, title in (("r7-state-probe-mlsae", "mlsae (TopK) sources"),
                          ("r7-state-probe-l1", "trained-L1 sources")):
        df = tbl_csv(suffix, "state_probe.csv")
        js = tbl_json(suffix, "state_probe.json")
        if df is None or js is None:
            continue
        dfs[title] = df
        majority = js["majority_acc"]["object"]
        ci = js["band_contrasts"].get("object/state−code_pca/mid")
        if ci:
            notes[title] = (f"mid-band state−code: {ci['diff']:+.3f} "
                            f"[{ci['ci_lo']:+.3f}, {ci['ci_hi']:+.3f}]")
    if not dfs:
        return None
    return figures.state_probe(dfs, majority=majority, mid_notes=notes)


PAPER_FIGURES = [
    ("fig01_h1_predictability", h1_predictability),
    ("fig02_h1_ar_r2_vs_k", h1_ar),
    ("fig03_h1_granger_real", h1_granger),
    ("fig04_h2_cross_band", h2_cross_band),
    ("fig05_kstar_vs_depth", kstar),
    ("fig06_depth_scaling", depth_scaling),
    ("fig07_baseline_pareto", baseline_pareto),
    ("fig08_clt_three_axis", clt_three_axis),
    ("fig09_distributed_code", distributed_code),
    ("fig10_crossclass", crossclass),
    ("fig11_seed_stability", seed_stability),
    ("fig12_aL_spectrum", aL_spectrum),
    ("fig13_source_prefix", source_prefix),
    ("fig14_tautology_factual_vs_generic", tautology_factual_vs_generic),
    ("fig15_t5_floor", t5_floor),
    ("fig16_state_probe", state_probe),
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
        rel = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
        print(f"  wrote {rel}")
    print(f"\n{written}/{len(PAPER_FIGURES)} paper figures written to "
          f"{args.out.relative_to(ROOT)}/")
    for s in skipped:
        print(f"  ! skipped {s}")
    return 0 if not skipped else 1  # D1 reproduce.sh relies on this exit code


if __name__ == "__main__":
    raise SystemExit(main())
