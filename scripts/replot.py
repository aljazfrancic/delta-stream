#!/usr/bin/env python3
"""Re-emit every table-backed figure as SVG, in place, from saved run intermediates.

No model, no GPU: reads each run's ``tables/`` (CSV/JSON) and redraws via
``layer_dynamics.figures``. Figures whose inputs were never persisted as tables
(the ``reconstruction`` eyeballs and the A_ℓ ``eig_plane``/``modulus_hist``, which
need the SSM checkpoint) are skipped and reported — regenerate those with
``scripts/phase4_checkpoint.py`` / ``scripts/phase4_aL_operator.py``.

Usage:
    python scripts/replot.py                 # all runs/
    python scripts/replot.py --run runs/2026-06-25-ph3-gate3-410md
    python scripts/replot.py --dry-run       # list what would be written
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from layer_dynamics import figures

ROOT = Path(__file__).resolve().parent.parent


def _csv(td: Path, name: str, **kw) -> pd.DataFrame | None:
    p = td / name
    return pd.read_csv(p, **kw) if p.exists() else None


def _json(td: Path, name: str) -> dict | None:
    p = td / name
    return json.loads(p.read_text()) if p.exists() else None


# Each emitter: tables-dir -> {output_stem: Figure}. Missing inputs -> {}.

def em_ridge(td):
    df = _csv(td, "ridge_r2.csv")
    if df is None:
        return {}
    return {"r2_by_layer": figures.ridge_by_layer(df), "r2_heatmap": figures.ridge_heatmap(df)}


def em_cca(td):
    df = _csv(td, "cca_correlations.csv")
    if df is None:
        return {}
    out = {"cca_by_layer": figures.cca_by_layer(df), "cca_heatmap": figures.cca_heatmap(df)}
    if {"ridge_best_r2", "mean_te_top3"} <= set(df.columns):
        out["cca_vs_r2"] = figures.cca_vs_r2(df)
    return out


def em_ar(td):
    df = _csv(td, "ar_r2.csv")
    return {"ar_r2_vs_k": figures.ar_r2_vs_k(df)} if df is not None else {}


def em_granger(td):
    real, null = _csv(td, "granger_real.csv"), _csv(td, "granger_null.csv")
    out = {}
    vmax = None
    if real is not None:
        vmax = float(real["neg_log10_p"].max())
    if null is not None:
        vmax = max(vmax or 0.0, float(null["neg_log10_p"].max()))
    for tag, grid in (("real", real), ("null", null)):
        if grid is not None:
            out[f"granger_{tag}"] = figures.granger_heatmap(
                grid, title=f"Granger nested-F (−log10 p), {tag}", vmax=vmax)
    return out


def em_stationarity(td):
    out = {}
    cb = _csv(td, "cross_band_r2.csv")
    if cb is not None:
        out["cross_band_r2"] = figures.cross_band(cb)
    pl = _csv(td, "per_lag_frobenius.csv")
    if pl is not None:
        out["per_lag_frobenius"] = figures.per_lag_frobenius(pl)
    bs = _csv(td, "bootstrap.csv")
    samp = _csv(td, "bootstrap_samples.csv")
    if samp is not None:  # full violin when the within-band bootstrap samples were persisted
        between = dict(zip(bs["pair"], bs["between"], strict=False)) if bs is not None else {}
        out["bootstrap_separation"] = figures.bootstrap_violin(
            {nm: samp[nm].to_numpy() for nm in samp.columns}, between)
    elif bs is not None:
        out["bootstrap_separation"] = figures.bootstrap_bars(bs)
    st = _json(td, "step1_stats.json")
    if st and "between_angles" in st:
        out["principal_angles"] = figures.principal_angles(st["between_angles"])
    return out


def em_kstar(td):
    out = {}
    cdf = _csv(td, "per_depth_r2_curve.csv")
    if cdf is not None:
        out["r2_vs_k_per_depth"] = figures.kstar_r2_vs_k(cdf)
        out["r2_depth_k_heatmap"] = figures.kstar_r2_heatmap(cdf)
    kdf = _csv(td, "per_depth_kstar.csv")
    if kdf is not None:
        out["kstar_vs_depth"] = figures.kstar_vs_depth(kdf)
    return out


def em_tokenpos(td):
    out = {}
    for tag in ("multi", "single"):
        conf = _csv(td, f"confusion_{tag}.csv", index_col=0)
        if conf is not None:
            out[f"confusion_{tag}"] = figures.confusion(
                conf, title=f"Cross-position R² confusion ({tag} subset)")
            if tag == "multi":
                out["within_vs_cross"] = figures.within_vs_cross(conf)
    return out


def em_pareto(td):
    df = _csv(td, "pareto.csv")
    return {"pareto": figures.pareto(df)} if df is not None else {}


def em_altmin(td):
    df = _csv(td, "altmin_history.csv")
    return {"altmin_curves": figures.altmin_curves(df)} if df is not None else {}


def em_baseline_pareto(td):
    js = _json(td, "baseline_comparison.json")
    if js is None or "ssm" not in js:
        return {}
    return {"baseline_pareto": figures.baseline_pareto(js)}


def em_clt(td):
    js = _json(td, "baseline_clt.json")
    if js is None or "rows" not in js:
        return {}
    return {"clt_vs_ssm": figures.clt_vs_ssm(js["rows"])}


def em_distributed(td):
    js = _json(td, "distributed_control.json")
    if js is None or "ks" not in js:
        return {}
    return {"distributed_control": figures.distributed_control(js)}


def em_gauge(td):
    js = _json(td, "gauge_stability.json")
    if js is None or "summary" not in js:
        return {}
    pf = _csv(td, "gauge_perfeature_cos.csv")
    if pf is not None:  # full 2-panel when the per-feature cosine arrays were persisted
        return {"gauge_stability": figures.gauge_full(
            js["summary"], pf["raw_D_cos"].to_numpy(), pf["impulse_cos"].to_numpy())}
    return {"gauge_stability": figures.gauge_bars(js["summary"])}


def em_aL(td):
    out = {}
    spec = _csv(td, "spectrum_by_depth.csv")
    if spec is not None:
        out["rho_by_depth"] = figures.rho_by_depth(spec)
    decay = _csv(td, "state_path_decay.csv")
    if decay is not None:
        out["state_decay"] = figures.state_decay(decay)
    return out


def em_source_prefix(td):
    js = _json(td, "embed_only.json")
    if js is None or not js.get("rows") or "cutoff" not in js["rows"][0]:
        return {}
    return {"source_prefix_sweep": figures.source_prefix(js)}


def em_pressure(td):
    t1 = _csv(td, "t1_persistence_sweep.csv")
    t2 = _csv(td, "t2_surrogate.csv")
    t3 = _csv(td, "t3_granger_by_lag.csv")
    if t1 is None or t2 is None or t3 is None:
        return {}
    return {"t2_t3_summary": figures.pressure_t2_t3(t1, t2.iloc[0].to_dict(), t3.iloc[0].to_dict())}


def em_t5(td):
    df = _csv(td, "t5_taut_floor.csv")
    if df is None:
        return {}
    stats = _json(td, "t5_stats.json") or {}
    return {"t5_floor": figures.t5_floor(df, r2_full=stats.get("r2_full"))}


def em_crossclass(td):
    js = _json(td, "crossclass.json")
    if js is None or "matrix" not in js:
        return {}
    return {"crossclass": figures.crossclass(js)}


def em_depth_scaling(td):
    js = _json(td, "depth_scaling_gaps.json")
    if js is None:
        return {}
    rows = js if isinstance(js, list) else js.get("rows")
    return {"depth_scaling": figures.depth_scaling(rows)} if rows else {}


def em_state_probe(td):
    df = _csv(td, "state_probe.csv")
    js = _json(td, "state_probe.json")
    if df is None or js is None:
        return {}
    return {"state_probe": figures.state_probe(
        {"answer-identity probe (d=32 arms)": df},
        majority=js["majority_acc"]["object"])}


EMITTERS = [
    em_ridge, em_cca, em_ar, em_granger, em_stationarity, em_kstar, em_tokenpos,
    em_pareto, em_altmin, em_baseline_pareto, em_clt, em_distributed, em_gauge,
    em_aL, em_source_prefix, em_pressure, em_t5, em_crossclass, em_depth_scaling,
    em_state_probe,
]


def process_dir(rundir: Path, dry: bool) -> tuple[int, list[str]]:
    td = rundir / "tables"
    if not td.is_dir():
        return 0, []
    written: list[str] = []
    errors: list[str] = []
    for em in EMITTERS:
        try:
            figs = em(td)
        except Exception as e:  # noqa: BLE001 — one bad table must not halt the walk
            errors.append(f"{rundir.name}:{em.__name__}: {type(e).__name__}: {e}")
            continue
        for stem, fig in figs.items():
            out = rundir / "figures" / f"{stem}.svg"
            if dry:
                figures.plt.close(fig)
            else:
                figures.save_svg(fig, out)
            written.append(str(out.relative_to(ROOT)))
    return len(written), errors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, help="single run dir (default: all of runs/)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    figures.apply_style()
    if args.run:
        dirs = [args.run]
    else:
        dirs = sorted({p.parent for p in ROOT.glob("runs/**/tables")})

    total, all_errors = 0, []
    for d in dirs:
        n, errs = process_dir(d, args.dry_run)
        total += n
        all_errors += errs
        if n:
            print(f"  {'(dry) ' if args.dry_run else ''}{n:2d}  {d.relative_to(ROOT)}")
    print(f"\n{'would write' if args.dry_run else 'wrote'} {total} SVG figures "
          f"across {len(dirs)} run dirs")
    if all_errors:
        print(f"\n{len(all_errors)} emitter error(s):")
        for e in all_errors:
            print(f"  ! {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
