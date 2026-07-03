"""Shared, table-driven figure functions (pure ``data -> matplotlib Figure``).

Every plot the analysis scripts emit is also expressible here from the numbers
already saved under a run's ``tables/`` (CSV/JSON) — no model, no GPU. Two drivers
consume this module:

- ``scripts/replot.py``        — walk run dirs and re-emit each figure as SVG in place.
- ``scripts/make_paper_figures.py`` — the curated headline set, combined across runs.

Design: functions take already-loaded data (a ``DataFrame``, dict, or array) and
return a ``Figure``. IO + SVG writing live in :func:`save_svg`. Styling (SVG with
selectable text, tight layout, no overlaps) is centralised in :func:`apply_style`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_STYLED = False


def apply_style() -> None:
    """Idempotent global style: keep SVG text selectable, sane sizes, tight bbox."""
    global _STYLED
    if _STYLED:
        return
    matplotlib.rcParams.update({
        "svg.fonttype": "none",       # text stays text in the SVG (editable, crisp)
        "svg.hashsalt": "delta-stream",  # stable element IDs -> byte-reproducible SVGs
        "savefig.bbox": "tight",      # crop to content -> no clipped labels/legends
        "figure.max_open_warning": 0,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 110,
    })
    _STYLED = True


def save_svg(fig: plt.Figure, path: str | Path) -> Path:
    """Write ``fig`` as SVG (coercing any ``.png`` suffix) and close it."""
    path = Path(path).with_suffix(".svg")
    path.parent.mkdir(parents=True, exist_ok=True)
    # metadata Date=None drops the render timestamp so re-renders are byte-identical.
    fig.savefig(path, format="svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)
    return path


# --- small shared helpers ---------------------------------------------------

def _rank_cols(df: pd.DataFrame) -> list[str]:
    import re
    return [c for c in df.columns if re.fullmatch(r"r\d+", str(c))]


def _annot_heatmap(ax, mat, *, cmap="viridis", vmin=None, vmax=None, fmt="{:.2f}",
                   annot=True):
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    if annot:
        thr = np.nanmin(mat) + 0.6 * (np.nanmax(mat) - np.nanmin(mat))
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if np.isfinite(v):
                    ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8,
                            color="w" if v < thr else "k")
    return im


# --- Phase 1: predictability + AR + Granger ---------------------------------

def ridge_by_layer(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(df["layer"], df["best_r2"], "-o", ms=3, label="ridge (best rank)")
    if "tuned_lens" in df:
        ax.plot(df["layer"], df["tuned_lens"], "-s", ms=3, color="tab:red",
                label="Tuned-Lens implied (gate baseline)")
    if "zero" in df:
        ax.plot(df["layer"], df["zero"], "--", color="grey", label="zero-pred")
    ax.axhline(0, color="k", lw=0.6, label="mean-pred (R²=0)")
    ax.set_xlabel("layer pair ℓ"); ax.set_ylabel("pooled R² on Δ_ℓ")
    ax.set_title("Depth-local predictability of Δ_ℓ (ridge vs Tuned Lens)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def ridge_heatmap(df: pd.DataFrame) -> plt.Figure:
    cols = _rank_cols(df) + (["full"] if "full" in df else [])
    mat = df[cols].to_numpy().T
    fig, ax = plt.subplots(figsize=(9, 3.2))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", origin="lower", vmin=0)
    ax.set_yticks(range(len(cols)), [c.lstrip("r") for c in cols])
    ax.set_xlabel("layer pair ℓ"); ax.set_ylabel("rank")
    ax.set_title("Δ_ℓ ridge held-out R²(ℓ, rank)")
    fig.colorbar(im, ax=ax, label="pooled R²")
    fig.tight_layout()
    return fig


def cca_by_layer(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(df["layer"], df["te1"], "-o", ms=3, label="held-out c1")
    if "tr1" in df:
        ax.plot(df["layer"], df["tr1"], "--", color="grey", label="in-sample c1")
    if "mean_te_top3" in df:
        ax.plot(df["layer"], df["mean_te_top3"], "-s", ms=3, color="tab:green",
                label="held-out mean(c1..c3)")
    if "null_te1" in df:
        ax.plot(df["layer"], df["null_te1"], "-^", ms=3, color="tab:red",
                label="scrambled-null c1")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("layer pair ℓ"); ax.set_ylabel("canonical correlation")
    ax.set_title("Leading canonical correlation by depth"); ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def cca_heatmap(df: pd.DataFrame) -> plt.Figure:
    import re
    te_cols = [c for c in df.columns if re.fullmatch(r"te\d+", str(c))]
    mat = df[te_cols].to_numpy().T
    fig, ax = plt.subplots(figsize=(9, 3.2))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", origin="lower", vmin=0, vmax=1)
    ax.set_yticks(range(len(te_cols)), [c.replace("te", "c") for c in te_cols])
    ax.set_xlabel("layer pair ℓ"); ax.set_ylabel("canonical component")
    ax.set_title("Held-out canonical correlation between v_ℓ and Δ_ℓ")
    fig.colorbar(im, ax=ax, label="held-out corr")
    fig.tight_layout()
    return fig


def ar_r2_vs_k(df: pd.DataFrame) -> plt.Figure:
    real = df[df["label"] == "real"]; null = df[df["label"] == "null"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for c in _rank_cols(df):
        ax.plot(real["k"], real[c], "-", lw=0.8, alpha=0.5, label=f"real {c}")
    ax.plot(real["k"], real["best_r2"], "-o", color="tab:blue", lw=2, label="real (best rank)")
    ax.plot(null["k"], null["best_r2"], "--s", color="tab:red", lw=1.5, label="scrambled null")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("AR order k"); ax.set_ylabel("held-out pooled R² on Δ_ℓ")
    ax.set_title("AR(k) depth-local predictability of Δ"); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    return fig


def granger_heatmap(df: pd.DataFrame, *, title: str, vmax: float | None = None,
                    thr: float | None = None) -> plt.Figure:
    piv = df.pivot(index="lag", columns="depth", values="neg_log10_p")
    fig, ax = plt.subplots(figsize=(8.5, 3.0))
    im = ax.imshow(piv.to_numpy(), aspect="auto", cmap="magma", origin="lower",
                   vmin=0, vmax=vmax,
                   extent=[piv.columns.min() - 0.5, piv.columns.max() + 0.5,
                           -0.5, len(piv.index) - 0.5])
    ax.set_yticks(range(len(piv.index)), [str(j) for j in piv.index])
    ax.set_xlabel("target depth ℓ"); ax.set_ylabel("lag j")
    ttl = title + (f"  [Bonferroni thr={thr:.1f}]" if thr else "")
    ax.set_title(ttl)
    fig.colorbar(im, ax=ax, label="−log10 p")
    fig.tight_layout()
    return fig


# --- Phase 2: stationarity, effective depth, position ------------------------

def cross_band(df: pd.DataFrame, *, title="Cross-band held-out R²") -> plt.Figure:
    bands = [c.replace("eval_", "") for c in df.columns if c.startswith("eval_")]
    mat = df[[f"eval_{b}" for b in bands]].to_numpy()
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = _annot_heatmap(ax, mat, cmap="viridis", vmax=float(mat.max()), fmt="{:.3f}")
    ax.set_xticks(range(len(bands)), bands); ax.set_yticks(range(len(bands)), bands)
    ax.set_xlabel("evaluated on band"); ax.set_ylabel("kernel fit on band")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="held-out R²")
    fig.tight_layout()
    return fig


def kstar_vs_depth(kdf: pd.DataFrame, *, granger_reach: int | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.bar(kdf["depth"], kdf["kstar"], color="tab:blue", label="reconstruction k*")
    if "max_feasible_k" in kdf:
        ax.plot(kdf["depth"], kdf["max_feasible_k"], "k:", lw=0.8, label="max feasible k")
    if granger_reach is not None:
        ax.axhline(granger_reach, color="tab:red", lw=1.5, ls="--",
                   label=f"Granger reach (lag {granger_reach})")
    ax.set_xlabel("target depth ℓ"); ax.set_ylabel("k")
    ax.set_title("Effective depth k*(ℓ): reconstruction (≈1) vs Granger reach")
    ax.legend()
    fig.tight_layout()
    return fig


def kstar_r2_heatmap(cdf: pd.DataFrame) -> plt.Figure:
    piv = cdf.pivot(index="k", columns="depth", values="r2_real")
    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    im = ax.imshow(piv.to_numpy(), aspect="auto", origin="lower", cmap="magma",
                   extent=[piv.columns.min() - 0.5, piv.columns.max() + 0.5,
                           -0.5, len(piv.index) - 0.5])
    ax.set_yticks(range(len(piv.index)), [str(k) for k in piv.index])
    ax.set_xlabel("target depth ℓ"); ax.set_ylabel("AR order k")
    ax.set_title("Held-out R²(depth, k) — value concentrated at k=1")
    fig.colorbar(im, ax=ax, label="held-out R²")
    fig.tight_layout()
    return fig


def kstar_r2_vs_k(cdf: pd.DataFrame) -> plt.Figure:
    depths = sorted(cdf["depth"].unique())
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, ell in enumerate(depths):
        sub = cdf[cdf["depth"] == ell].sort_values("k")
        ax.plot(sub["k"], sub["r2_real"], "-", lw=0.8, alpha=0.7,
                color=cmap(i / max(1, len(depths) - 1)))
    nm = cdf.groupby("k")["r2_null"].mean()
    ax.plot(nm.index, nm.to_numpy(), "k--", lw=1.5, label="scrambled null (mean)")
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_xlabel("AR order k"); ax.set_ylabel("held-out R² on Δ_ℓ (single depth)")
    ax.set_title("Reconstruction R²(k) per depth — saturates at k≈1; null≈0")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(min(depths), max(depths)))
    fig.colorbar(sm, ax=ax, label="target depth ℓ"); ax.legend()
    fig.tight_layout()
    return fig


def cca_vs_r2(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    sc = ax.scatter(df["ridge_best_r2"], df["mean_te_top3"], c=df["layer"], cmap="plasma")
    for _, r in df.iterrows():
        ax.annotate(int(r["layer"]), (r["ridge_best_r2"], r["mean_te_top3"]),
                    fontsize=6, xytext=(2, 2), textcoords="offset points")
    ax.axvline(0.5, color="grey", lw=0.6, ls=":"); ax.axhline(0.5, color="grey", lw=0.6, ls=":")
    ax.set_xlabel("ridge held-out R² (best rank)")
    ax.set_ylabel("mean held-out canonical corr (c1..c3)")
    ax.set_title("Interpretation key: predictability vs shared subspace")
    fig.colorbar(sc, ax=ax, label="layer ℓ")
    fig.tight_layout()
    return fig


def pressure_t2_t3(t1: pd.DataFrame, t2: dict, t3: dict) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(t1["m_removed"], t1["real_best_r2"], "-o", label="real")
    axes[0].plot(t1["m_removed"], t1["scramble_null_r2"], "--s", label="scramble null")
    axes[0].axhline(0, color="k", lw=0.6)
    axes[0].set_xlabel("# top state directions removed (m)")
    axes[0].set_ylabel("best held-out AR R² on Δ")
    axes[0].set_title("T1 · persistence-stratified AR"); axes[0].legend(fontsize=8)
    if "state_var_removed" in t1:
        sx = axes[0].twiny(); sx.set_xlim(axes[0].get_xlim())
        sx.set_xticks(t1["m_removed"], [f"{v:.2f}" for v in t1["state_var_removed"]],
                      fontsize=7)
        sx.set_xlabel("state variance removed", fontsize=8)
    labels = ["real", "surrogate\n(T2)", "scramble"]
    vals = [t2["real_best_r2"], t2["surrogate_r2"], t2["scramble_r2"]]
    axes[1].bar(labels, vals, color=["tab:blue", "tab:orange", "tab:red"])
    axes[1].axhline(0, color="k", lw=0.6); axes[1].set_ylabel("best held-out AR R² on Δ")
    axes[1].set_title(f"T2 nulls + T3 lag≥2 Granger: {t3['lagge2_sig']}/{t3['lagge2_total']} sig")
    fig.tight_layout()
    return fig


def confusion(df: pd.DataFrame, *, title: str) -> plt.Figure:
    names = list(df.columns)
    mat = df.to_numpy(dtype=float)
    n = len(names)
    fig, ax = plt.subplots(figsize=(0.9 * n + 2.5, 0.9 * n + 2))
    im = _annot_heatmap(ax, mat, cmap="viridis", vmin=min(0.0, float(mat.min())),
                        vmax=float(mat.max()), fmt="{:.2f}")
    ax.set_xticks(range(n), names, rotation=30, ha="right")
    ax.set_yticks(range(n), names)
    ax.set_xlabel("evaluated on"); ax.set_ylabel("kernel fit on")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="held-out R²")
    fig.tight_layout()
    return fig


# --- Phase 3 / 4: decomposition, baselines, causal ---------------------------

def pareto(df: pd.DataFrame, *,
           title="Gate-3 Pareto: Branch B vs per-layer baseline") -> plt.Figure:
    styles = {"ssm": ("o-", "C0", "Branch B SSM (with state)"),
              "baseline": ("s--", "C1", "per-layer dict (no state)")}
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    for vname, (st, col, label) in styles.items():
        pts = df[df["variant"] == vname].sort_values("test_l0")
        if pts.empty:
            continue
        ax.plot(pts["test_l0"], pts["test_r2"], st, color=col, label=label)
        for _, p in pts.iterrows():
            ax.annotate(f"λ={p['lambda_l1']:g}", (p["test_l0"], p["test_r2"]),
                        fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("held-out input density (L0 fraction)")
    ax.set_ylabel("held-out reconstruction R²")
    ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def altmin_curves(df: pd.DataFrame) -> plt.Figure:
    its = df["iter"] if "iter" in df else range(len(df))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    axes[0].plot(its, df["loss"], "o-", color="C0"); axes[0].set_yscale("log")
    axes[0].set_title("loss"); axes[0].set_xlabel("outer iter")
    axes[1].plot(its, df["r2"], "o-", color="C2"); axes[1].axhline(0, ls=":", c="grey")
    axes[1].set_title("train reconstruction R²"); axes[1].set_xlabel("outer iter")
    axes[2].plot(its, df["l0"], "o-", color="C3")
    axes[2].set_title("input density (L0 fraction)"); axes[2].set_xlabel("outer iter")
    fig.tight_layout()
    return fig


def _pareto_frontier(rows: list[dict]) -> list[dict]:
    """Non-dominated subset of a λ-sweep trace (lower L0 or higher R², sorted by L0).

    At extreme λ the held-out inferred L0 *saturates* (more penalty buys no sparsity,
    only destroys R²), so the raw trace folds back on itself — e.g. the λ=30 point is
    strictly worse than λ=10 on both axes. Connecting the raw trace by L0 then draws a
    spurious dip (and can make the two methods' polylines cross inside a regime no one
    would operate at). The frontier keeps only points not dominated by any other.
    """
    pts = sorted(rows, key=lambda r: (r["l0"], -r["r2"]))
    front: list[dict] = []
    for p in pts:
        if not front or p["r2"] > front[-1]["r2"]:
            front.append(p)
    return front


def baseline_pareto(js: dict[str, Any]) -> plt.Figure:
    ssm = sorted(js["ssm"], key=lambda r: r["l0"])
    base = sorted(js["baseline_A_perlayer_dict"], key=lambda r: r["l0"])
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    any_dom = False
    for rows, color, marker in ((ssm, "C0", "o"), (base, "C1", "s")):
        dom = [r for r in rows if r not in _pareto_frontier(rows)]
        if dom:  # dominated λ points: shown, not connected (L0-saturated regime)
            any_dom = True
            ax.scatter([r["l0"] for r in dom], [r["r2"] for r in dom], marker=marker,
                       facecolors="none", edgecolors=color, s=36, alpha=0.6, zorder=2)
    if any_dom:  # one neutral legend entry for both hollow-marker sets
        ax.scatter([], [], marker="o", facecolors="none", edgecolors="gray",
                   label="dominated λ point (off frontier)")
    ssm_f, base_f = _pareto_frontier(ssm), _pareto_frontier(base)
    ax.plot([r["l0"] for r in ssm_f], [r["r2"] for r in ssm_f], "o-", color="C0",
            label="Branch B SSM (my method)")
    ax.plot([r["l0"] for r in base_f], [r["r2"] for r in base_f], "s--", color="C1",
            label="Baseline A: per-layer dict on Δ")
    mlsae = js.get("baseline_B_mlsae_implied_delta")
    if isinstance(mlsae, dict) and "l0" in mlsae:
        ax.scatter([mlsae["l0"]], [mlsae["r2"]], marker="*", s=160, color="C3", zorder=5,
                   label="Baseline B: MLSAE implied Δ")
    ax.axhline(0.0, color="gray", lw=0.8, ls=":")
    # Baseline D (CLT) has numbers since E3/R2, but under the FROZEN-SOURCE protocol
    # (no LASSO input inference) — a different operating point than this plot's axes,
    # so it gets its own figure rather than a curve here.
    ax.annotate("CLT (Baseline D): see CLT-vs-SSM figure (frozen-source protocol)",
                (0.97, 0.04), xycoords="axes fraction", ha="right",
                fontsize=8, color="gray")
    ax.set_xscale("log")
    ax.set_xlabel("held-out input density (L0 fraction)")
    ax.set_ylabel("held-out Δ reconstruction R²")
    ax.set_title("Phase 4 Step 1 — Δ reconstruction vs sparsity")
    ax.legend(loc="center right"); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def clt_vs_ssm(rows: list[dict], *, title="Residual-write CLT vs state-space SSM") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for row in rows:
        p = row.get("n_params", row.get("n_params_M", 0) * 1e6) / 1e6
        ax.scatter(p, row["test_r2"], s=80)
        ax.annotate(f"{row['model']}\n({p:.1f}M, train {row['train_r2']:.3f})",
                    (p, row["test_r2"]), textcoords="offset points", xytext=(6, 6),
                    fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("effective parameters (millions, log)")
    ax.set_ylabel("held-out Δ reconstruction R²")
    ax.set_title(title); ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def distributed_control(js: dict[str, Any], *, title="Distributed-code control") -> plt.Figure:
    ks = np.asarray(js["ks"], dtype=float)
    fig, ax = plt.subplots(figsize=(6.0, 4.4))

    def _band(y, sd, style, color, label, ms=5):
        y = np.asarray(y, dtype=float)
        ax.plot(ks, y, style, color=color, ms=ms, label=label)
        if sd is not None:
            sd = np.asarray(sd, dtype=float)
            ax.fill_between(ks, y - sd, y + sd, color=color, alpha=0.15)

    _band(js["top_drop_mean"], js.get("top_drop_sd"), "-o", "C3", "top-k co-firing")
    _band(js["random_drop_mean"], js.get("random_drop_sd"), "-s", "C0",
          "matched-random (count)")
    if "normrand_iso_drop_mean" in js:
        _band(js["normrand_iso_drop_mean"], js.get("normrand_iso_drop_sd"), "-D", "C1",
              "norm-matched random dir (isotropic)", ms=4)
    if "normrand_shaped_drop_mean" in js:
        _band(js["normrand_shaped_drop_mean"], js.get("normrand_shaped_drop_sd"), "-v", "C4",
              "norm-matched random dir (shaped)", ms=4)
    if "mass_match_k" in js and "massmatched_drop_mean" in js:
        ax.plot([js["mass_match_k"]], [js["massmatched_drop_mean"]], "^", color="C2",
                ms=10, ls="", label=f"matched-random (mass, ≈top-{js['mass_match_k']})")
    ax.axhline(0.20, ls="--", color="0.4", lw=1, label="0.20 threshold")
    ax.set_xscale("log")
    ax.set_xlabel("group size k (features ablated per example)")
    ax.set_ylabel("mean drop in answer prob")
    ax.set_title(title); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def _gauge_bar_ax(ax, summary: dict[str, float]) -> None:
    labels = ["raw-D\nper-dir", "Δ-contrib\nper-dir", "raw-D\nsubspace",
              "Δ-contrib\nsubspace", "A-eig\nspectrum"]
    keys = ["raw_D_match@0.7", "impulse_match@0.7", "raw_D_subspace_overlap_mean",
            "impulse_subspace_overlap_mean", "eig_spectrum_mean_cos"]
    vals = [summary[k] for k in keys]
    colors = ["tab:red", "tab:blue", "tab:red", "tab:blue", "tab:green"]
    ax.bar(labels, vals, color=colors, alpha=0.8)
    ax.axhline(0.7, color="k", ls=":", lw=1, label="0.7 bar")
    ax.set_ylim(0, 1.05); ax.set_ylabel("stability (match@0.7 / overlap / cos)")
    ax.set_title("System-level invariants are seed-stable")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    ax.legend()


def gauge_bars(summary: dict[str, float]) -> plt.Figure:
    """Summary-only (1-panel) — used when per-feature cosine arrays aren't on disk."""
    fig, ax = plt.subplots(figsize=(6.5, 4))
    _gauge_bar_ax(ax, summary)
    fig.tight_layout()
    return fig


def gauge_full(summary: dict[str, float], raw_cos: np.ndarray,
               imp_cos: np.ndarray) -> plt.Figure:
    """Full 2-panel: per-feature cosine histogram + the system-invariant bars."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    ax, bins = axes[0], np.linspace(-1, 1, 41)
    ax.hist(raw_cos, bins=bins, alpha=0.6, color="tab:red",
            label=f"raw D[:,j]  (match@0.7 {summary['raw_D_match@0.7']:.2f})")
    ax.hist(imp_cos, bins=bins, alpha=0.6, color="tab:blue",
            label=f"gauge-inv Δ-contrib  (match@0.7 {summary['impulse_match@0.7']:.2f})")
    ax.axvline(0.7, color="k", ls=":", lw=1, label="0.7 threshold")
    ax.set_xlabel("cross-seed cosine"); ax.set_ylabel("# features")
    ax.set_title("Per-feature seed stability:\nraw decoder column vs gauge-invariant contribution")
    ax.legend(fontsize=8)
    _gauge_bar_ax(axes[1], summary)
    fig.tight_layout()
    return fig


def aL_spectrum(spec_df: pd.DataFrame, decay_df: pd.DataFrame | None = None) -> plt.Figure:
    n = 1 if decay_df is None else 2
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 3.8), squeeze=False)
    ax = axes[0][0]
    ax.plot(spec_df["layer"], spec_df["rho"], "-o", color="tab:red", label="ρ(A_ℓ)")
    ax.axhline(1.0, color="k", lw=0.8, ls="--", label="unit circle (ρ=1)")
    ax.set_xlabel("layer ℓ"); ax.set_ylabel("spectral radius ρ(A_ℓ)")
    ax.set_title("A_ℓ contractive at every depth"); ax.legend()
    if decay_df is not None:
        ax2 = axes[0][1]
        ax2.plot(decay_df["gap"], decay_df["mean_fro"], "-o", color="tab:purple")
        ax2.set_xlabel("depth gap (layers)"); ax2.set_ylabel("‖state-path kernel‖_F")
        ax2.set_title("State-path halves per layer → k*≈1")
    fig.tight_layout()
    return fig


def source_prefix(js: dict[str, Any]) -> plt.Figure:
    rows = js["rows"]
    cut = [r["cutoff"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(cut, [r["test_r2"] for r in rows], "-o", color="tab:blue")
    ax1.axhline(0, color="k", lw=0.6)
    ax1.set_xlabel("source-prefix cutoff ℓ_max (sources kept for ℓ≤ℓ_max)")
    ax1.set_ylabel("held-out Δ reconstruction R²")
    ax1.set_title("Reconstruction builds up as per-layer sources are added")
    ax1.grid(alpha=0.3)
    # right panel: per-layer R² at the endpoints (full sources vs embedding-only).
    tpl = js.get("test_per_layer_r2_by_cutoff") or {}
    if tpl:
        keys = sorted(tpl, key=int)
        full_pl, embed_pl = tpl[keys[-1]], tpl[keys[0]]
        ax2.plot(range(len(full_pl)), full_pl, "-o", label="full (all sources)")
        ax2.plot(range(len(embed_pl)), embed_pl, "-s", label="embedding-only (ℓ_max=0)")
        ax2.axhline(0, color="k", lw=0.6)
        ax2.set_xlabel("layer ℓ"); ax2.set_ylabel("held-out Δ_ℓ R²")
        ax2.set_title("Per-layer R²: endpoints"); ax2.legend()
    fig.tight_layout()
    return fig


def tautology_t1(dfs: dict[str, pd.DataFrame]) -> plt.Figure:
    """T1 persistence-stratified AR R² vs #state-dirs removed; overlay any stores."""
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    colors = {"factual": "tab:blue", "generic": "tab:green"}
    for name, df in dfs.items():
        c = colors.get(name, None)
        ax.plot(df["m_removed"], df["real_best_r2"], "-o", color=c, label=f"{name}: real")
        ax.plot(df["m_removed"], df["scramble_null_r2"], "--", color=c, alpha=0.6,
                label=f"{name}: scramble null")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("# top persistent state directions removed (m)")
    ax.set_ylabel("best held-out AR R² on Δ")
    ax.set_title("T1 · predictability survives stripping state variance (not the tautology)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def t5_floor(df: pd.DataFrame, *, r2_full: float | None = None, label="") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(df["m"], df["frac_par"], "-o", color="tab:red", label="frac_par (tautology floor)")
    ax.plot(df["m"], df["frac_perp"], "-s", color="tab:blue", label="frac_perp (genuine)")
    if r2_full is not None:
        ax.axhline(r2_full, color="k", lw=0.8, ls="--", label=f"R²_full = {r2_full:.3f}")
    ax.set_xlabel("# persistent state directions m")
    ax.set_ylabel("share of total Δ variance (additive)")
    ax.set_title(f"T5 · tautology floor vs genuine remainder{(' — ' + label) if label else ''}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def crossclass(js: dict[str, Any]) -> plt.Figure:
    classes = js["classes"]; m = js["matrix"]
    mat = np.array([[m[a][b] for b in classes] for a in classes])
    fig, ax = plt.subplots(figsize=(0.95 * len(classes) + 2.6, 0.95 * len(classes) + 2))
    im = _annot_heatmap(ax, mat, cmap="viridis", vmin=min(0.0, float(mat.min())),
                        vmax=float(mat.max()), fmt="{:.2f}")
    ax.set_xticks(range(len(classes)), classes, rotation=25, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("evaluated on class"); ax.set_ylabel("kernel fit on class")
    ax.set_title("Cross-class reconstruction — circuit-specific")
    fig.colorbar(im, ax=ax, label="held-out R²")
    fig.tight_layout()
    return fig


def rho_by_depth(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(df["layer"], df["rho"], "-o", color="tab:red", label="ρ(A_ℓ)")
    if "mean_mod" in df:
        ax.plot(df["layer"], df["mean_mod"], "-s", color="tab:blue", ms=3,
                label="mean |λ|")
    ax.axhline(1.0, color="k", lw=0.8, ls="--", label="unit circle (ρ=1)")
    ax.set_xlabel("layer ℓ"); ax.set_ylabel("spectral radius ρ(A_ℓ)")
    ax.set_title("A_ℓ is contractive at every depth"); ax.legend()
    fig.tight_layout()
    return fig


def state_decay(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(df["gap"], df["mean_fro"], "-o", color="tab:purple")
    ax.set_xlabel("depth gap (layers)"); ax.set_ylabel("‖C(∏A)B‖_F  (state-path kernel)")
    ax.set_title("State-path kernel halves per layer → first-order (k*≈1)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def per_lag_frobenius(df: pd.DataFrame) -> plt.Figure:
    pairs = list(dict.fromkeys(df["pair"]))
    lags = sorted(df["lag"].unique())
    width = 0.8 / max(1, len(pairs))
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, pair in enumerate(pairs):
        sub = df[df["pair"] == pair].sort_values("lag")
        ax.bar([x + (i - (len(pairs) - 1) / 2) * width for x in sub["lag"]],
               sub["rel_frobenius"], width=width, label=pair)
    ax.set_xlabel("lag j"); ax.set_ylabel("relative ‖W_j^A − W_j^B‖_F")
    ax.set_xticks(list(lags))
    ax.set_title("Per-lag kernel-block distance between bands"); ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def principal_angles(angles: dict, *, space: str = "output") -> plt.Figure:
    pairs = list(angles)
    means = [angles[p][space]["mean_deg"] for p in pairs]
    maxes = [angles[p][space]["max_deg"] for p in pairs]
    xs = np.arange(len(pairs))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(xs, means, width=0.5, color="tab:purple", alpha=0.8, label="mean angle")
    ax.plot(xs, maxes, "k_", ms=18, label="max angle")
    ax.set_xticks(xs, pairs, rotation=15, ha="right")
    ax.set_ylabel("principal angle (deg)"); ax.set_ylim(0, 92)
    ax.axhline(90, color="grey", lw=0.6, ls=":")
    ax.set_title("Output-subspace principal angles between bands (≈orthogonal)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def bootstrap_violin(spreads: dict[str, np.ndarray], between: dict[str, float]) -> plt.Figure:
    """Full within-band bootstrap distributions (violin) vs between-band distances."""
    bands = ("early", "middle", "late")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.violinplot([spreads[nm] for nm in bands], showmeans=True)
    ax.set_xticks(range(1, 4), [f"within-{nm}" for nm in bands])
    colors = {"early-middle": "tab:blue", "middle-late": "tab:orange", "early-late": "tab:green"}
    for key, c in colors.items():
        if key in between:
            ax.axhline(between[key], color=c, lw=1.6, ls="--",
                       label=f"between {key}={between[key]:.3f}")
    ax.set_ylabel("relative Frobenius distance")
    ax.set_title("Within-band bootstrap spread vs between-band distance")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def bootstrap_bars(df: pd.DataFrame) -> plt.Figure:
    """Bootstrap separation as bars (p95 within vs between) — summary fallback."""
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(df))
    within = df[["within_a_p95", "within_b_p95"]].max(axis=1)
    ax.bar(x - 0.2, within, 0.4, color="tab:grey", label="within-band p95 spread")
    ax.bar(x + 0.2, df["between"], 0.4, color="tab:purple", label="between-band distance")
    ax.set_xticks(x, df["pair"], rotation=15, ha="right")
    ax.set_ylabel("relative Frobenius distance")
    ax.set_title("Within-band spread vs between-band distance (bands separated)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def within_vs_cross(conf: pd.DataFrame) -> plt.Figure:
    names = list(conf.columns)
    mat = conf.to_numpy(dtype=float)
    within = np.diag(mat)
    cross = [np.mean([mat[i, j] for j in range(len(names)) if j != i])
             for i in range(len(names))]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    x = np.arange(len(names))
    ax.bar(x - 0.2, within, 0.4, label="within-position (diagonal)")
    ax.bar(x + 0.2, cross, 0.4, label="mean cross-position")
    ax.set_xticks(x, names, rotation=20, ha="right"); ax.set_ylabel("held-out R²")
    ax.set_title("Within- vs cross-position reconstruction"); ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def depth_scaling(rows: list[dict]) -> plt.Figure:
    # Prefer the robust λ-knee gap (raw same-λ SSM−baseline difference; review
    # 2026-07-02: gap_max's matched-L0 interpolation is fragile on the saturated
    # Pareto tail — the 160m ±0.015 error bar was that artifact, not seed noise).
    # Falls back to the gap_max stats for pre-review JSONs.
    knee = "knee_mean" in rows[0]
    gm, gs = ("knee_mean", "knee_sd") if knee else ("gap_mean", "gap_sd")
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.6))
    pyth = [r for r in rows if r["family"] == "Pythia"]
    axL.errorbar([r["L"] for r in pyth], [r[gm] for r in pyth],
                 yerr=[r[gs] for r in pyth], fmt="-o", color="tab:blue", capsize=4,
                 label="Pythia / mlsae (mean±sd)", zorder=3)
    xcol = {"GPT-2": "tab:orange", "Qwen": "tab:green"}
    for r in rows:
        if r["family"] != "Pythia":
            axL.scatter([r["L"]], [r[gm]], marker="D", s=70,
                        color=xcol.get(r["family"], "tab:orange"), zorder=4,
                        label=f"{r['family']} / {r['suite']}")
    xmax = max(r["L"] for r in rows)
    for r in rows:
        tag = (f"+{r[gm]:.3f}±{r[gs]:.3f}" if r["n_seeds"] > 1 else f"+{r[gm]:.3f}")
        left = r["L"] == xmax  # rightmost point: label to the left or it clips outside
        axL.annotate(f"{r['label']}\n{tag}", (r["L"], r[gm]),
                     textcoords="offset points", xytext=(-6, -2) if left else (6, -2),
                     ha="right" if left else "left", fontsize=8)
    axL.axhline(0, color="k", lw=0.8)
    axL.set_xlabel("depth L (layers)")
    axL.set_ylabel(
        f"SSM − per-layer-baseline R² gap at λ_L1={rows[0].get('knee_lambda', 1):g}"
        if knee else "max SSM − per-layer-baseline R² gap")
    axL.set_title("Cross-layer propagation gain grows with depth")
    # lower right is the one empty region (Qwen occupies the upper left at (12, ~0.044));
    # anchored a little up so the box clears the y=0 line
    axL.legend(fontsize=8, loc="lower right", bbox_to_anchor=(0.98, 0.10))
    axL.grid(alpha=0.3)

    for sae_type, color in (("TopK", "tab:blue"), ("L1", "tab:orange")):
        pts = [r for r in rows if r.get("sae_type") == sae_type]
        axR.scatter([r["L"] for r in pts], [r["seed"] for r in pts], s=70, color=color,
                    label=f"{sae_type} SAEs")
    for r in rows:
        left = r["L"] == xmax
        axR.annotate(f"{r['label']}\n{r['seed']:.2f}", (r["L"], r["seed"]),
                     textcoords="offset points", xytext=(-6, -4) if left else (6, -4),
                     ha="right" if left else "left", fontsize=8)
    axR.axhline(0.7, color="tab:red", ls="--", lw=1, label="0.7 bar")
    axR.set_xlabel("depth L (layers)"); axR.set_ylabel("seed match@0.7")
    axR.set_ylim(0, 1.05); axR.set_title("Seed-stability tracks SAE source type, not depth")
    axR.legend(fontsize=8, loc="center left"); axR.grid(alpha=0.3)
    fig.tight_layout()
    return fig
