#!/usr/bin/env python3
"""R7 — probe the SSM latent state: does ``z_ℓ`` carry *functional* information?

The 4th-pass review's top experiment: the decomposition's demonstrated value is
reconstruction (Step 1) — the causal story is a clean null — so the one cheap result
that could still give it a functional payoff (and change the abstract) is whether the
learned 32-dim cross-layer state ``z_ℓ`` linearly predicts task quantities *better than
a dimension-matched summary of the per-layer code*. If yes, the state is an
information-bearing object, not just a reconstruction device; if no, "reconstruction-
only" is the honest abstract.

Arms (all d = d_z per layer, probed with the same linear model + preprocessing):

  * ``state``            — ``z_ℓ`` from the canonical checkpoint (LASSO codes u → states).
                           ``z_ℓ`` is a function of ``u_0 … u_{ℓ-1}`` only.
  * ``code_pca``         — top-d_z PCA (train-fit) of the same layer's code ``u_ℓ``:
                           the registered "matched per-layer SAE code at equal dimension".
  * ``code_prefix_pca``  — top-d_z PCA (train-fit) of the concatenated prefix codes
                           ``u_0 … u_ℓ``: an equal-dimension summary of a *superset* of
                           the state's inputs (if ``state`` beats even this, the learned
                           A/B compression is doing real selection, not just aggregation).
  * ``resid_pca``        — top-d_z PCA (train-fit) of the raw residual ``v_ℓ``: reference
                           upper bound (the model's own stream, which the unembedding
                           reads — expected to win; the question is how much of it the
                           state retains).

Targets (correctness is degenerate — the dataset is filtered to model-correct facts):

  * ``object``           — answer identity (205 classes), multinomial logistic, top-1/5 acc.
  * ``relation_class``   — 3-way logistic accuracy.
  * ``logit_conf``       — logit of the model's answer prob (ridge R²) — confidence.

Honesty devices: probe hyperparameter (C / α) selected on a *train-internal* validation
carve-out, never on test (4th-pass finding 4); a permuted-train-label null on the
``state`` arm calibrates accuracy floors; per-example test correctness is saved for all
arms and the headline contrast (state − code_pca) gets a **paired bootstrap CI
band-aggregated over depth thirds** (no max-over-layers selection).

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python \\
        scripts/phase4_state_probe.py --config configs/pythia-410m-deduped.yaml \\
        --desc r7-state-probe-mlsae --checkpoint-dir runs/2026-06-26-ph4-checkpoint-410md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from layer_dynamics.checkpoint import load_ssm
from layer_dynamics.cli import add_common_args, resolve_out
from layer_dynamics.diagnostics.dataset import load_residual_stream
from layer_dynamics.runlog import init_run

BANDS = {"early": (0.0, 1 / 3), "mid": (1 / 3, 2 / 3), "late": (2 / 3, 1.0)}


def _band_layers(n_layers: int, lo_frac: float, hi_frac: float) -> list[int]:
    return [ell for ell in range(n_layers)
            if lo_frac <= ell / n_layers < hi_frac or (hi_frac == 1.0 and ell == n_layers - 1)]


def _pca_fit(x_tr: np.ndarray, d: int) -> tuple[np.ndarray, np.ndarray]:
    """Train-fit PCA → (mean[F], components[F, d]). Gram trick when F ≫ N."""
    mean = x_tr.mean(axis=0)
    xc = x_tr - mean
    n, f = xc.shape
    if f <= 4 * n:
        # economy SVD in feature space
        _, s, vt = np.linalg.svd(xc, full_matrices=False)
        comp = vt[:d].T
    else:
        k = xc @ xc.T  # [n, n]
        w, v = np.linalg.eigh(k)
        order = np.argsort(w)[::-1][:d]
        w, v = w[order], v[:, order]
        w = np.maximum(w, 1e-12)
        comp = xc.T @ (v / np.sqrt(w))  # [f, d], unit columns
    if comp.shape[1] < d:  # degenerate layer (e.g. u_0 = 0)
        comp = np.pad(comp, ((0, 0), (0, d - comp.shape[1])))
    return mean, comp


def _standardize(tr: np.ndarray, te: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m, s = tr.mean(axis=0), tr.std(axis=0)
    s[s < 1e-12] = 1.0
    return (tr - m) / s, (te - m) / s


def _classifier(x_tr, y_tr, x_val, y_val, x_te, y_te, *, c_grid, seed):
    """Logistic probe: C on validation, refit train+val, evaluate test.

    Returns (correct_top1[N_te] bool, correct_top5[N_te] bool | None, best_C).
    """
    from sklearn.linear_model import LogisticRegression

    best_c, best_acc = c_grid[0], -1.0
    for c in c_grid:
        clf = LogisticRegression(C=c, max_iter=2000, random_state=seed)
        clf.fit(x_tr, y_tr)
        acc = float((clf.predict(x_val) == y_val).mean())
        if acc > best_acc:
            best_c, best_acc = c, acc
    clf = LogisticRegression(C=best_c, max_iter=2000, random_state=seed)
    clf.fit(np.concatenate([x_tr, x_val]), np.concatenate([y_tr, y_val]))
    correct1 = clf.predict(x_te) == y_te
    correct5 = None
    if len(clf.classes_) > 5:
        proba = clf.predict_proba(x_te)
        top5 = clf.classes_[np.argsort(proba, axis=1)[:, -5:]]
        correct5 = np.array([y in row for y, row in zip(y_te, top5, strict=True)])
    return correct1, correct5, best_c


def _ridge(x_tr, y_tr, x_val, y_val, x_te, y_te, *, alpha_grid, seed):
    """Ridge probe: α on validation, refit train+val; returns (per-example squared
    residuals on test, ss_tot vs train-mean, best α)."""
    from sklearn.linear_model import Ridge

    best_a, best_r2 = alpha_grid[0], -np.inf
    for a in alpha_grid:
        reg = Ridge(alpha=a, random_state=seed).fit(x_tr, y_tr)
        pred = reg.predict(x_val)
        r2 = 1.0 - float(((y_val - pred) ** 2).sum()) / float(
            ((y_val - y_tr.mean()) ** 2).sum())
        if r2 > best_r2:
            best_a, best_r2 = a, r2
    x_all, y_all = np.concatenate([x_tr, x_val]), np.concatenate([y_tr, y_val])
    reg = Ridge(alpha=best_a, random_state=seed).fit(x_all, y_all)
    ss_res = (y_te - reg.predict(x_te)) ** 2
    ss_tot = (y_te - y_all.mean()) ** 2
    return ss_res, ss_tot, best_a


def _paired_acc_ci(correct_a: np.ndarray, correct_b: np.ndarray, *, n_boot: int,
                   seed: int) -> dict:
    """Paired bootstrap CI for mean(correct_a) − mean(correct_b) over test examples.

    Inputs may be [N] (one layer) or [N, n_layers] (band: per-example accuracy is the
    mean over the band's layers before differencing)."""
    a = correct_a.reshape(len(correct_a), -1).mean(axis=1).astype(np.float64)
    b = correct_b.reshape(len(correct_b), -1).mean(axis=1).astype(np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    boots = (a[idx] - b[idx]).mean(axis=1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"diff": float((a - b).mean()), "ci_lo": float(lo), "ci_hi": float(hi),
            "significant": bool(lo > 0 or hi < 0)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--checkpoint-dir", type=Path, required=True,
                        help="phase4_checkpoint run dir (operators + LASSO codes + split)")
    parser.add_argument("--c-grid", type=float, nargs="+", default=[0.1, 1.0, 10.0])
    parser.add_argument("--alpha-grid", type=float, nargs="+",
                        default=[1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3])
    parser.add_argument("--val-frac", type=float, default=0.2,
                        help="Train-internal validation fraction for C/α selection.")
    parser.add_argument("--n-boot", type=int, default=10_000)
    args = parser.parse_args(argv)

    run = init_run(resolve_out(args), config_path=args.config, seed=args.seed)
    mcfg = run.config["model"]

    # --- cached inputs: codes u (LASSO, what the SSM consumes), states z, residuals v ---
    ckpt = args.checkpoint_dir / "checkpoints"
    split = np.load(ckpt / "split.npz")
    train_idx, test_idx = split["train_idx"], split["test_idx"]
    u = np.load(ckpt / "u_infer_full.npy")  # [N, L, M]
    model = load_ssm(ckpt / "ssm_seed0.pt")
    z = model.states(torch.from_numpy(u)).numpy()  # [N, L, d_z]
    d_z = z.shape[2]
    rs, rel_class, _ = load_residual_stream(Path("data") / mcfg["name"] / "residual_stream.h5")
    n, n_layers = u.shape[0], u.shape[1]
    run.log(f"codes u {u.shape}, states z {z.shape}, residuals {rs.shape}; "
            f"split {len(train_idx)}/{len(test_idx)}; probe dim d={d_z}")

    # --- labels, aligned to the store (same check as phase4_checkpoint) ---
    import pandas as pd

    df = pd.read_parquet(Path("data") / mcfg["name"] / "filtered_triples.parquet")
    df = df.reset_index(drop=True)
    if len(df) != n or int((df["relation_class"].to_numpy() != rel_class).sum()):
        raise ValueError("metadata not aligned to store")
    y_obj = df["object"].to_numpy()
    y_rel = df["relation_class"].to_numpy()
    p = df["top1_prob"].to_numpy().clip(1e-6, 1 - 1e-6)
    y_conf = np.log(p / (1 - p)).astype(np.float64)

    # train-internal validation carve-out (selection never sees test)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(train_idx))
    n_val = int(round(len(train_idx) * args.val_frac))
    val_idx = train_idx[perm[:n_val]]
    fit_idx = train_idx[perm[n_val:]]
    run.log(f"probe selection split: {len(fit_idx)} fit / {len(val_idx)} val")

    majority = {
        "object": float((y_obj[test_idx] ==
                         pd.Series(y_obj[train_idx]).mode()[0]).mean()),
        "relation_class": float((y_rel[test_idx] ==
                                 pd.Series(y_rel[train_idx]).mode()[0]).mean()),
    }
    run.log(f"majority-class test acc: object {majority['object']:.3f} "
            f"relation {majority['relation_class']:.3f}")

    # --- per-(arm, layer) feature matrices → probes ---
    arms = ["state", "code_pca", "code_prefix_pca", "resid_pca"]
    acc1 = {a: np.zeros(n_layers) for a in arms}
    acc5 = {a: np.zeros(n_layers) for a in arms}
    accr = {a: np.zeros(n_layers) for a in arms}
    r2c = {a: np.zeros(n_layers) for a in arms}
    null_acc1 = np.zeros(n_layers)  # permuted-label null, state arm, object target
    correct_obj = {a: np.zeros((len(test_idx), n_layers), dtype=bool) for a in arms}
    correct_rel = {a: np.zeros((len(test_idx), n_layers), dtype=bool) for a in arms}

    def features(arm: str, ell: int) -> np.ndarray:
        if arm == "state":
            return z[:, ell].astype(np.float64)
        if arm == "code_pca":
            x = u[:, ell]
        elif arm == "code_prefix_pca":
            x = u[:, : ell + 1].reshape(n, -1)
        else:
            x = rs[:, ell]
        # PCA in float32 (the prefix arm's X is up to [N, L·M] ≈ 660 MB — float64 would
        # double it twice over); only the projected [N, d_z] features go to float64.
        mean, comp = _pca_fit(x[train_idx].astype(np.float32, copy=False), d_z)
        return ((x - mean) @ comp).astype(np.float64)

    for ell in range(n_layers):
        for arm in arms:
            x = features(arm, ell)
            x_fit, x_te = _standardize(x[fit_idx], x[test_idx])
            _, x_val = _standardize(x[fit_idx], x[val_idx])
            c1, c5, _ = _classifier(x_fit, y_obj[fit_idx], x_val, y_obj[val_idx],
                                    x_te, y_obj[test_idx], c_grid=args.c_grid,
                                    seed=args.seed)
            acc1[arm][ell] = float(c1.mean())
            acc5[arm][ell] = float(c5.mean()) if c5 is not None else np.nan
            correct_obj[arm][:, ell] = c1
            r1, _, _ = _classifier(x_fit, y_rel[fit_idx], x_val, y_rel[val_idx],
                                   x_te, y_rel[test_idx], c_grid=args.c_grid,
                                   seed=args.seed)
            accr[arm][ell] = float(r1.mean())
            correct_rel[arm][:, ell] = r1
            ss_res, ss_tot, _ = _ridge(x_fit, y_conf[fit_idx], x_val, y_conf[val_idx],
                                       x_te, y_conf[test_idx],
                                       alpha_grid=args.alpha_grid, seed=args.seed)
            r2c[arm][ell] = 1.0 - float(ss_res.sum()) / float(ss_tot.sum())
            if arm == "state":  # permuted-label null (floor calibration)
                y_perm = rng.permutation(y_obj[fit_idx])
                n1, _, _ = _classifier(x_fit, y_perm, x_val,
                                       rng.permutation(y_obj[val_idx]),
                                       x_te, y_obj[test_idx], c_grid=args.c_grid,
                                       seed=args.seed)
                null_acc1[ell] = float(n1.mean())
        run.log(f"  ℓ={ell:2d} object acc: state {acc1['state'][ell]:.3f} "
                f"code {acc1['code_pca'][ell]:.3f} prefix {acc1['code_prefix_pca'][ell]:.3f} "
                f"resid {acc1['resid_pca'][ell]:.3f} (null {null_acc1[ell]:.3f})")

    # --- band-aggregated paired contrasts (no max-over-layers selection) ---
    contrasts = {}
    for name, (lo, hi) in BANDS.items():
        band = _band_layers(n_layers, lo, hi)
        for other in ("code_pca", "code_prefix_pca", "resid_pca"):
            contrasts[f"object/state−{other}/{name}"] = _paired_acc_ci(
                correct_obj["state"][:, band], correct_obj[other][:, band],
                n_boot=args.n_boot, seed=args.seed)
            contrasts[f"relation/state−{other}/{name}"] = _paired_acc_ci(
                correct_rel["state"][:, band], correct_rel[other][:, band],
                n_boot=args.n_boot, seed=args.seed)

    summary = {
        "checkpoint_dir": str(args.checkpoint_dir), "d_probe": int(d_z),
        "n_layers": int(n_layers), "n_test": int(len(test_idx)),
        "c_grid": args.c_grid, "alpha_grid": args.alpha_grid,
        "majority_acc": majority,
        "object_top1_acc": {a: acc1[a].tolist() for a in arms},
        "object_top5_acc": {a: acc5[a].tolist() for a in arms},
        "relation_acc": {a: accr[a].tolist() for a in arms},
        "conf_logit_r2": {a: r2c[a].tolist() for a in arms},
        "object_null_acc_state": null_acc1.tolist(),
        "band_contrasts": contrasts,
        "note": ("All arms probed at d=d_z with the identical linear model; C/α chosen "
                 "on a train-internal validation split (test never used for selection). "
                 "state=z_ℓ (function of u_<ℓ); code_pca=PCA-d(u_ℓ); code_prefix_pca="
                 "PCA-d(u_0..u_ℓ) — an information superset of the state's inputs; "
                 "resid_pca=PCA-d(v_ℓ) reference upper bound. Correctness target is "
                 "degenerate (dataset filtered to model-correct facts)."),
    }
    run.save_json("tables/state_probe.json", summary)
    rows = [{"arm": a, "layer": ell, "object_top1": acc1[a][ell],
             "object_top5": acc5[a][ell], "relation": accr[a][ell],
             "conf_r2": r2c[a][ell]} for a in arms for ell in range(n_layers)]
    pd.DataFrame(rows).to_csv(run.path("tables/state_probe.csv"), index=False)
    np.savez_compressed(run.path("tables/probe_correct_test.npz"),
                        test_idx=test_idx,
                        **{f"obj_{a}": correct_obj[a] for a in arms},
                        **{f"rel_{a}": correct_rel[a] for a in arms})
    _plot(run, arms, acc1, accr, r2c, null_acc1, majority, n_layers)

    print(f"\n=== R7 state probe ({mcfg.get('name')}, d={d_z}) ===")
    print(f"  majority: object {majority['object']:.3f}  relation "
          f"{majority['relation_class']:.3f}")
    for a in arms:
        print(f"  {a:16s} object top-1 max {acc1[a].max():.3f} (ℓ={int(acc1[a].argmax())}) "
              f"relation max {accr[a].max():.3f}  conf-R² max {r2c[a].max():+.3f}")
    print("  band contrasts (object top-1, state − other, paired 95% CI):")
    for key, c in contrasts.items():
        if key.startswith("object"):
            star = " *" if c["significant"] else ""
            print(f"    {key:38s} {c['diff']:+.4f} [{c['ci_lo']:+.4f}, "
                  f"{c['ci_hi']:+.4f}]{star}")
    return 0


def _plot(run, arms, acc1, accr, r2c, null_acc1, majority, n_layers) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"state": "tab:blue", "code_pca": "tab:orange",
              "code_prefix_pca": "tab:green", "resid_pca": "tab:gray"}
    labels = {"state": "SSM state z_ℓ", "code_pca": "PCA-d(u_ℓ)",
              "code_prefix_pca": "PCA-d(u_≤ℓ)", "resid_pca": "PCA-d(v_ℓ) (reference)"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    x = range(n_layers)
    for ax, data, title, ylab in (
        (axes[0], acc1, "Answer identity (205-way)", "test top-1 accuracy"),
        (axes[1], accr, "Relation class (3-way)", "test accuracy"),
        (axes[2], r2c, "Model confidence (logit)", "test R²"),
    ):
        for a in arms:
            ax.plot(x, data[a], "-o", ms=3, lw=1.6, color=colors[a], label=labels[a])
        ax.set_xlabel("layer ℓ")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].plot(x, null_acc1, ":", color="tab:red", lw=1.2, label="permuted-label null")
    axes[0].axhline(majority["object"], color="k", lw=0.8, ls="--", label="majority class")
    axes[1].axhline(majority["relation_class"], color="k", lw=0.8, ls="--")
    axes[2].axhline(0, color="k", lw=0.6)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("R7 — linear probes at equal dimension d=d_z (410m, canonical checkpoint)",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(run.path("figures/state_probe.svg"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
