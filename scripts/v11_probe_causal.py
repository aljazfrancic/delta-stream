#!/usr/bin/env python3
"""v1.1 check A, part 2 — the state probe with depth-causal arms and a feedthrough arm.

The recorded probe (``scripts/phase4_state_probe.py``, R7) found the SSM state ``z_ℓ``
decoding answer identity better than dimension-matched PCA of the codes it is built from.
Two things were left open (paper §6.7):

  1. **Look-ahead.** The codes are fit to each example's whole update trajectory, and what
     the earlier codes learn about ``Δ_ℓ`` reaches them only through the state map. The
     state is therefore the arm best placed to read that look-ahead back out.
  2. **No per-layer arm trained on Δ.** PCA is variance-ranked; the state is
     reconstruction-trained. The contrast does not separate a *cross-layer* state from
     any reconstruction-trained projection of the codes.

This script repeats the recorded probe verbatim (same four arms, same linear models, same
selection split, same RNG stream — so it must reproduce the tracked table) and adds:

  * ``feed_image``         — the feedthrough's rank-r image ``D2_ℓᵀ u_ℓ`` of the same
                             full-trajectory codes: per-layer, reconstruction-trained,
                             with the same exposure to ``Δ_ℓ`` as the recorded state.
  * ``state_causal``       — the state built from codes inferred against ``Δ_{<ℓ}`` only.
  * ``prefix_pca_causal``  — PCA of those same prefix codes (exactly the causal state's
                             inputs).
  * ``feed_prev_causal``   — the feedthrough's image of the last past code ``u_{ℓ-1}``.
  * ``sae_pca``            — PCA of the SAE latents of ``v_ℓ`` (the encoder's warm start;
                             no LASSO, no access to any update at or after ℓ).

The causal features come from ``scripts/v11_causal_codes.py``. The primary contrast,
fixed in ``runs/v11-onbox-checks/memo.md`` before any run, is
``state_causal − prefix_pca_causal`` on answer identity in the middle depth band.

    uv run python scripts/v11_probe_causal.py --config configs/pythia-410m-deduped.yaml \\
        --out runs/<date>-v11c-probe-causal-mlsae-410md \\
        --checkpoint-dir runs/<date>-v11c-checkpoint-mlsae-410md \\
        --causal-dir runs/<date>-v11c-causal-codes-mlsae-410md \\
        --sources-npz data/pythia-410m-deduped/sources_mlsae.npz \\
        --recorded-json runs/2026-07-04-r7-state-probe-mlsae/tables/state_probe.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from layer_dynamics.checkpoint import load_ssm
from layer_dynamics.cli import add_common_args, resolve_out
from layer_dynamics.diagnostics.dataset import load_residual_stream
from layer_dynamics.diagnostics.v11_checks import PROBE_PRIMARY, probe_reproduction, probe_verdict
from layer_dynamics.runlog import init_run
from layer_dynamics.saes_mlsae import load_cached_u

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase4_state_probe import (  # noqa: E402  (the recorded probe's own helpers, verbatim)
    BANDS,
    _band_layers,
    _classifier,
    _paired_acc_ci,
    _pca_fit,
    _ridge,
    _standardize,
)

RECORDED_ARMS = ["state", "code_pca", "code_prefix_pca", "resid_pca"]
NEW_ARMS = ["feed_image", "state_causal", "prefix_pca_causal", "feed_prev_causal", "sae_pca"]
# (a, b): band-aggregated paired contrast a − b on answer identity and relation class.
CONTRASTS = [
    ("state", "code_pca"), ("state", "code_prefix_pca"), ("state", "resid_pca"),   # recorded
    ("state_causal", "prefix_pca_causal"),                                           # primary
    ("state_causal", "feed_prev_causal"), ("state_causal", "sae_pca"),
    ("state_causal", "resid_pca"), ("state", "feed_image"), ("state", "state_causal"),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--causal-dir", type=Path, required=True,
                        help="v11_causal_codes run dir (checkpoints/features_causal.npz).")
    parser.add_argument("--sources-npz", type=Path, required=True)
    parser.add_argument("--recorded-json", type=Path, default=None,
                        help="Tracked R7 state_probe.json the recorded arms must reproduce.")
    parser.add_argument("--c-grid", type=float, nargs="+", default=[0.1, 1.0, 10.0])
    parser.add_argument("--alpha-grid", type=float, nargs="+",
                        default=[1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3])
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Smoke only: probe these layers and skip the band contrasts.")
    args = parser.parse_args(argv)

    run = init_run(resolve_out(args), config_path=args.config, seed=args.seed)
    mcfg = run.config["model"]

    ckpt = args.checkpoint_dir / "checkpoints"
    split = np.load(ckpt / "split.npz")
    train_idx, test_idx = split["train_idx"], split["test_idx"]
    u = np.load(ckpt / "u_infer_full.npy")  # [N, L, M]
    model = load_ssm(ckpt / "ssm_seed0.pt")
    z = model.states(torch.from_numpy(u)).numpy()  # [N, L, d_z]
    d_z = z.shape[2]
    d2 = model.D2.detach().numpy()  # [L, M, r]
    rs, rel_class, _ = load_residual_stream(Path("data") / mcfg["name"] / "residual_stream.h5")
    n, n_layers = u.shape[0], u.shape[1]
    with np.load(args.causal_dir / "checkpoints" / "features_causal.npz") as cf:
        causal = {k: cf[k] for k in ("z_causal", "feed_prev", "prefix_pca")}
        causal_depths = [int(d) for d in cf["depths"]]
        causal_n = int(cf["n_examples"])
    if causal_n != n:
        raise SystemExit(f"causal features cover {causal_n} examples, the checkpoint {n}")
    u_sae = load_cached_u(args.sources_npz)
    run.log(f"codes u {u.shape}, states z {z.shape}, residuals {rs.shape}; split "
            f"{len(train_idx)}/{len(test_idx)}; probe dim d={d_z}; causal depths "
            f"{causal_depths[0]}..{causal_depths[-1]} ({len(causal_depths)})")

    import pandas as pd

    df = pd.read_parquet(Path("data") / mcfg["name"] / "filtered_triples.parquet")
    df = df.reset_index(drop=True)
    if len(df) != n or int((df["relation_class"].to_numpy() != rel_class).sum()):
        raise ValueError("metadata not aligned to store")
    y_obj = df["object"].to_numpy()
    y_rel = df["relation_class"].to_numpy()
    p = df["top1_prob"].to_numpy().clip(1e-6, 1 - 1e-6)
    y_conf = np.log(p / (1 - p)).astype(np.float64)

    # Same selection split and RNG stream as the recorded probe: the permutation first,
    # then two draws per layer for the state arm's permuted-label null. The new arms
    # consume no random numbers, so the recorded arms' numbers are unchanged.
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(train_idx))
    n_val = int(round(len(train_idx) * args.val_frac))
    val_idx = train_idx[perm[:n_val]]
    fit_idx = train_idx[perm[n_val:]]
    run.log(f"probe selection split: {len(fit_idx)} fit / {len(val_idx)} val")

    majority = {
        "object": float((y_obj[test_idx] == pd.Series(y_obj[train_idx]).mode()[0]).mean()),
        "relation_class": float((y_rel[test_idx] ==
                                 pd.Series(y_rel[train_idx]).mode()[0]).mean()),
    }

    arms = RECORDED_ARMS + NEW_ARMS
    layers = sorted(set(args.layers)) if args.layers else list(range(n_layers))
    acc1 = {a: np.full(n_layers, np.nan) for a in arms}
    acc5 = {a: np.full(n_layers, np.nan) for a in arms}
    accr = {a: np.full(n_layers, np.nan) for a in arms}
    r2c = {a: np.full(n_layers, np.nan) for a in arms}
    null_acc1 = np.full(n_layers, np.nan)
    correct_obj = {a: np.zeros((len(test_idx), n_layers), dtype=bool) for a in arms}
    correct_rel = {a: np.zeros((len(test_idx), n_layers), dtype=bool) for a in arms}

    def pca_features(x: np.ndarray) -> np.ndarray:
        mean, comp = _pca_fit(x[train_idx].astype(np.float32, copy=False), d_z)
        return ((x - mean) @ comp).astype(np.float64)

    def features(arm: str, ell: int) -> np.ndarray:
        if arm == "state":
            return z[:, ell].astype(np.float64)
        if arm == "code_pca":
            return pca_features(u[:, ell])
        if arm == "code_prefix_pca":
            return pca_features(u[:, : ell + 1].reshape(n, -1))
        if arm == "resid_pca":
            return pca_features(rs[:, ell])
        if arm == "feed_image":
            return (u[:, ell] @ d2[ell]).astype(np.float64)
        if arm == "state_causal":
            return causal["z_causal"][:, ell].astype(np.float64)
        if arm == "prefix_pca_causal":
            return causal["prefix_pca"][:, ell].astype(np.float64)
        if arm == "feed_prev_causal":
            return causal["feed_prev"][:, ell].astype(np.float64)
        if arm == "sae_pca":
            return pca_features(u_sae[:, ell])
        raise ValueError(arm)

    for ell in layers:
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
            if arm == "state":  # permuted-label null (floor calibration), as recorded
                y_perm = rng.permutation(y_obj[fit_idx])
                n1, _, _ = _classifier(x_fit, y_perm, x_val,
                                       rng.permutation(y_obj[val_idx]),
                                       x_te, y_obj[test_idx], c_grid=args.c_grid,
                                       seed=args.seed)
                null_acc1[ell] = float(n1.mean())
        run.log(f"  ℓ={ell:2d} object acc: state {acc1['state'][ell]:.3f} "
                f"prefix {acc1['code_prefix_pca'][ell]:.3f} feed {acc1['feed_image'][ell]:.3f} | "
                f"causal: state {acc1['state_causal'][ell]:.3f} "
                f"prefix {acc1['prefix_pca_causal'][ell]:.3f} "
                f"feed {acc1['feed_prev_causal'][ell]:.3f} sae {acc1['sae_pca'][ell]:.3f} | "
                f"resid {acc1['resid_pca'][ell]:.3f}")

    contrasts: dict = {}
    full = len(layers) == n_layers
    if full:
        for name, (lo, hi) in BANDS.items():
            band = _band_layers(n_layers, lo, hi)
            for a, b in CONTRASTS:
                contrasts[f"object/{a}−{b}/{name}"] = _paired_acc_ci(
                    correct_obj[a][:, band], correct_obj[b][:, band],
                    n_boot=args.n_boot, seed=args.seed)
                contrasts[f"relation/{a}−{b}/{name}"] = _paired_acc_ci(
                    correct_rel[a][:, band], correct_rel[b][:, band],
                    n_boot=args.n_boot, seed=args.seed)

    def as_list(d):
        return {a: [None if np.isnan(v) else float(v) for v in d[a]] for a in arms}

    summary = {
        "checkpoint_dir": str(args.checkpoint_dir), "causal_dir": str(args.causal_dir),
        "sources_npz": str(args.sources_npz), "d_probe": int(d_z),
        "n_layers": int(n_layers), "n_test": int(len(test_idx)), "layers": layers,
        "c_grid": args.c_grid, "alpha_grid": args.alpha_grid,
        "majority_acc": majority, "arms": arms,
        "object_top1_acc": as_list(acc1), "object_top5_acc": as_list(acc5),
        "relation_acc": as_list(accr), "conf_logit_r2": as_list(r2c),
        "object_null_acc_state": [None if np.isnan(v) else float(v) for v in null_acc1],
        "band_contrasts": contrasts,
        "note": ("Recorded arms (state, code_pca, code_prefix_pca, resid_pca) are the R7 "
                 "probe verbatim, on codes fit to each example's whole trajectory. "
                 "feed_image = D2_ℓᵀ u_ℓ of those codes. state_causal / prefix_pca_causal / "
                 "feed_prev_causal use codes inferred against Δ_0..Δ_{ℓ-1} only "
                 "(v11_causal_codes.py); at ℓ=0 they are constant. sae_pca = PCA of the SAE "
                 "latents of v_ℓ. All arms at d=d_z, identical linear probes, C/α selected "
                 "on a train-internal validation split."),
    }
    if full:
        summary["primary"] = {"contrast": PROBE_PRIMARY, **contrasts[PROBE_PRIMARY]}
        summary["verdict"] = probe_verdict(contrasts[PROBE_PRIMARY])
    if args.recorded_json is not None:
        recorded = json.loads(args.recorded_json.read_text(encoding="utf-8"))
        summary["reproduction"] = probe_reproduction(summary, recorded, RECORDED_ARMS, layers)
        rep = summary["reproduction"]
        run.log(f"recorded arms vs {args.recorded_json}: max |Δacc| = "
                f"{rep['max_abs_diff_object_top1']:.2e} -> "
                f"{'REPRODUCED' if rep['ok'] else 'NOT reproduced'}")
    run.save_json("tables/state_probe_causal.json", summary)
    rows = [{"arm": a, "layer": ell, "object_top1": acc1[a][ell], "object_top5": acc5[a][ell],
             "relation": accr[a][ell], "conf_r2": r2c[a][ell]} for a in arms for ell in layers]
    pd.DataFrame(rows).to_csv(run.path("tables/state_probe_causal.csv"), index=False)
    np.savez_compressed(run.path("tables/probe_correct_test.npz"), test_idx=test_idx,
                        **{f"obj_{a}": correct_obj[a] for a in arms},
                        **{f"rel_{a}": correct_rel[a] for a in arms})

    print(f"\n=== v1.1 causal state probe ({mcfg.get('name')}, d={d_z}) ===")
    print(f"  majority: object {majority['object']:.3f}")
    for a in arms:
        best = int(np.nanargmax(acc1[a]))
        print(f"  {a:18s} object top-1 max {np.nanmax(acc1[a]):.3f} (ℓ={best})")
    if full:
        print("  band contrasts (object top-1, paired 95% CI):")
        for key, c in contrasts.items():
            if key.startswith("object"):
                star = " *" if c["significant"] else ""
                print(f"    {key:52s} {c['diff']:+.4f} [{c['ci_lo']:+.4f}, "
                      f"{c['ci_hi']:+.4f}]{star}")
        print(f"  primary ({PROBE_PRIMARY}): {summary['verdict']}")
    if args.recorded_json is not None and not summary["reproduction"]["ok"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
