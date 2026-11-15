#!/usr/bin/env python3
"""v1.1 E1 (post-hoc) — dictionary-level seed stability of the trained 410m SAEs.

The Gate-3 / E2 ``match@0.7`` re-seeds the **SSM** over one fixed dictionary and pairs
features by index (``scripts/step4_sanity.py``). This script answers the question that
statistic does *not*: do SAEs trained on the same cached activations from different
random initialisations converge to the same decoder directions? It reads only the
artefacts ``scripts/train_410m_saes.py`` writes (one ``sources_*.npz`` +
``.weights.pkl`` per seed), so it needs no GPU and no activation store.

Per mechanism (``--l1`` / ``--topk``, each a list of ≥2 per-seed npz files), per layer
and per seed pair, on alive latents: best-match and one-to-one decoder cosines, with a
random-dictionary null, a cross-seed cross-layer null, and two initialisation
diagnostics — the same seed's dictionary at a distant layer, and each latent's cosine
with its own initial decoder column (see
``layer_dynamics.diagnostics.sae_seed_stability``). **Primary statistic, fixed in
``runs/v11-seed-stability/memo.md`` before any run:** match@0.7 over the top-100 latents
by activation mass per layer, best-match, averaged over layers and seed pairs.

    uv run python scripts/sae_seed_stability.py --config configs/pythia-410m-deduped.yaml \
        --desc v11-e1-sae-seed-stability --seed 0 \
        --l1 data/pythia-410m-deduped/v11/sources_trained_l1_s{0,1,2}.npz \
        --topk data/pythia-410m-deduped/v11/sources_trained_topk_s{0,1,2}.npz \
        --l1-ref runs/2026-06-29-ph4-e2-train-l1-410md/tables/sources_trained_l1.json \
        --topk-ref runs/2026-06-29-ph4-e2-train-topk-410md/tables/sources_trained_topk.json

``--{l1,topk}-ref`` checks that the **first** npz of that mechanism (seed 0) reproduces
the recorded run's per-layer alive counts and reconstruction R² — the acceptance test
for a retrained seed 0 (and for the store and hyperparameters behind it). ``--seed``
seeds the random-dictionary null only.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from layer_dynamics.cli import add_common_args, resolve_out
from layer_dynamics.diagnostics.sae_seed_stability import (
    PRIMARY,
    STABLE_BAR,
    STAT_KEYS,
    compare_pair,
    init_retention,
    load_dictionaries,
    null_status,
    reproduction_check,
    summarise,
    verdict,
)
from layer_dynamics.runlog import init_run


def _mechanism(name: str, paths: list[Path], ref: Path | None, *, top_n: int,
               threshold: float, rng: np.random.Generator, log,
               ) -> tuple[list[dict], list[dict], dict]:
    """All seed pairs of one mechanism → (per-layer rows, init-retention rows, summary)."""
    dicts, alive, init_rows = [], {}, []
    for i, p in enumerate(paths):
        d = load_dictionaries(p)
        dicts.append(d)
        alive[str(p)] = {int(ell): d[ell].n_alive for ell in sorted(d)}
        log(f"[{name}] draw {i}: {p}  layers {min(d)}..{max(d)}  "
            f"alive min/mean/max = {min(alive[str(p)].values())}/"
            f"{np.mean(list(alive[str(p)].values())):.0f}/{max(alive[str(p)].values())}")
        for ell in sorted(d):
            ret = init_retention(d[ell], top_n=top_n, threshold=threshold)
            if ret:
                init_rows.append({"mechanism": name, "draw": i, "layer": ell,
                                  "n_alive": d[ell].n_alive, **ret})
    block: dict = {"paths": [str(p) for p in paths], "alive_per_layer": alive}
    if init_rows:  # mean over layers within a draw, then over draws
        keys = [k for k in init_rows[0] if k.startswith("init_")]
        per_draw = [{k: float(np.mean([r[k] for r in init_rows if r["draw"] == i]))
                     for k in keys} for i in sorted({r["draw"] for r in init_rows})]
        block["init_retention"] = {k: float(np.mean([x[k] for x in per_draw])) for k in keys}
        log(f"[{name}] init retention (alive latents vs their own initial column): "
            f"mean cos {block['init_retention']['init_cos_mean_all']:.3f}, "
            f"fraction >= {threshold:g}: {block['init_retention']['init_frac_all']:.3f}")
    if ref is not None:
        with np.load(paths[0]) as z:
            block["seed0_reproduction"] = reproduction_check(z["alive_count"], z["recon_r2"],
                                                             ref)
        block["seed0_reproduction"]["reference"] = str(ref)
        r = block["seed0_reproduction"]
        log(f"[{name}] seed-0 reproduction vs {ref}: {'OK' if r['ok'] else 'FAILED'}  "
            f"M* {r.get('M_star')} vs {r.get('ref_M_star')}, max per-layer alive rel diff "
            f"{r.get('max_alive_rel_diff', float('nan')):.4f}, mean recon R² diff "
            f"{r.get('r2_mean_abs_diff', float('nan')):.5f}")
    rows: list[dict] = []
    for i, j in combinations(range(len(dicts)), 2):
        pair_rows = compare_pair(dicts[i], dicts[j], top_n=top_n, threshold=threshold, rng=rng)
        for r in pair_rows:
            rows.append({"mechanism": name, "pair": f"{i}-{j}", "draw_a": i, "draw_b": j, **r})
        real = [r[PRIMARY] for r in pair_rows if r["kind"] == "real"]
        log(f"[{name}] draws {i}-{j}: primary (top-{top_n} best-match match@{threshold:g}) "
            f"= {np.mean(real):.3f} over {len(real)} layers")
    summ = summarise(rows)
    block.update({"summary": summ, "null_status": null_status(summ, bar=threshold),
                  "primary": summ["real"][PRIMARY],
                  "stable": bool(summ["real"][PRIMARY] >= threshold)})
    return rows, init_rows, block


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--l1", type=Path, nargs="+", default=None,
                        help="per-seed sources npz of the trained L1 dictionaries (seed 0 first)")
    parser.add_argument("--topk", type=Path, nargs="+", default=None,
                        help="per-seed sources npz of the trained TopK dictionaries")
    parser.add_argument("--l1-ref", type=Path, default=None,
                        help="tracked sources_trained_l1.json the first --l1 npz must reproduce")
    parser.add_argument("--topk-ref", type=Path, default=None)
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=STABLE_BAR)
    args = parser.parse_args(argv)
    mechanisms = {k: v for k, v in (("l1", args.l1), ("topk", args.topk)) if v}
    if not mechanisms or any(len(v) < 2 for v in mechanisms.values()):
        parser.error("give --l1 and/or --topk, each with at least two per-seed npz files")

    run = init_run(resolve_out(args), config_path=args.config, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    refs = {"l1": args.l1_ref, "topk": args.topk_ref}
    rows: list[dict] = []
    init_rows: list[dict] = []
    blocks: dict[str, dict] = {}
    for name, paths in mechanisms.items():
        r, ir, blocks[name] = _mechanism(name, paths, refs[name], top_n=args.top_n,
                                         threshold=args.threshold, rng=rng, log=run.log)
        rows += r
        init_rows += ir

    if init_rows:
        pd.DataFrame(init_rows).to_csv(run.path("tables/e1_init_retention.csv"), index=False)
    df = pd.DataFrame(rows)
    df.to_csv(run.path("tables/e1_layers.csv"), index=False)
    by_pair = (df.groupby(["mechanism", "kind", "pair"], sort=True)[list(STAT_KEYS)].mean()
               .reset_index())
    by_pair.to_csv(run.path("tables/e1_pairs.csv"), index=False)
    by_layer = (df[df["kind"] == "real"]
                .groupby(["mechanism", "layer"], sort=True)[
                    [*STAT_KEYS, "n_alive_a", "n_alive_b"]].mean().reset_index())
    by_layer.to_csv(run.path("tables/e1_by_layer.csv"), index=False)

    stats = {"model": run.config.get("model", {}).get("name"), "top_n": args.top_n,
             "threshold": args.threshold, "primary_statistic": PRIMARY,
             "mechanisms": blocks}
    if {"l1", "topk"} <= set(blocks):
        stats["verdict"] = verdict(blocks["l1"]["primary"], blocks["topk"]["primary"],
                                   bar=args.threshold)
    run.save_json("tables/e1_summary.json", stats)

    print(f"\n=== v1.1 E1 — dictionary-level seed stability ({stats['model']}) ===")
    ok = True
    for name, b in blocks.items():
        s, ns = b["summary"], b["null_status"]
        print(f"  [{name}] primary top-{args.top_n} best-match match@{args.threshold:g} = "
              f"{b['primary']:.3f}  ({'STABLE' if b['stable'] else 'unstable'}; "
              f"{s['real']['n_pairs']} seed pair(s))")
        print(f"         all-alive match {s['real']['match_all']:.3f} | mean max-cos "
              f"{s['real']['mean_maxcos_all']:.3f} | one-to-one mean cos "
              f"{s['real']['onetoone_mean_cos']:.3f}")
        print("         nulls (primary): " + ", ".join(
            f"{k.removeprefix('null_')} {v:.3f}" for k, v in ns["nulls"].items())
            + ("" if ns["informative"] else "   <-- STATISTIC UNINFORMATIVE vs nulls"))
        diag = s.get("diag_xlayer_same_seed")
        ret = b.get("init_retention")
        if diag is not None or ret is not None:
            print("         init diagnostics: "
                  + (f"same-seed distant-layer {diag[PRIMARY]:.3f}" if diag else "")
                  + (" | " if diag and ret else "")
                  + (f"alive latents within cos {args.threshold:g} of their own init: "
                     f"{ret['init_frac_all']:.3f} (mean cos {ret['init_cos_mean_all']:.3f})"
                     if ret else ""))
        rep = b.get("seed0_reproduction")
        if rep is not None:
            print(f"         seed-0 reproduction of the recorded run: "
                  f"{'OK' if rep['ok'] else 'FAILED — do not trust the cross-seed numbers'}")
            ok = ok and rep["ok"]
    if "verdict" in stats:
        print(f"  verdict row: {stats['verdict']['row']}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
