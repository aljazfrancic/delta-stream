#!/usr/bin/env python3
"""Assemble and filter the factual-recall dataset (Phase 1 Step 1).

Unions the configured sources, runs the primary model over every candidate
prompt, and keeps the triples the model actually knows: top-1 next token equals
the object's first token AND top-1 probability exceeds the threshold. Survivors
are stratified by ``relation_class`` (classes below the floor are dropped and
recorded) and written to ``filtered_triples.parquet``.

    python scripts/build_dataset.py --config configs/pythia-410m-deduped.yaml \
        --desc ph1-dataset-410m

Outputs (under the run dir, plus the parquet mirrored into data/<model>/):
    filtered_triples.parquet   subject, relation, object, prompt, top1_token,
                               top1_prob, relation_class (+ relation_template)
    tables/stats.json          candidate/kept counts, per-class breakdown, config
    tables/sample_50.txt       50 random kept rows (the read-it sanity check)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from layer_dynamics.cli import add_common_args, resolve_out
from layer_dynamics.data.filtering import first_token_id, top1_next_token
from layer_dynamics.data.sources import load_sources
from layer_dynamics.device import describe
from layer_dynamics.modeling import load_model
from layer_dynamics.runlog import init_run


def score_candidates(run, mcfg, dcfg, fcfg) -> pd.DataFrame:
    """Run the model over all candidate prompts and tag top-1 + correctness."""
    candidates = load_sources(list(dcfg["sources"]))
    run.log(f"loaded {len(candidates)} candidate triples from {dcfg['sources']}")

    model = load_model(mcfg["name"], device=mcfg.get("device", "auto"), dtype=mcfg["dtype"])
    run.log(f"model {mcfg['name']} on {describe(next(model.parameters()).device.type)}")

    prompts = candidates["prompt"].tolist()
    top1_ids, top1_probs = top1_next_token(
        model, prompts, batch_size=int(fcfg.get("batch_size", 32))
    )
    obj_first = np.array([first_token_id(model, o) for o in candidates["object"]])
    return candidates.assign(
        top1_token=[model.tokenizer.decode([int(i)]) for i in top1_ids],
        top1_token_id=top1_ids,
        top1_prob=top1_probs,
        object_first_id=obj_first,
        correct=top1_ids == obj_first,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="Force re-running the model even if cached scored candidates exist.",
    )
    args = parser.parse_args(argv)

    run = init_run(resolve_out(args), config_path=args.config, seed=args.seed)
    cfg = run.config
    mcfg, dcfg = cfg["model"], cfg["dataset"]
    fcfg = cfg.get("filtering", {})
    threshold = float(dcfg["top1_prob_threshold"])
    min_per_class = int(dcfg["min_examples_per_relation"])

    # 1-3. Score candidates, or reuse the cache. Scoring (the model run) is the
    # only expensive step; threshold/floor are applied below, so cached scores
    # make re-filtering at a new threshold near-instant. The cache is keyed on the
    # source set (encoded in the filename) so changing `dataset.sources` does not
    # silently reuse scores for a different candidate pool.
    sources = list(dcfg["sources"])
    sources_tag = "-".join(sorted(sources))
    scored_path = Path("data") / mcfg["name"] / f"scored_candidates_{sources_tag}.parquet"
    if scored_path.exists() and not args.rescore:
        candidates = pd.read_parquet(scored_path)
        run.log(f"reusing {len(candidates)} cached scored candidates from {scored_path} "
                f"(pass --rescore to re-run the model)")
    else:
        candidates = score_candidates(run, mcfg, dcfg, fcfg)
        scored_path.parent.mkdir(parents=True, exist_ok=True)
        candidates.to_parquet(scored_path, index=False)
        run.log(f"saved {len(candidates)} scored candidates -> {scored_path}")

    # 4. Keep: correct first token AND confident.
    correct = candidates["correct"].to_numpy()
    confident = candidates["top1_prob"].to_numpy() > threshold
    kept = candidates[correct & confident].copy()
    run.log(
        f"correct top-1: {int(correct.sum())}/{len(candidates)} | "
        f"+ prob>{threshold}: {len(kept)} kept"
    )

    # 5. Stratify by relation_class; drop classes below the floor.
    class_counts = kept["relation_class"].value_counts().to_dict()
    dropped = {c: n for c, n in class_counts.items() if n < min_per_class}
    if dropped:
        run.log(f"dropping classes below {min_per_class}: {dropped}")
        kept = kept[~kept["relation_class"].isin(dropped)].reset_index(drop=True)

    out_cols = [
        "subject", "relation", "relation_template", "object",
        "prompt", "top1_token", "top1_prob", "relation_class",
    ]
    kept = kept[out_cols]

    # 6. Write parquet (run dir = provenance; data/<model> = canonical for Step 2).
    run_parquet = run.path("filtered_triples.parquet")
    kept.to_parquet(run_parquet, index=False)
    canonical = Path("data") / mcfg["name"] / "filtered_triples.parquet"
    canonical.parent.mkdir(parents=True, exist_ok=True)
    kept.to_parquet(canonical, index=False)
    run.log(f"wrote {len(kept)} rows -> {run_parquet} and {canonical}")

    # 7. Stats + sanity sample.
    stats = {
        "n_candidates": len(candidates),
        "n_correct_top1": int(correct.sum()),
        "n_kept": len(kept),
        "threshold": threshold,
        "min_examples_per_relation": min_per_class,
        "target_size": int(dcfg.get("target_size", 0)),
        "meets_target": len(kept) >= int(dcfg.get("target_size", 0)),
        "kept_per_class": kept["relation_class"].value_counts().to_dict(),
        "kept_per_relation": kept["relation"].value_counts().to_dict(),
        "dropped_classes": dropped,
        "model": mcfg["name"],
    }
    run.save_json("tables/stats.json", stats)

    n_sample = min(50, len(kept))
    sample = kept.sample(n=n_sample, random_state=args.seed)
    lines = [
        f"[{r.relation_class}/{r.relation}] {r.prompt!r} -> {r.object!r} "
        f"(p={r.top1_prob:.3f})"
        for r in sample.itertuples()
    ]
    run.path("tables/sample_50.txt").write_text("\n".join(lines) + "\n")
    run.log(f"kept per class: {stats['kept_per_class']}")
    print("\n=== 50-row sanity sample ===")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
