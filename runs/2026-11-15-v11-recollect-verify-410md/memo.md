# v1.1 — the Pythia-410m-deduped activation store, rebuilt and verified

**Date:** 2026-11-15. **Why:** the original `data/pythia-410m-deduped/` (the query-position
residual stream every 410m result was computed from, collected 2026-06-21) no longer
exists on the study machine, and there is no backup. `data/` is git-ignored by design
("regenerable; never the source of truth"), so the store was regenerated with the
original scripts, config and seed, from the locally cached model weights and CounterFact
snapshot (offline, so no newer upstream revision could slip in). The author signed off
on re-collecting. This memo is the record that the rebuilt store **is** the original.

> **Verdict — VERIFIED, bit-level.** The rebuilt dataset matches the recorded dataset run
> on every field, and the Phase-1 Step-3 per-layer table recomputed from the rebuilt
> store agrees with the tracked `ridge_r2.csv` to **≤ 2.2 × 10⁻¹⁶ on every column of
> all 24 layers** — the limit of a float64 written to CSV. Same activations, same row
> order. Nothing computed on the rebuilt store needs an asterisk for provenance.

## Commands (run in this order)

    HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 uv run python scripts/build_dataset.py \
        --config configs/pythia-410m-deduped.yaml --desc v11-recollect-dataset-410md --seed 0
    HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 uv run python scripts/collect_activations.py \
        --config configs/pythia-410m-deduped.yaml --desc v11-recollect-collect-410md --seed 0
    uv run python scripts/verify_store.py --config configs/pythia-410m-deduped.yaml \
        --desc v11-recollect-verify-410md --seed 0 \
        --dataset-ref runs/2026-06-21-ph1-dataset-410md/tables/stats.json \
        --collect-ref runs/2026-06-21-ph1-collect-410md/tables/collect_stats.json \
        --ridge-ref runs/2026-06-21-ph1-step3-410md/tables/ridge_r2.csv \
        --dataset-new runs/2026-11-15-v11-recollect-dataset-410md/tables/stats.json

Wall-clock on the GTX 1050: 3 min 18 s (scoring 21,601 candidates), 35 s (collection),
1 min 30 s (verification, CPU).

## What was compared

| Check | Recorded (2026-06-21) | Rebuilt (2026-11-15) | |
|---|---|---|---|
| candidates scored | 21,601 | 21,601 | same |
| correct top-1 | 1,819 | 1,819 | same |
| kept (prob > 0.1, class floor 300) | 1,555 | 1,555 | same |
| per class | 711 / 511 / 333 | 711 / 511 / 333 | same |
| per relation (22 relations) | — | — | identical counts |
| dropped classes | occupation 32, religion 62 | same | same |
| store shape | [1555, 25, 1024] | [1555, 25, 1024] | same |
| top-1 recovery sanity | 5/5 | 5/5 | same |
| Step-3 table: `zero`, `identity`, `full`, `r16..r128` × 24 layers | `ridge_r2.csv` | recomputed | max \|Δ\| 2.2e-16 |

The Step-3 table is a good fingerprint because its `zero` column is pure arithmetic on
the held-out updates (no fit at all), `identity` and `full` are closed-form ridge fits,
and every column goes through the seed-0 train/test split, so it depends on the **order**
of the rows as well as their values. Per-layer values, recorded next to rebuilt, are in
`tables/store_fingerprint.csv`; the summary and the content hashes of the verified store
are in `tables/verify_stats.json` (`residual_stream_float32_sha256`
`6b5bcfb7…04da2fdc`), so any future rebuild can be checked against a hash directly.

## Scope

Only the primary factual store was rebuilt. The other stores the paper used (the
attention/MLP split streams, the untrained-network store, generic-text and multi-position
stores, the replication models) are still absent; nothing in v1.1 needs them, and
`scripts/reproduce.sh` needs none of them. The recorded trained dictionaries
(`sources_trained_{l1,topk}.npz` and weights) are also gone; the v1.1 seed sweep retrains
seed 0 and accepts it only if it reproduces their tracked tables
(`runs/v11-seed-stability/memo.md`).
