#!/usr/bin/env bash
# D1 — reproduce.sh: regenerate every headline paper figure from saved intermediates
# (Phase-4 Steps 5–6: "all figures regenerate from saved intermediates via
# scripts/reproduce.sh"). The default mode needs NO GPU, NO downloads and NO data/
# stores — everything it reads is tracked in git (runs/*/tables + configs).
#
#   scripts/reproduce.sh                 # env check + tests + size guards + figures
#   scripts/reproduce.sh --with-runs     # also re-emit each run's own diagnostic
#                                        #   figures in place (scripts/replot.py)
#   scripts/reproduce.sh --toy-pipeline  # additionally run the full pipeline
#                                        #   end-to-end on the 70m debug lane
#                                        #   (GPU + HF downloads; ~30 min)
#
# Fails (nonzero) if the environment is broken, a test fails, a tracked dataset /
# store on disk has drifted from the recorded sizes (review finding: stale rows
# must not silently enter an analysis), or any of the headline figures cannot be
# rebuilt from the tracked tables.
set -euo pipefail
cd "$(dirname "$0")/.."

WITH_RUNS=0
TOY=0
for arg in "$@"; do
    case "$arg" in
        --with-runs) WITH_RUNS=1 ;;
        --toy-pipeline) TOY=1 ;;
        *) echo "unknown arg: $arg (known: --with-runs --toy-pipeline)"; exit 2 ;;
    esac
done

echo "=== [1/4] environment + provenance ==="
uv run layer-dynamics

echo "=== [2/4] unit tests ==="
uv run pytest -q

echo "=== [3/4] dataset / store size guards (skipped where data/ absent) ==="
uv run python - << 'EOF'
"""Recorded sizes for the primary lane; a mismatch means stale/foreign rows."""
import sys
from pathlib import Path

failures = []

pq = Path("data/pythia-410m-deduped/filtered_triples.parquet")
if pq.exists():
    import pandas as pd
    df = pd.read_parquet(pq)
    ok = len(df) == 1555 and df["relation_class"].nunique() == 3
    print(f"  {pq}: {len(df)} rows / {df['relation_class'].nunique()} classes "
          f"{'OK' if ok else 'MISMATCH (expect 1555 / 3)'}")
    if not ok:
        failures.append(str(pq))
else:
    print(f"  {pq}: absent (fresh install) — skipped")

expected = {
    "data/pythia-410m-deduped/residual_stream.h5": (1555, 25, 1024),
    "data/pythia-410m-deduped-attnstream/residual_stream.h5": (1555, 25, 1024),
    "data/pythia-410m-deduped-mlpstream/residual_stream.h5": (1555, 25, 1024),
}
for name, shape in expected.items():
    p = Path(name)
    if not p.exists():
        print(f"  {p}: absent (fresh install) — skipped")
        continue
    import h5py
    with h5py.File(p, "r") as f:
        got = f["residual_stream"].shape
    ok = tuple(got) == shape
    print(f"  {p}: {tuple(got)} {'OK' if ok else f'MISMATCH (expect {shape})'}")
    if not ok:
        failures.append(name)

if failures:
    sys.exit(f"size guard failed for: {failures}")
EOF

echo "=== [4/4] headline paper figures from tracked tables ==="
uv run python scripts/make_paper_figures.py   # exits nonzero unless all figures build

if [ "$WITH_RUNS" = 1 ]; then
    echo "=== [--with-runs] re-emit per-run diagnostic figures in place ==="
    uv run python scripts/replot.py
fi

if [ "$TOY" = 1 ]; then
    echo "=== [--toy-pipeline] end-to-end on the 70m debug lane (GPU + downloads) ==="
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    CFG=configs/pythia-70m.yaml
    uv run python scripts/build_dataset.py --config "$CFG" --desc repro-toy-dataset --seed 0
    uv run python scripts/collect_activations.py --config "$CFG" --desc repro-toy-collect --seed 0
    uv run python scripts/step5_ar.py --config "$CFG" --desc repro-toy-step5 --seed 0
    echo "toy pipeline done — inspect runs/*repro-toy-*"
fi

echo "=== reproduce.sh done ==="
