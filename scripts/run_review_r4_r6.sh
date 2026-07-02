#!/usr/bin/env bash
# Review items R4–R6 (2026-07-02 code-level review) — second serial GPU chain.
#
#   S1  R4 · Retrain the E2 L1 SAEs on the Gate-3 TRAIN rows only (--exclude-test:
#       dictionary + alive masks never see the 311 eval rows), then re-run the exact
#       recorded E2 Gate-3 λ-sweep and step4 seed-stability on the clean sources.
#       Removes the dictionary-level train/test contamination asterisk from E2.
#   S2  R5 · Entity-disjoint split robustness. Objects recur heavily (205 unique /
#       1555; "French" ×163), so --split-by object is the split that matters:
#       (a) Step-5 AR + Granger + scrambled null, (b) Gate-3 SSM-vs-baseline at the
#       λ knee. If H1/Gate-3 hold on an object-disjoint split, the shared-answer
#       leakage objection is closed.
#   S3  R6 · T4 untrained-model control on GPT-2-small (cross-family): pressure
#       battery on the TRAINED gpt2 store (never run — T1 baseline), untrained
#       collection, pressure battery on the untrained store. Tests whether the
#       "AR-on-Δ structure is substantially architectural" deflation and its
#       magnitude (~50% learned share on 410m) replicate off-Pythia.

set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY="uv run python"
CFG410=configs/pythia-410m-deduped.yaml

echo "=== S1a: R4 train-only L1 SAE retrain ($(date -Is)) ==="
$PY scripts/train_410m_saes.py --config $CFG410 --mode l1 --l1-coeff 0.5 \
    --epochs 1000 --exclude-test --desc r4-train-l1-trainonly-410md

echo "=== S1b: R4 Gate-3 Pareto on train-only sources ($(date -Is)) ==="
$PY scripts/gate3_pareto.py --config $CFG410 --desc r4-gate3-l1-trainonly-410md \
    --sources-npz data/pythia-410m-deduped/sources_trained_l1_trainonly.npz \
    --rank 32 --lambdas 0.3,1,3,10,30 --outer-iters 10 --batch-size 128 \
    --device cuda --seed 0

echo "=== S1c: R4 step4 seed-stability on train-only sources ($(date -Is)) ==="
$PY scripts/step4_sanity.py --config $CFG410 --desc r4-step4-l1-trainonly-410md \
    --sources-npz data/pythia-410m-deduped/sources_trained_l1_trainonly.npz \
    --rank 32 --lambda-l1 1.0 --outer-iters 10 --batch-size 128 \
    --device cuda --seed 0

echo "=== S2a: R5 Step-5 AR on object-disjoint split ($(date -Is)) ==="
$PY scripts/step5_ar.py --config $CFG410 --desc r5-step5-objsplit-410md \
    --split-by object

echo "=== S2b: R5 Gate-3 on object-disjoint split ($(date -Is)) ==="
$PY scripts/gate3_pareto.py --config $CFG410 --desc r5-gate3-objsplit-410md \
    --sources-npz data/pythia-410m-deduped/sources_mlsae.npz \
    --rank 32 --lambdas 0.3,1,3 --outer-iters 10 --batch-size 128 \
    --device cuda --seed 0 --split-by object

echo "=== S3a: R6 pressure battery on the TRAINED gpt2 store ($(date -Is)) ==="
$PY scripts/phase4_pressure_tests.py --config configs/gpt2-small.yaml \
    --desc r6-pressure-gpt2-trained

echo "=== S3b: R6 collect untrained GPT-2 store ($(date -Is)) ==="
$PY scripts/phase4_t4_untrained.py --config configs/gpt2-small.yaml \
    --desc r6-t4-untrained-gpt2 --source factual

echo "=== S3c: R6 pressure battery on the UNTRAINED gpt2 store ($(date -Is)) ==="
$PY scripts/phase4_pressure_tests.py --config configs/gpt2-small.yaml \
    --desc r6-pressure-gpt2-untrained --h5 data/gpt2-untrained/residual_stream.h5

echo "=== R4–R6 chain complete ($(date -Is)) ==="
