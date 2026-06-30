#!/usr/bin/env bash
# Phase 4 E2 downstream evaluation: re-run Gate-3 Pareto + seed-stability on a
# trained 410m dictionary (sources_trained_<mode>.npz), with the *exact* args of the
# recorded mlsae Gate-3/step4 runs so the only thing that changed is the dictionary.
#
#   scripts/run_e2.sh <l1|topk> [extra gate3/step4 args]
#
# Headline E2 number = step4_sanity match@0.7 (seed-stability): does the within-Pythia-410m
# L1 dictionary clear 0.7 where the TopK one sits at ~0.45? Gate-3 confirms the
# reconstruction win survives the dictionary swap. Train the dictionary first with
# scripts/train_410m_saes.py and inspect L0/recon before launching this (it is the
# expensive part: a full λ-sweep × 2 variants + a 5-fit seed/dataset sanity, ~2-3h).
set -euo pipefail

MODE="${1:?usage: run_e2.sh <l1|topk> [extra args]}"; shift || true
CFG=configs/pythia-410m-deduped.yaml
NPZ="data/pythia-410m-deduped/sources_trained_${MODE}.npz"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -f "$NPZ" ] || { echo "missing $NPZ — run scripts/train_410m_saes.py --mode $MODE first"; exit 1; }

echo "=== E2 [$MODE] Gate-3 Pareto (reconstruction vs per-layer baseline) ==="
python scripts/gate3_pareto.py --config "$CFG" --desc "ph4-e2-gate3-${MODE}-410md" \
    --sources-npz "$NPZ" --rank 32 --lambdas 0.3,1,3,10,30 --outer-iters 10 \
    --batch-size 128 --device cuda --seed 0 "$@"

echo "=== E2 [$MODE] step4 seed/dataset stability (the headline match@0.7) ==="
python scripts/step4_sanity.py --config "$CFG" --desc "ph4-e2-step4-${MODE}-410md" \
    --sources-npz "$NPZ" --rank 32 --lambda-l1 1.0 --outer-iters 10 \
    --batch-size 128 --device cuda --seed 0 "$@"

echo "=== E2 [$MODE] done — see runs/*ph4-e2-{gate3,step4}-${MODE}-410md* ==="
