#!/usr/bin/env bash
# ProgC restart — gate3-24l was OOM-killed at iter 7 of its last λ block (a second
# OOM when the R8 rerun's rank-512 bootstrap competed for host RAM). gate3_pareto
# only persists pareto.csv at the end, so nothing survived. The two trained-L1 Qwen
# dictionaries ARE cached (sources_trained_l1_{24l,12l}.npz), so rerun only steps 3-4.
#
# MUST run with the box to itself — do not overlap with the R8 rank-512 bootstrap.
set -euo pipefail

CFG=configs/qwen2-0.5b.yaml
H5=data/Qwen/Qwen2-0.5B/residual_stream.h5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== ProgC [3/4, restart] Gate-3 Pareto, full 24-layer coverage ==="
python scripts/gate3_pareto.py --config "$CFG" --desc progc-gate3-qwen-24l \
    --h5 "$H5" --sources-npz data/Qwen/Qwen2-0.5B/sources_trained_l1_24l.npz \
    --rank 32 --lambdas 0.3,1,3,10 --outer-iters 10 --batch-size 128 \
    --device cuda --seed 0

echo "=== ProgC [4/4, restart] Gate-3 Pareto, 12-layer prefix (matched dictionary) ==="
python scripts/gate3_pareto.py --config "$CFG" --desc progc-gate3-qwen-12l \
    --h5 "$H5" --sources-npz data/Qwen/Qwen2-0.5B/sources_trained_l1_12l.npz \
    --rank 32 --lambdas 0.3,1,3,10 --outer-iters 10 --batch-size 128 \
    --device cuda --seed 0

echo "=== ProgC done — knee(24L) vs knee(12L) from runs/*progc-gate3-qwen-* ==="
