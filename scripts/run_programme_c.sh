#!/usr/bin/env bash
# Programme C — fix the fig06 Qwen point ON-BOX (it was misfiled as cloud, like E3:
# the activations are cached and nothing here needs >3 GB).
#
# fig06 plots Qwen2-0.5B at x=12 because the public chanind L1 suite covers only
# blocks 0..11 of the 24-layer model; the 4th-pass annotation admits the figure
# cannot distinguish "gap tracks covered depth" from "gap tracks model depth".
# This resolves it with a *dictionary-controlled coverage contrast*: train our own
# L1 SAEs (R10/E2 protocol, R4-clean --exclude-test) on the SAME cached Qwen store
# at (a) all 24 layers and (b) the first 12 layers only, then run the identical
# Gate-3 Pareto on each. Same model, same rows, same dictionary protocol — the only
# thing that differs is SAE coverage, so knee(24L) vs knee(12L) is the coverage
# effect in isolation. Seed 0 first; add seeds 1-2 for error bars if the contrast
# is marginal.
#
#   scripts/run_programme_c.sh        # overnight on the GTX 1050
set -euo pipefail

CFG=configs/qwen2-0.5b.yaml
H5=data/Qwen/Qwen2-0.5B/residual_stream.h5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== ProgC [1/4] train L1 SAEs on ALL 24 Qwen layers (R4-clean) ==="
python scripts/train_410m_saes.py --mode l1 --l1-coeff 0.5 --config "$CFG" \
    --h5 "$H5" --exclude-test \
    --out-npz data/Qwen/Qwen2-0.5B/sources_trained_l1_24l.npz \
    --desc progc-train-l1-qwen-24l --seed 0

echo "=== ProgC [2/4] train L1 SAEs on the FIRST 12 layers (coverage control) ==="
python scripts/train_410m_saes.py --mode l1 --l1-coeff 0.5 --config "$CFG" \
    --h5 "$H5" --exclude-test --max-layers 12 \
    --out-npz data/Qwen/Qwen2-0.5B/sources_trained_l1_12l.npz \
    --desc progc-train-l1-qwen-12l --seed 0

echo "=== ProgC [3/4] Gate-3 Pareto, full 24-layer coverage ==="
python scripts/gate3_pareto.py --config "$CFG" --desc progc-gate3-qwen-24l \
    --h5 "$H5" --sources-npz data/Qwen/Qwen2-0.5B/sources_trained_l1_24l.npz \
    --rank 32 --lambdas 0.3,1,3,10 --outer-iters 10 --batch-size 128 \
    --device cuda --seed 0

echo "=== ProgC [4/4] Gate-3 Pareto, 12-layer prefix (matched dictionary) ==="
python scripts/gate3_pareto.py --config "$CFG" --desc progc-gate3-qwen-12l \
    --h5 "$H5" --sources-npz data/Qwen/Qwen2-0.5B/sources_trained_l1_12l.npz \
    --rank 32 --lambdas 0.3,1,3,10 --outer-iters 10 --batch-size 128 \
    --device cuda --seed 0

echo "=== ProgC done — knee(24L) vs knee(12L) from runs/*progc-gate3-qwen-* ==="
