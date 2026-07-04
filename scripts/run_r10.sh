#!/usr/bin/env bash
# R10 — T4-recon on GPT-2-small: is the SSM *reconstruction* win architectural or
# learned, cross-family? (4th-pass action plan; R6 covered only the AR part and found
# the learned AR share collapses to ~5% on GPT-2 — if the recon learned-share also
# collapses, the 410m "≈half learned" sentence is single-model.)
#
# Exact 410m T4-recon recipe (runs/2026-06-30-ph4-t4recon-train-l1-untrained-410md +
# runs/2026-07-01-ph4-t4recon-gate3-l1-untrained-410md), applied to the GPT-2 stores:
# train a matched L1 dictionary on the TRAINED and the UNTRAINED (random-init, R6)
# residual store, then run the identical Gate-3 Pareto (SSM vs use_state=False) on
# each. Both arms use our own trained-L1 dictionary (not res-jb) so the trained vs
# untrained contrast is dictionary-protocol-matched.
#
#   scripts/run_r10.sh            # ~1.5-2h on the GTX 1050
set -euo pipefail

CFG=configs/gpt2-small.yaml
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== R10 [1/4] train L1 SAE on the TRAINED GPT-2 store ==="
python scripts/train_410m_saes.py --mode l1 --l1-coeff 0.5 --config "$CFG" \
    --h5 data/gpt2/residual_stream.h5 \
    --out-npz data/gpt2/sources_trained_l1.npz \
    --desc r10-train-l1-gpt2 --seed 0

echo "=== R10 [2/4] train L1 SAE on the UNTRAINED GPT-2 store (R6 collection) ==="
python scripts/train_410m_saes.py --mode l1 --l1-coeff 0.5 --config "$CFG" \
    --h5 data/gpt2-untrained/residual_stream.h5 \
    --out-npz data/gpt2-untrained/sources_trained_l1.npz \
    --desc r10-train-l1-gpt2-untrained --seed 0

echo "=== R10 [3/4] Gate-3 Pareto on the TRAINED store ==="
python scripts/gate3_pareto.py --config "$CFG" --desc r10-gate3-l1-gpt2 \
    --h5 data/gpt2/residual_stream.h5 \
    --sources-npz data/gpt2/sources_trained_l1.npz \
    --rank 32 --lambdas 0.3,1,3,10 --outer-iters 10 --batch-size 128 \
    --device cuda --seed 0

echo "=== R10 [4/4] Gate-3 Pareto on the UNTRAINED store ==="
python scripts/gate3_pareto.py --config "$CFG" --desc r10-gate3-l1-gpt2-untrained \
    --h5 data/gpt2-untrained/residual_stream.h5 \
    --sources-npz data/gpt2-untrained/sources_trained_l1.npz \
    --rank 32 --lambdas 0.3,1,3,10 --outer-iters 10 --batch-size 128 \
    --device cuda --seed 0

echo "=== R10 done — compare SSM−baseline gaps per λ across runs/*r10-gate3-l1-gpt2* ==="
