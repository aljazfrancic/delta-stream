#!/usr/bin/env bash
# R8 restart — the 2026-07-05 chain was OOM-killed mid MLP-stationarity bootstrap
# (concurrent replot.py + gate3 GPU job). Attn trio + MLP step5 completed and are
# on disk; this reruns only the two missing MLP steps (same descs -> same run dirs).
set -euo pipefail

CFG=configs/pythia-410m-deduped.yaml
H5=data/pythia-410m-deduped-mlpstream/residual_stream.h5

echo "=== R8 [mlp 2/3, restart] step1 stationarity (k=5, r=256) ==="
python scripts/step1_stationarity.py --config "$CFG" --h5 "$H5" \
    --k 5 --rank 256 --desc "r8-stationarity-mlp-410md" --seed 0

echo "=== R8 [mlp 3/3, restart] step2 k* (rank 256) ==="
python scripts/step2_kstar.py --config "$CFG" --h5 "$H5" \
    --rank 256 --desc "r8-kstar-mlp-410md" --seed 0

echo "=== R8 done — compare runs/*r8-{step5,stationarity,kstar}-{attn,mlp}-410md ==="
