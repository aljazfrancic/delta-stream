#!/usr/bin/env bash
# R8 — attn-out vs MLP-out Δ split: where does the cross-layer AR structure live?
# (Optional mechanism polish from the 2026-07-02 review; pairs with the A_ℓ
# contractivity result and the k*≈1 story.)
#
# Assumes scripts/collect_stream_split.py has already written the two pseudo-stream
# stores (data/<model>-{attn,mlp}stream/residual_stream.h5). Per stream, re-run the
# identical H1 + H2 + k* diagnostics via --h5:
#   step5_ar           — AR(k) + Granger + scrambled null   (H1)
#   step1_stationarity — cross-band kernel transfer          (H2), k=5 r=256 as the
#                        factual primary run derived them (spot-check still guards)
#   step2_kstar        — per-depth effective order           (k*), rank 256
#
#   scripts/run_r8.sh          # CPU-heavy (numpy lstsq/SVD); a few hours
set -euo pipefail

CFG=configs/pythia-410m-deduped.yaml
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for STREAM in attn mlp; do
    H5=data/pythia-410m-deduped-${STREAM}stream/residual_stream.h5

    echo "=== R8 [$STREAM 1/3] step5 AR + Granger + scrambled null ==="
    python scripts/step5_ar.py --config "$CFG" --h5 "$H5" \
        --desc "r8-step5-${STREAM}-410md" --seed 0

    echo "=== R8 [$STREAM 2/3] step1 stationarity (k=5, r=256) ==="
    python scripts/step1_stationarity.py --config "$CFG" --h5 "$H5" \
        --k 5 --rank 256 --desc "r8-stationarity-${STREAM}-410md" --seed 0

    echo "=== R8 [$STREAM 3/3] step2 k* (rank 256) ==="
    python scripts/step2_kstar.py --config "$CFG" --h5 "$H5" \
        --rank 256 --desc "r8-kstar-${STREAM}-410md" --seed 0
done

echo "=== R8 done — compare runs/*r8-{step5,stationarity,kstar}-{attn,mlp}-410md ==="
