#!/usr/bin/env bash
# Resume of run_review_r0_r3.sh after the S4 rank-64 CLT OOM (2026-07-03 ~00:00 UTC).
#
# The rank-64 CLT backward at batch 256 needs ~1.3 GB of retained per-target
# activations on top of the 104.8M-param Adam state — over the 3 GB box. The
# rank-64 per_layer/ssm fits COMPLETED before the crash (in runs/review-r0r3.log;
# ssm r64 best test R² 0.8142 at λ_reg=1, the grid edge again). This script:
#
#   S4'  rank-64 ssm + clt_full at batch 128 (halved activation memory) with the
#        grid extended to λ=3 (both models landed on the λ=1 edge at rank 32/64);
#        ssm re-run at batch 128 so the rank-64 pair is internally consistent.
#   S5   rank-16 sweep (unchanged from the original chain; batch 256 is safe).
#   S6   R3 sample-efficiency (unchanged).
#   →    then chains straight into run_review_r4_r6.sh (R4–R6 + fig06 seeds
#        + the S0 λ-probe) so the whole programme finishes unattended.

set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY="uv run python"
CFG410=configs/pythia-410m-deduped.yaml
SRC410=data/pythia-410m-deduped/sources_mlsae.npz

echo "=== S4'': rank-64 ssm+clt, batch 64, grid {0.3, 1, 3} ($(date -Is)) ==="
# batch 128 still OOMs on the CLT backward (104.8M params: 1.68 GB Adam block +
# ~0.68 GB retained per-target activations). Batch 64 halves the activations again
# (~2.4 GB total). ssm re-run at the same batch so the CI pair shares a protocol.
$PY scripts/phase4_baseline_clt.py --config $CFG410 --sources-npz $SRC410 \
    --desc r2-clt-410md-rank64 --rank 64 --epochs 180 --batch-size 64 \
    --models ssm clt_full --reg-grid 0.3 1 3

echo "=== S5: rank-16 sweep ($(date -Is)) ==="
$PY scripts/phase4_baseline_clt.py --config $CFG410 --sources-npz $SRC410 \
    --desc r2-clt-410md-rank16 --rank 16 --epochs 180 --reg-grid 0.01 0.1 0.3 1

echo "=== S6a: R3 n_train=600 ($(date -Is)) ==="
$PY scripts/phase4_baseline_clt.py --config $CFG410 --sources-npz $SRC410 \
    --desc r3-clt-410md-n600 --rank 32 --epochs 360 --n-train 600 \
    --models ssm clt_full --reg-grid 0.1 0.3 1

echo "=== S6b: R3 n_train=300 ($(date -Is)) ==="
$PY scripts/phase4_baseline_clt.py --config $CFG410 --sources-npz $SRC410 \
    --desc r3-clt-410md-n300 --rank 32 --epochs 720 --n-train 300 \
    --models ssm clt_full --reg-grid 0.1 0.3 1

echo "=== R0–R3 chain complete; handing over to R4–R6 ($(date -Is)) ==="
exec bash scripts/run_review_r4_r6.sh
