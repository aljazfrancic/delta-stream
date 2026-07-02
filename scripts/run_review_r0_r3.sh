#!/usr/bin/env bash
# Review items R0–R3 (2026-07-02 code-level review) — serial GPU chain for the 3 GB box.
#
#   S1  R2/R0 · 410m rank-32 CLT-vs-SSM re-run: extended λ_reg grid (E3's best hit the
#       old grid edge 0.1), convergence curve, per-example SS → paired bootstrap CI on
#       the E3 "tie" (the R0-E3 number).
#   S2  R0 · Gate-3 SSM-vs-baseline gap CIs at the canonical operating point, from the
#       saved checkpoints (mlsae + trained-L1), baseline refit with the identical
#       Gate-3 protocol.
#   S3  R1 · E3 depth series on the cached cross-model sources: GPT-2-small (12L, L1),
#       Pythia-160m (12L, TopK), Pythia-70m-deduped (6L, TopK, sources built on the
#       fly) → fig08 becomes a real depth series instead of 2 confounded points.
#   S4  R2 · rank-64 sweep (all three families).  Rank 128 = 3.35 GB Adam state > 3 GB
#       box → documented infeasible on-box; 64 is the on-box ceiling.
#   S5  R2 · rank-16 sweep (the other direction of the rank axis).
#   S6  R3 · sample-efficiency: n_train ∈ {600, 300} at rank 32 (ssm + clt only),
#       epochs scaled inversely so total gradient steps stay ~constant.
#
# Each stage fails hard (set -e) so a broken stage stops the chain for inspection
# instead of burning GPU on garbage.

set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY="uv run python"
CFG410=configs/pythia-410m-deduped.yaml
SRC410=data/pythia-410m-deduped/sources_mlsae.npz

echo "=== S1: 410m rank-32, extended reg grid + convergence + CI ($(date -Is)) ==="
$PY scripts/phase4_baseline_clt.py --config $CFG410 --sources-npz $SRC410 \
    --desc r2-clt-410md-rank32 --rank 32 --epochs 180 \
    --reg-grid 0.01 0.1 0.3 1 --eval-every 30

echo "=== S2a: Gate-3 gap CI, mlsae checkpoint ($(date -Is)) ==="
$PY scripts/r0_bootstrap_ci.py --config $CFG410 --desc r0-gate3-ci-mlsae \
    --checkpoint-dir runs/2026-06-26-ph4-checkpoint-410md --sources-npz $SRC410

echo "=== S2b: Gate-3 gap CI, trained-L1 checkpoint ($(date -Is)) ==="
$PY scripts/r0_bootstrap_ci.py --config $CFG410 --desc r0-gate3-ci-l1 \
    --checkpoint-dir runs/2026-06-30-ph4-e2-checkpoint-l1-410md \
    --sources-npz data/pythia-410m-deduped/sources_trained_l1.npz

echo "=== S3a: R1 GPT-2-small (12L, res-jb L1) ($(date -Is)) ==="
$PY scripts/phase4_baseline_clt.py --config configs/gpt2-small.yaml \
    --sources-npz data/gpt2/sources_resjb.npz \
    --desc r1-clt-gpt2 --rank 32 --epochs 180 --reg-grid 0.01 0.1 0.3 1

echo "=== S3b: R1 Pythia-160m (12L, mlsae TopK) ($(date -Is)) ==="
$PY scripts/phase4_baseline_clt.py --config configs/pythia-160m-deduped.yaml \
    --sources-npz data/pythia-160m-deduped/sources_mlsae.npz \
    --desc r1-clt-p160 --rank 32 --epochs 180 --reg-grid 0.01 0.1 0.3 1

echo "=== S3c: R1 Pythia-70m-deduped (6L, EleutherAI TopK, on-the-fly sources) ($(date -Is)) ==="
$PY scripts/phase4_baseline_clt.py --config configs/pythia-70m.yaml \
    --sae-repo EleutherAI/sae-pythia-70m-deduped-32k \
    --desc r1-clt-p70 --rank 32 --epochs 180 --reg-grid 0.01 0.1 0.3 1

echo "=== S4: R2 rank-64 sweep, 410m ($(date -Is)) ==="
$PY scripts/phase4_baseline_clt.py --config $CFG410 --sources-npz $SRC410 \
    --desc r2-clt-410md-rank64 --rank 64 --epochs 180 --reg-grid 0.1 0.3 1

echo "=== S5: R2 rank-16 sweep, 410m ($(date -Is)) ==="
$PY scripts/phase4_baseline_clt.py --config $CFG410 --sources-npz $SRC410 \
    --desc r2-clt-410md-rank16 --rank 16 --epochs 180 --reg-grid 0.01 0.1 0.3

echo "=== S6a: R3 n_train=600 ($(date -Is)) ==="
$PY scripts/phase4_baseline_clt.py --config $CFG410 --sources-npz $SRC410 \
    --desc r3-clt-410md-n600 --rank 32 --epochs 360 --n-train 600 \
    --models ssm clt_full --reg-grid 0.1 0.3 1

echo "=== S6b: R3 n_train=300 ($(date -Is)) ==="
$PY scripts/phase4_baseline_clt.py --config $CFG410 --sources-npz $SRC410 \
    --desc r3-clt-410md-n300 --rank 32 --epochs 720 --n-train 300 \
    --models ssm clt_full --reg-grid 0.1 0.3 1

echo "=== R0–R3 chain complete ($(date -Is)) ==="
