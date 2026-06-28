#!/usr/bin/env bash
# S2 — extra seeds for the depth-scaling gap error bars (Phase 2 Step 5 / paper figure).
#
# Seed 0 already exists per model (the Gate-3 Pareto runs). This adds seeds 1 and 2 with each
# model's EXACT seed-0 config / sources / λ-grid so plot_depth_scaling.py can aggregate the
# matched-L0 SSM−baseline gap to mean±sd. Sequential + unattended (each gate3_pareto is its own
# process, so the GPU is freed between fits); cheapest model first so a late failure on the 410m
# doesn't cost the cheap points. Continue-on-error. ~6–7h total on the GTX 1050.
#
#     scripts/run_s2_seeds.sh        # -> runs/<date>-{xm-p160-gate3,ph3-gate3-70md,ph3-gate3-410md}-s{1,2}
set -u
LOG="runs/s2-seeds-overnight.log"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run() {  # <tag> <seed> <gate3_pareto.py args...>
  local tag="$1" seed="$2"; shift 2
  echo "=== $(date -Is)  START $tag seed=$seed ===" | tee -a "$LOG"
  if uv run python scripts/gate3_pareto.py "$@" --seed "$seed" >>"$LOG" 2>&1; then
    echo "=== $(date -Is)  DONE  $tag seed=$seed ===" | tee -a "$LOG"
  else
    echo "=== $(date -Is)  FAIL  $tag seed=$seed (rc=$?, continuing) ===" | tee -a "$LOG"
  fi
}

echo "=== $(date -Is)  S2 seed sweep started ===" | tee -a "$LOG"
for s in 1 2; do
  run p160 "$s" --config configs/pythia-160m-deduped.yaml --desc "xm-p160-gate3-s$s" \
    --sources-npz data/pythia-160m-deduped/sources_mlsae.npz \
    --rank 32 --lambdas 0.3,1,3,10,30 --outer-iters 10 --device cuda
done
for s in 1 2; do
  run p70 "$s" --config configs/pythia-70m.yaml --desc "ph3-gate3-70md-s$s" \
    --rank 32 --lambdas 0.1,0.3,1,3,10 --outer-iters 10 --device cuda
done
for s in 1 2; do
  run p410 "$s" --config configs/pythia-410m-deduped.yaml --desc "ph3-gate3-410md-s$s" \
    --sources-npz data/pythia-410m-deduped/sources_mlsae.npz \
    --rank 32 --lambdas 0.3,1,3,10,30 --outer-iters 10 --batch-size 128 --device cuda
done
echo "=== $(date -Is)  S2 seed sweep ALL DONE ===" | tee -a "$LOG"
