#!/usr/bin/env bash
# v1.1 post-hoc E1 + E2 — what does "seed" mean in the TopK stability claim?
#
# v1.0's match@0.7 re-seeds the SSM over ONE trained dictionary per sparsity mechanism.
# This trains the missing dictionary seeds and measures both readings:
#   E1  dictionary-level: do SAEs trained from different seeds find the same decoder
#       directions?                      (scripts/sae_seed_stability.py, no GPU)
#   E2  SSM-level, n = 3 draws: is the recorded SSM-refit statistic (1.00 L1 / 0.08 TopK)
#       a property of the mechanism or of one dictionary draw?
#                                        (scripts/step4_sanity.py per draw, then
#                                         scripts/ssm_draw_stability.py)
# Verdict criteria are frozen in runs/v11-seed-stability/memo.md — commit that memo
# BEFORE launching the full run.
#
#   scripts/run_v11_seeds.sh --smoke    # 2 layers x 50 epochs, end to end (minutes)
#   scripts/run_v11_seeds.sh            # full run; ~9-11 h on the GTX 1050 (measure on
#                                       #   the smoke test), sequential, one log
#   V11_DATE=2026-11-15 scripts/run_v11_seeds.sh   # resume a run started on that date
#
# Sequential + unattended, continue-on-error, single log (the run_s2_seeds.sh pattern),
# with ONE hard stop: if the retrained seed-0 dictionaries do not reproduce the recorded
# E2 tables, nothing downstream can be trusted, so the run aborts there.
#
# Recorded seed-0 recipe (runs/2026-06-29-ph4-e2-train-{l1,topk}-410md/tables/*.json;
# scripts/run_review_r4_r6.sh for the command line): M 8192, 1000 epochs, standardize on,
# lr 1e-3 (the trainer default, never overridden), L1 --l1-coeff 0.5, TopK --k 32.
# Every dictionary is written to an explicit per-seed path under data/<model>/v11/, so
# the recorded data/<model>/sources_trained_<mode>.npz is never touched, and an existing
# per-seed file is reused, never overwritten.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CFG=configs/pythia-410m-deduped.yaml
MODEL_DIR=data/pythia-410m-deduped
DATE="${V11_DATE:-$(date +%F)}"
SMOKE=0
for arg in "$@"; do
  case "$arg" in
    --smoke) SMOKE=1 ;;
    *) echo "unknown arg: $arg (known: --smoke)"; exit 2 ;;
  esac
done

if [ "$SMOKE" = 1 ]; then
  TAG=v11-smoke; OUT="$MODEL_DIR/v11-smoke"; LOG=runs/v11-smoke.log
  TRAIN_EXTRA=(--max-layers 2 --epochs 50); STEP4_EXTRA=(--outer-iters 2)
else
  TAG=v11; OUT="$MODEL_DIR/v11"; LOG=runs/v11-seeds-overnight.log
  TRAIN_EXTRA=(--epochs 1000); STEP4_EXTRA=(--outer-iters 10)
fi
REF_L1=runs/2026-06-29-ph4-e2-train-l1-410md/tables/sources_trained_l1.json
REF_TOPK=runs/2026-06-29-ph4-e2-train-topk-410md/tables/sources_trained_topk.json
REC_STEP4_L1=runs/2026-06-29-ph4-e2-step4-l1-410md
REC_STEP4_TOPK=runs/2026-06-30-ph4-e2-step4-topk-410md

for f in residual_stream.h5 filtered_triples.parquet; do
  [ -f "$MODEL_DIR/$f" ] || {
    echo "missing $MODEL_DIR/$f — the activation store is not on this machine."
    echo "Restore it (or get explicit sign-off to re-collect); do not re-collect silently."
    exit 1; }
done
mkdir -p "$OUT"

step() {  # <label> <command...>
  local label="$1"; shift
  echo "=== $(date -Is)  START $label ===" | tee -a "$LOG"
  if "$@" >>"$LOG" 2>&1; then
    echo "=== $(date -Is)  DONE  $label ===" | tee -a "$LOG"
  else
    echo "=== $(date -Is)  FAIL  $label (rc=$?, continuing) ===" | tee -a "$LOG"
  fi
}

npz() { echo "$OUT/sources_trained_${1}_s${2}.npz"; }                 # <mode> <seed>
step4_dir() { echo "runs/${DATE}-${TAG}-e2-step4-${1}-s${2}-410md"; }  # <mode> <seed>

train() {  # <mode> <seed> <mode args...>
  local mode="$1" s="$2"; shift 2
  if [ -e "$(npz "$mode" "$s")" ]; then
    echo "=== $(date -Is)  SKIP  train $mode seed=$s ($(npz "$mode" "$s") exists; reused, never overwritten) ===" | tee -a "$LOG"
    return
  fi
  step "train $mode seed=$s" uv run python scripts/train_410m_saes.py --config "$CFG" \
    --mode "$mode" "$@" "${TRAIN_EXTRA[@]}" --lr 1e-3 --seed "$s" \
    --out-npz "$(npz "$mode" "$s")" --out "runs/${DATE}-${TAG}-train-${mode}-s${s}-410md"
}

step4() {  # <mode> <dictionary seed> — the exact recorded E2 arguments (scripts/run_e2.sh)
  local mode="$1" s="$2"
  if [ -e "$(step4_dir "$mode" "$s")/tables/sanity_stats.json" ]; then
    echo "=== $(date -Is)  SKIP  E2 step4 $mode dict-seed=$s (already done) ===" | tee -a "$LOG"
    return
  fi
  step "E2 step4 $mode dict-seed=$s" uv run python scripts/step4_sanity.py --config "$CFG" \
    --out "$(step4_dir "$mode" "$s")" --sources-npz "$(npz "$mode" "$s")" \
    --rank 32 --lambda-l1 1.0 "${STEP4_EXTRA[@]}" --batch-size 128 --device cuda --seed 0
}

echo "=== $(date -Is)  v1.1 seed sweep started (tag $TAG, date $DATE) ===" | tee -a "$LOG"

# --- 1. seed 0 of both mechanisms, then the reproduction gate ---------------------
train l1 0 --l1-coeff 0.5
train topk 0 --k 32
if [ "$SMOKE" = 0 ]; then
  echo "=== $(date -Is)  seed-0 reproduction gate ===" | tee -a "$LOG"
  if ! uv run python - "$(npz l1 0)" "$REF_L1" "$(npz topk 0)" "$REF_TOPK" 2>&1 <<'EOF' | tee -a "$LOG"
import sys

import numpy as np

from layer_dynamics.diagnostics.sae_seed_stability import reproduction_check

ok = True
for npz, ref in zip(sys.argv[1::2], sys.argv[2::2]):
    with np.load(npz) as z:
        r = reproduction_check(z["alive_count"], z["recon_r2"], ref)
    ok = ok and r["ok"]
    print(f"  {npz}: {'OK' if r['ok'] else 'FAILED'}  M* {r.get('M_star')} vs "
          f"{r.get('ref_M_star')} (identical alive counts: {r.get('alive_identical')}), "
          f"max per-layer alive rel diff {r.get('max_alive_rel_diff', float('nan')):.4f}, "
          f"mean recon R² diff {r.get('r2_mean_abs_diff', float('nan')):.5f}")
sys.exit(0 if ok else 1)
EOF
  then
    echo "=== $(date -Is)  ABORT: retrained seed 0 does not reproduce the recorded E2 dictionaries — store or hyperparameters differ; nothing downstream is trustworthy ===" | tee -a "$LOG"
    exit 1
  fi
fi

# --- 2. seeds 1 and 2 --------------------------------------------------------------
for s in 1 2; do
  train l1 "$s" --l1-coeff 0.5
  train topk "$s" --k 32
done

# --- 3. E1: dictionary-level seed stability (CPU, minutes) --------------------------
E1_REFS=()
[ "$SMOKE" = 0 ] && E1_REFS=(--l1-ref "$REF_L1" --topk-ref "$REF_TOPK")
step "E1 dictionary-level seed stability" uv run python scripts/sae_seed_stability.py \
  --config "$CFG" --out "runs/${DATE}-${TAG}-e1-sae-seed-stability" --seed 0 \
  --l1 "$(npz l1 0)" "$(npz l1 1)" "$(npz l1 2)" \
  --topk "$(npz topk 0)" "$(npz topk 1)" "$(npz topk 2)" "${E1_REFS[@]}"

# --- 4. E2: SSM-refit stability on each new draw (new seeds first) ------------------
for s in 1 2; do
  step4 l1 "$s"
  step4 topk "$s"
done
step4 l1 0      # seed-0 retrain: should reproduce the recorded 1.00 / 0.08; not a new draw
step4 topk 0

if [ "$SMOKE" = 1 ]; then  # smoke draws are 2-layer fits; the recorded runs are not comparable
  E2_DRAWS=(--draw "l1:0:$(step4_dir l1 0)" --draw "topk:0:$(step4_dir topk 0)")
else
  E2_DRAWS=(--draw "l1:0:$REC_STEP4_L1" --draw "topk:0:$REC_STEP4_TOPK"
            --check "l1:0:$(step4_dir l1 0)" --check "topk:0:$(step4_dir topk 0)")
fi
step "E2 draw summary" uv run python scripts/ssm_draw_stability.py --config "$CFG" \
  --out "runs/${DATE}-${TAG}-e2-ssm-draw-stability" --seed 0 "${E2_DRAWS[@]}" \
  --draw "l1:1:$(step4_dir l1 1)" --draw "l1:2:$(step4_dir l1 2)" \
  --draw "topk:1:$(step4_dir topk 1)" --draw "topk:2:$(step4_dir topk 2)"

echo "=== $(date -Is)  v1.1 seed sweep ALL DONE — see runs/${DATE}-${TAG}-{e1,e2}-* ===" | tee -a "$LOG"
