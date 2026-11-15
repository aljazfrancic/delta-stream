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
# Verdict criteria are frozen in runs/v11-seed-stability/memo.md, committed before any run.
#
#   scripts/run_v11_seeds.sh --smoke    # minutes: 3 layers x 50 epochs end to end, plus a
#                                       #   full-length seed-0 preview of the reproduction
#                                       #   gate on the first two layers. Writes only to
#                                       #   git-ignored paths (runs/v11-smoke/, data/).
#   scripts/run_v11_seeds.sh            # the full run, sequential, one log
#   V11_DATE=2026-11-15 scripts/run_v11_seeds.sh   # resume a run started on that date
#
# Launch the full run detached, so it survives the terminal and blocks sleep:
#
#   systemd-inhibit --what=sleep:idle --why="v1.1 seed sweep" \
#     setsid nohup bash scripts/run_v11_seeds.sh > runs/v11-seeds-launch.out 2>&1 &
#   tail -f runs/v11-seeds-overnight.log        # progress: one START/DONE line per step
#
# Sequential + unattended, continue-on-error, single log (the run_s2_seeds.sh pattern),
# with two hard stops, because nothing downstream means anything if either fails:
#   (a) the activation store must match the tracked records (scripts/verify_store.py);
#   (b) the retrained seed-0 dictionaries must reproduce the recorded E2 tables.
# Every step is skipped if its output already exists, so a stopped run resumes where it
# left off (use V11_DATE if the date has changed).
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

REF_L1=runs/2026-06-29-ph4-e2-train-l1-410md/tables/sources_trained_l1.json
REF_TOPK=runs/2026-06-29-ph4-e2-train-topk-410md/tables/sources_trained_topk.json
REC_STEP4_L1=runs/2026-06-29-ph4-e2-step4-l1-410md
REC_STEP4_TOPK=runs/2026-06-30-ph4-e2-step4-topk-410md
REF_DATASET=runs/2026-06-21-ph1-dataset-410md/tables/stats.json
REF_COLLECT=runs/2026-06-21-ph1-collect-410md/tables/collect_stats.json
REF_RIDGE=runs/2026-06-21-ph1-step3-410md/tables/ridge_r2.csv

if [ "$SMOKE" = 1 ]; then
  # runs/v11-smoke/<name>/ sits two levels down, so the repo's ignore rules drop all of
  # it: a smoke test leaves the working tree clean for the real run's provenance.
  OUT="$MODEL_DIR/v11-smoke"; LOG=runs/v11-smoke.log
  TRAIN_EXTRA=(--max-layers 3 --epochs 50); STEP4_EXTRA=(--outer-iters 2)
  rundir() { echo "runs/v11-smoke/$1"; }
else
  OUT="$MODEL_DIR/v11"; LOG=runs/v11-seeds-overnight.log
  TRAIN_EXTRA=(--epochs 1000); STEP4_EXTRA=(--outer-iters 10)
  rundir() { echo "runs/${DATE}-v11-$1"; }
fi

# --- preflight ------------------------------------------------------------------------
for f in residual_stream.h5 filtered_triples.parquet; do
  [ -f "$MODEL_DIR/$f" ] || {
    echo "missing $MODEL_DIR/$f — the activation store is not on this machine."
    echo "Restore it (or get explicit sign-off to re-collect); do not re-collect silently."
    exit 1; }
done
for f in "$REF_L1" "$REF_TOPK" "$REC_STEP4_L1/tables/sanity_stats.json" \
         "$REC_STEP4_TOPK/tables/sanity_stats.json" "$REF_DATASET" "$REF_COLLECT" "$REF_RIDGE"; do
  [ -f "$f" ] || { echo "missing tracked reference $f"; exit 1; }
done
uv run python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" || {
  echo "CUDA is not available — this run needs the GPU (a CPU run would take days)"; exit 1; }
mkdir -p "$OUT" runs

FAILED=()
step() {  # <label> <command...>
  local label="$1"; shift
  echo "=== $(date -Is)  START $label ===" | tee -a "$LOG"
  if "$@" >>"$LOG" 2>&1; then
    echo "=== $(date -Is)  DONE  $label ===" | tee -a "$LOG"
  else
    echo "=== $(date -Is)  FAIL  $label (rc=$?, continuing) ===" | tee -a "$LOG"
    FAILED+=("$label")
  fi
}
note() { echo "=== $(date -Is)  $* ===" | tee -a "$LOG"; }

npz() { echo "$OUT/sources_trained_${1}_s${2}.npz"; }       # <mode> <seed>
step4_dir() { rundir "e2-step4-${1}-s${2}-410md"; }          # <mode> <seed>

train() {  # <mode> <seed> <mode args...>
  local mode="$1" s="$2"; shift 2
  if [ -e "$(npz "$mode" "$s")" ]; then
    note "SKIP  train $mode seed=$s ($(npz "$mode" "$s") exists; reused, never overwritten)"
    return
  fi
  step "train $mode seed=$s" uv run python scripts/train_410m_saes.py --config "$CFG" \
    --mode "$mode" "$@" "${TRAIN_EXTRA[@]}" --lr 1e-3 --seed "$s" \
    --out-npz "$(npz "$mode" "$s")" --out "$(rundir "train-${mode}-s${s}-410md")"
}

step4() {  # <mode> <dictionary seed> — the exact recorded E2 arguments (scripts/run_e2.sh)
  local mode="$1" s="$2"
  if [ -e "$(step4_dir "$mode" "$s")/tables/sanity_stats.json" ]; then
    note "SKIP  E2 step4 $mode dict-seed=$s (already done)"
    return
  fi
  step "E2 step4 $mode dict-seed=$s" uv run python scripts/step4_sanity.py --config "$CFG" \
    --out "$(step4_dir "$mode" "$s")" --sources-npz "$(npz "$mode" "$s")" \
    --rank 32 --lambda-l1 1.0 "${STEP4_EXTRA[@]}" --batch-size 128 --device cuda --seed 0
}

reproduces() {  # <npz> <reference json> ... — per-layer alive counts + recon R² vs tracked
  uv run python - "$@" <<'EOF'
import json
import sys

import numpy as np

from layer_dynamics.diagnostics.sae_seed_stability import reproduction_check

ok = True
for npz, ref_path in zip(sys.argv[1::2], sys.argv[2::2]):
    with open(ref_path, encoding="utf-8") as fh:
        ref = json.load(fh)
    with np.load(npz) as z:
        alive, r2 = z["alive_count"], z["recon_r2"]
    n = len(alive)                       # a preview covers only the first layers
    part = {**ref, "alive_count": ref["alive_count"][:n], "recon_r2": ref["recon_r2"][:n],
            "M_star": int(max(ref["alive_count"][:n]))}
    r = reproduction_check(alive, r2, part)
    ok = ok and r["ok"]
    print(f"  {npz}: {'OK' if r['ok'] else 'FAILED'}  layers 1..{n - 1}: alive "
          f"{[int(a) for a in alive[1:]]} vs recorded {part['alive_count'][1:]} "
          f"(identical: {r.get('alive_identical')}, max rel diff "
          f"{r.get('max_alive_rel_diff', float('nan')):.4f}); mean recon R² diff "
          f"{r.get('r2_mean_abs_diff', float('nan')):.2e}, max per-layer "
          f"{r.get('max_r2_abs_diff', float('nan')):.2e}")
sys.exit(0 if ok else 1)
EOF
}

note "v1.1 seed sweep started ($([ "$SMOKE" = 1 ] && echo smoke || echo full), date $DATE, commit $(git rev-parse --short HEAD)$([ -n "$(git status --porcelain)" ] && echo ' DIRTY'))"

# --- 0. the store must be the one the tracked tables came from -----------------------
if [ "$SMOKE" = 0 ]; then
  verified=$(grep -ls '"ok": true' runs/*-v11-recollect-verify-410md/tables/verify_stats.json 2>/dev/null | head -1)
  if [ -n "$verified" ]; then
    note "store already verified: $verified"
  else
    step "verify the activation store against the tracked records" \
      uv run python scripts/verify_store.py --config "$CFG" \
        --out "$(rundir recollect-verify-410md)" --seed 0 \
        --dataset-ref "$REF_DATASET" --collect-ref "$REF_COLLECT" --ridge-ref "$REF_RIDGE"
    grep -qs '"ok": true' "$(rundir recollect-verify-410md)/tables/verify_stats.json" || {
      note "ABORT: the activation store does not match the tracked records"; exit 1; }
  fi
fi

# --- 1. seed 0 of both mechanisms, then the reproduction gate ---------------------
train l1 0 --l1-coeff 0.5
train topk 0 --k 32
if [ "$SMOKE" = 0 ]; then
  note "seed-0 reproduction gate"
  if ! reproduces "$(npz l1 0)" "$REF_L1" "$(npz topk 0)" "$REF_TOPK" 2>&1 | tee -a "$LOG"; then
    note "ABORT: retrained seed 0 does not reproduce the recorded E2 dictionaries — store or hyperparameters differ; nothing downstream is trustworthy"
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
  --config "$CFG" --out "$(rundir e1-sae-seed-stability)" --seed 0 \
  --l1 "$(npz l1 0)" "$(npz l1 1)" "$(npz l1 2)" \
  --topk "$(npz topk 0)" "$(npz topk 1)" "$(npz topk 2)" "${E1_REFS[@]}"

# --- 4. E2: SSM-refit stability on each new draw (new seeds first) ------------------
for s in 1 2; do
  step4 l1 "$s"
  step4 topk "$s"
done
step4 l1 0      # seed-0 retrain: should reproduce the recorded 1.00 / 0.08; not a new draw
step4 topk 0

if [ "$SMOKE" = 1 ]; then  # smoke draws are 3-layer fits; the recorded runs are not comparable
  E2_DRAWS=(--draw "l1:0:$(step4_dir l1 0)" --draw "topk:0:$(step4_dir topk 0)")
else
  E2_DRAWS=(--draw "l1:0:$REC_STEP4_L1" --draw "topk:0:$REC_STEP4_TOPK"
            --check "l1:0:$(step4_dir l1 0)" --check "topk:0:$(step4_dir topk 0)")
fi
step "E2 draw summary" uv run python scripts/ssm_draw_stability.py --config "$CFG" \
  --out "$(rundir e2-ssm-draw-stability)" --seed 0 "${E2_DRAWS[@]}" \
  --draw "l1:1:$(step4_dir l1 1)" --draw "l1:2:$(step4_dir l1 2)" \
  --draw "topk:1:$(step4_dir topk 1)" --draw "topk:2:$(step4_dir topk 2)"

# --- smoke only: full-length preview of the seed-0 reproduction gate -----------------
# The 50-epoch dictionaries above test the plumbing, not the gate. Train seed 0 at the
# real 1,000 epochs on the first two trained layers and compare with the tracked tables,
# so a store or hyperparameter mismatch shows up now and not 90 minutes into the night.
if [ "$SMOKE" = 1 ]; then
  for mode in l1 topk; do
    p="$OUT/sources_trained_${mode}_s0_gatepreview.npz"
    margs=(--l1-coeff 0.5); [ "$mode" = topk ] && margs=(--k 32)
    [ -e "$p" ] || step "gate preview: $mode seed 0, 1000 epochs, layers 1-2" \
      uv run python scripts/train_410m_saes.py --config "$CFG" --mode "$mode" "${margs[@]}" \
        --max-layers 3 --epochs 1000 --lr 1e-3 --seed 0 --out-npz "$p" \
        --out "$(rundir "gatepreview-${mode}-s0-410md")"
  done
  note "seed-0 reproduction gate — PREVIEW on layers 1-2"
  if reproduces "$OUT/sources_trained_l1_s0_gatepreview.npz" "$REF_L1" \
                "$OUT/sources_trained_topk_s0_gatepreview.npz" "$REF_TOPK" 2>&1 | tee -a "$LOG"; then
    note "gate preview PASSED"
  else
    note "gate preview FAILED — the full run would abort at the seed-0 gate"
    FAILED+=("gate preview")
  fi
fi

if [ "${#FAILED[@]}" -gt 0 ]; then
  note "FINISHED WITH ${#FAILED[@]} FAILED STEP(S): ${FAILED[*]}"
  exit 1
fi
note "ALL DONE — see $(rundir e1-sae-seed-stability) and $(rundir e2-ssm-draw-stability)"
