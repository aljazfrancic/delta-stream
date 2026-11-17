#!/usr/bin/env bash
# v1.1 post-hoc on-box checks — the four checks the paper names in §8 and does not run.
#
#   A  the state probe with codes inferred from the layers below ℓ only, plus a
#      feedthrough-image arm, over both dictionaries      (sizes the §6.7 look-ahead)
#   B  the source-prefix sweep with the bias initialised at the per-depth mean (§6.9)
#   C  the parameter-matched gap at further seeds, over the L1 dictionary, on the
#      untrained network and at 12 layers                  (§6.1: one fit per arm so far)
#   D  AR-kernel transfer between the trained and the untrained network (§3.4, T4)
#
# Design, statistics and verdict rows are frozen in runs/v11-onbox-checks/memo.md,
# committed before any run. Everything here is post hoc.
#
#   scripts/run_v11_checks.sh --check    # seconds: preflight + which steps are done
#   scripts/run_v11_checks.sh --smoke    # ~40 min: every step end to end at toy settings
#                                        #   (git-ignored paths only; rebuilds the real
#                                        #   inputs, which the full run then reuses)
#   scripts/run_v11_checks.sh            # the full run, sequential, one log (~10 h)
#   V11_DATE=2026-11-17 scripts/run_v11_checks.sh   # resume a run started on that date
#   scripts/run_v11_checks.sh --stages transfer,probe-l1   # a subset, in the fixed order
#
# Launch detached (survives the terminal, the editor or an assistant session):
#   scripts/launch_v11_checks.sh         # start | smoke | status | stop | check
#
# Sequential, unattended, continue-on-error, single log. Every step is skipped if its
# output exists, so `stop` then `start` resumes; a stop costs at most the step in
# progress (the causal-code step also caches per depth). Nothing recorded is touched:
# rebuilt inputs go to data/<model>/v11c/, run tables to runs/<date>-v11c-*.
#
# Rebuilt inputs are GATED: each must reproduce the tracked table it originally produced
# (MLSAE sources, both SSM checkpoints, the rank-64 baseline, the embedding-only fit, the
# untrained store through its recorded AR(1) levels, the untrained dictionary and knee,
# the 12-layer knee). A failed gate skips what depends on it and is reported; it never
# silently falls through.
#
# Order = cheapest decisive result first: sources → D → A (L1, then MLSAE) → B → C
# (410m) → C (untrained) → C (12 layers) → summary.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Everything needed is in the local Hugging Face cache; fail fast rather than fetch.
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1

CFG=configs/pythia-410m-deduped.yaml
CFG160=configs/pythia-160m-deduped.yaml
MODEL_DIR=data/pythia-410m-deduped
UN_DIR=data/pythia-410m-deduped-untrained
P160_DIR=data/pythia-160m-deduped
TEMPLATE160="tim-lawson/sae-pythia-160m-deduped-x64-k32-layers-{layer}"
DATE="${V11_DATE:-$(date +%F)}"
SMOKE=0; CHECK=0
ALL_STAGES="sources transfer probe-l1 probe-mlsae prefix pm untrained depth summary"
STAGES="$ALL_STAGES"
while [ $# -gt 0 ]; do
  case "$1" in
    --smoke) SMOKE=1 ;;
    --check) CHECK=1 ;;
    --stages) shift; STAGES="${1//,/ }" ;;
    *) echo "unknown arg: $1 (known: --smoke, --check, --stages a,b,…)"; exit 2 ;;
  esac
  shift
done
for s in $STAGES; do
  case " $ALL_STAGES " in *" $s "*) ;; *) echo "unknown stage '$s' (known: $ALL_STAGES)"; exit 2 ;; esac
done
want() { case " $STAGES " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

# --- tracked references --------------------------------------------------------------
REF_SRC_MLSAE=runs/2026-06-25-ph3-sources-410md/tables/sources_mlsae.json
REF_SRC_L1=runs/2026-06-29-ph4-e2-train-l1-410md/tables/sources_trained_l1.json
REF_CKPT_MLSAE=runs/2026-06-26-ph4-checkpoint-410md
REF_CKPT_L1=runs/2026-06-30-ph4-e2-checkpoint-l1-410md
REF_PROBE_MLSAE=runs/2026-07-04-r7-state-probe-mlsae/tables/state_probe.json
REF_PROBE_L1=runs/2026-07-04-r7-state-probe-l1/tables/state_probe.json
REF_PREFIX=runs/2026-06-30-ph4-e5-prefix-410md/tables/embed_only.json
REF_PM_CI=runs/2026-07-04-r9b-gate3-ci-rank64
REF_SSM_S1=runs/2026-06-29-ph3-gate3-410md-s1
REF_SSM_S2=runs/2026-06-29-ph3-gate3-410md-s2
REF_T1=runs/2026-07-09-r13-common-depth-pressure/tables/t1_common_depth.csv
REF_UN_DICT=runs/2026-06-30-ph4-t4recon-train-l1-untrained-410md/tables/sources_trained_l1.json
REF_UN_KNEE=runs/2026-07-01-ph4-t4recon-gate3-l1-untrained-410md
REF_160_SRC=runs/2026-06-26-xm-p160-sources/tables/sources_mlsae.json
REF_160_S0=runs/2026-06-26-xm-p160-gate3
REF_160_S1=runs/2026-06-28-xm-p160-gate3-s1
REF_160_S2=runs/2026-06-28-xm-p160-gate3-s2

SRC_L1="$MODEL_DIR/v11/sources_trained_l1_s0.npz"       # the verified seed-0 retrain (E1)
if [ "$SMOKE" = 1 ]; then
  LOG=runs/v11c-smoke.log; SUB=v11c-smoke
  rundir() { echo "runs/v11c-smoke/$1"; }
  CKPT_EXTRA=(--outer-iters 1 --n-seeds 2); CAUSAL_EXTRA=(--depths 1 2)
  PROBE_EXTRA=(--layers 0 1 2 --n-boot 200); FIT_EXTRA=(--outer-iters 1)
  PREFIX_EXTRA=(--epochs 3 --cutoffs 0 23); PREFIX_GATE_EXTRA=(--epochs 3)
  DICT_EXTRA=(--max-layers 3 --epochs 50)
else
  LOG=runs/v11-checks-overnight.log; SUB=v11c
  rundir() { echo "runs/${DATE}-v11c-$1"; }
  CKPT_EXTRA=(--outer-iters 10); CAUSAL_EXTRA=(); PROBE_EXTRA=(); FIT_EXTRA=(--outer-iters 10)
  PREFIX_EXTRA=(); PREFIX_GATE_EXTRA=(); DICT_EXTRA=()
fi
# Real inputs (deterministic rebuilds, gated) live in one place for both modes.
SRC_MLSAE="$MODEL_DIR/v11c/sources_mlsae.npz"
UN_H5="$UN_DIR/residual_stream.h5"
UN_SRC="$UN_DIR/$SUB/sources_trained_l1_s0.npz"
P160_SRC="$P160_DIR/v11c/sources_mlsae.npz"

# --- preflight ------------------------------------------------------------------------
for f in "$MODEL_DIR/residual_stream.h5" "$MODEL_DIR/filtered_triples.parquet" "$SRC_L1"; do
  [ -f "$f" ] || { echo "missing $f — rebuild the store / the seed-0 L1 dictionary first"; exit 1; }
done
for f in "$REF_SRC_MLSAE" "$REF_SRC_L1" "$REF_CKPT_MLSAE/tables/candidate_pool.json" \
         "$REF_CKPT_L1/tables/candidate_pool.json" "$REF_PROBE_MLSAE" "$REF_PROBE_L1" \
         "$REF_PREFIX" "$REF_PM_CI/tables/r0_bootstrap_ci.json" \
         "$REF_SSM_S1/tables/pareto.csv" "$REF_SSM_S2/tables/pareto.csv" "$REF_T1" \
         "$REF_UN_DICT" "$REF_UN_KNEE/tables/pareto.csv" "$REF_160_SRC" \
         "$REF_160_S0/tables/pareto.csv" "$REF_160_S1/tables/pareto.csv" \
         "$REF_160_S2/tables/pareto.csv" runs/v11-onbox-checks/memo.md; do
  [ -f "$f" ] || { echo "missing tracked reference $f"; exit 1; }
done
uv run python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" || {
  echo "CUDA is not available — this run needs the GPU"; exit 1; }
mkdir -p "$MODEL_DIR/v11c" "$UN_DIR/$SUB" "$P160_DIR/v11c" runs

FAILED=()
step() {  # <label> <command...>
  local label="$1"; shift
  echo "=== $(date -Is)  START $label ===" | tee -a "$LOG"
  if "$@" >>"$LOG" 2>&1; then
    echo "=== $(date -Is)  DONE  $label ===" | tee -a "$LOG"; return 0
  else
    echo "=== $(date -Is)  FAIL  $label (rc=$?, continuing) ===" | tee -a "$LOG"
    FAILED+=("$label"); return 1
  fi
}
note() { echo "=== $(date -Is)  $* ===" | tee -a "$LOG"; }
gate() {  # <label> <v11_gate.py args...> — 0 if the rebuilt artefact reproduces its record
  local label="$1" out rc; shift
  out="$(uv run python scripts/v11_gate.py "$@" 2>&1)"; rc=$?
  echo "$out" | tee -a "$LOG"
  if [ "$rc" -eq 0 ]; then
    note "GATE OK    $label"; return 0
  else
    note "GATE FAIL  $label — what depends on it is skipped"; FAILED+=("gate: $label"); return 1
  fi
}
have() { [ -e "$1" ] && echo "done" || echo "to do"; }
py() { uv run python "$@"; }

CK_MLSAE="$(rundir checkpoint-mlsae-410md)"; CK_L1="$(rundir checkpoint-l1-410md)"
CA_MLSAE="$(rundir causal-codes-mlsae-410md)"; CA_L1="$(rundir causal-codes-l1-410md)"
PR_MLSAE="$(rundir probe-causal-mlsae-410md)"; PR_L1="$(rundir probe-causal-l1-410md)"
PX_GATE="$(rundir prefix-zero-init-gate-410md)"; PX="$(rundir prefix-mean-init-410md)"
PM_CI="$(rundir pm-ci-mlsae-r64)"; PM_L1="$(rundir pm-ci-l1-r64)"
PM_S1="$(rundir pm-baseline-r64-mlsae-s1)"; PM_S2="$(rundir pm-baseline-r64-mlsae-s2)"
UN_COLLECT="$(rundir t4-untrained-410md)"; TRANSFER="$(rundir kernel-transfer-410md)"
UN_DICT="$(rundir train-l1-untrained-410md)"
UN_R32="$(rundir gate3-l1-untrained-r32)"; UN_R64="$(rundir pm-baseline-r64-l1-untrained)"
P160_R32="$(rundir p160-gate3-r32-s0)"
p160_r64() { rundir "p160-pm-baseline-r64-s$1"; }
SUMMARY="$(rundir summary)"

if [ "$CHECK" = 1 ]; then
  echo "preflight OK: store, seed-0 L1 dictionary, tracked references and CUDA are present"
  echo "commit $(git rev-parse --short HEAD), working tree $([ -n "$(git status --porcelain)" ] && echo DIRTY || echo clean); run date $DATE; stages: $STAGES"
  echo "  MLSAE sources:            $(have "$SRC_MLSAE")"
  echo "  D  untrained store:       $(have "$UN_H5") | transfer: $(have "$TRANSFER/tables/kernel_transfer.json")"
  for d in l1 mlsae; do
    ck="$(rundir "checkpoint-$d-410md")"; ca="$(rundir "causal-codes-$d-410md")"; pr="$(rundir "probe-causal-$d-410md")"
    echo "  A  $d: checkpoint $(have "$ck/tables/candidate_pool.json") | causal codes $(have "$ca/tables/causal_codes.json") | probe $(have "$pr/tables/state_probe_causal.json")"
  done
  echo "  B  prefix: gate $(have "$PX_GATE/tables/embed_only.json") | mean-init sweep $(have "$PX/tables/embed_only.json")"
  echo "  C  410m: MLSAE CI $(have "$PM_CI/tables/r0_bootstrap_ci.json") | seed 1 $(have "$PM_S1/tables/gate3_pareto.json") | seed 2 $(have "$PM_S2/tables/gate3_pareto.json") | L1 CI $(have "$PM_L1/tables/r0_bootstrap_ci.json")"
  echo "  C  untrained: dictionary $(have "$UN_DICT/tables/sources_trained_l1.json") | rank 32 $(have "$UN_R32/tables/gate3_pareto.json") | rank-64 baseline $(have "$UN_R64/tables/gate3_pareto.json")"
  echo "  C  12 layers: store $(have "$P160_DIR/residual_stream.h5") | sources $(have "$P160_SRC") | knee $(have "$P160_R32/tables/gate3_pareto.json") | rank-64 baselines $(have "$(p160_r64 0)/tables/gate3_pareto.json")/$(have "$(p160_r64 1)/tables/gate3_pareto.json")/$(have "$(p160_r64 2)/tables/gate3_pareto.json")"
  echo "  summary: $(have "$SUMMARY/tables/checks_summary.json")"
  exit 0
fi

note "v1.1 on-box checks started ($([ "$SMOKE" = 1 ] && echo smoke || echo full), date $DATE, commit $(git rev-parse --short HEAD)$([ -n "$(git status --porcelain)" ] && echo ' DIRTY'), stages: $STAGES)"

# --- sources: the MLSAE codes, rebuilt from the cached SAEs and gated ----------------
OK_L1SRC=0; OK_MLSAE=0
gate "seed-0 L1 dictionary vs its record" sources "$SRC_L1" "$REF_SRC_L1" && OK_L1SRC=1
if want sources || want probe-mlsae || want prefix || want pm; then
  [ -e "$SRC_MLSAE" ] || step "build MLSAE sources" py scripts/build_410m_sources.py \
    --config "$CFG" --suite mlsae --out-npz "$SRC_MLSAE" --out "$(rundir sources-mlsae-410md)" --seed 0
  [ -e "$SRC_MLSAE" ] && gate "MLSAE sources vs their record" sources "$SRC_MLSAE" "$REF_SRC_MLSAE" && OK_MLSAE=1
fi

# --- D: kernel transfer (also the gate on the untrained store) -----------------------
OK_UNSTORE=0
if want transfer || want untrained; then
  if [ ! -e "$UN_H5" ] || [ ! -e "$UN_COLLECT/tables/t4_collect_stats.json" ]; then
    step "collect the untrained-network store" py scripts/phase4_t4_untrained.py \
      --config "$CFG" --out "$UN_COLLECT" --source factual --seed 0
  else
    note "SKIP  untrained store (already collected)"
  fi
  if [ -e "$UN_H5" ]; then
    [ -e "$TRANSFER/tables/kernel_transfer.json" ] && note "SKIP  kernel transfer (already done)" || \
      step "D kernel transfer, trained vs untrained" py scripts/v11_kernel_transfer.py \
        --config "$CFG" --out "$TRANSFER" --seed 0 \
        --trained "$MODEL_DIR/residual_stream.h5" --untrained "$UN_H5" \
        --recorded-csv "$REF_T1" --recorded-labels 410m-factual-trained 410m-factual-untrained
    if grep -qs '"ok": true' "$TRANSFER/tables/kernel_transfer.json"; then
      OK_UNSTORE=1; note "GATE OK    both stores reproduce their recorded AR(1) levels"
    else
      note "GATE FAIL  the stores do not reproduce the recorded AR(1) levels"
      FAILED+=("gate: untrained store")
    fi
  fi
fi

# --- A: the probe, one dictionary at a time -------------------------------------------
probe_chain() {  # <tag> <sources npz> <recorded checkpoint dir> <recorded probe json>
  local tag="$1" src="$2" ref_ck="$3" ref_probe="$4"
  local ck ca pr; ck="$(rundir "checkpoint-$tag-410md")"; ca="$(rundir "causal-codes-$tag-410md")"
  pr="$(rundir "probe-causal-$tag-410md")"
  if [ -e "$ck/tables/candidate_pool.json" ]; then
    note "SKIP  checkpoint $tag (already done)"
  else
    local resume=()
    [ -e "$ck/checkpoints/u_infer_full.npy" ] && [ -e "$ck/checkpoints/infer_meta.json" ] && resume=(--resume)
    step "A checkpoint $tag" py scripts/phase4_checkpoint.py --config "$CFG" --out "$ck" \
      --sources-npz "$src" --rank 32 --lambda-l1 1.0 "${CKPT_EXTRA[@]}" --device cuda --seed 0 \
      "${resume[@]}" || return
  fi
  if [ "$SMOKE" = 0 ]; then
    gate "checkpoint $tag vs its record" checkpoint "$ck" "$ref_ck" || return
  fi
  [ -e "$ca/tables/causal_codes.json" ] && grep -qs '"ok": true' "$ca/tables/causal_codes.json" \
    && note "SKIP  causal codes $tag (already done)" || \
    step "A causal codes $tag" py scripts/v11_causal_codes.py --config "$CFG" --out "$ca" \
      --checkpoint-dir "$ck" --sources-npz "$src" --device cuda --seed 0 "${CAUSAL_EXTRA[@]}" || return
  local rec=(--recorded-json "$ref_probe"); [ "$SMOKE" = 1 ] && rec=()
  [ -e "$pr/tables/state_probe_causal.json" ] && note "SKIP  probe $tag (already done)" || \
    step "A probe $tag" py scripts/v11_probe_causal.py --config "$CFG" --out "$pr" \
      --checkpoint-dir "$ck" --causal-dir "$ca" --sources-npz "$src" --seed 0 \
      "${rec[@]}" "${PROBE_EXTRA[@]}"
}
if want probe-l1; then
  if [ "$OK_L1SRC" = 1 ]; then probe_chain l1 "$SRC_L1" "$REF_CKPT_L1" "$REF_PROBE_L1"
  else note "SKIP  probe l1 (L1 dictionary gate failed)"; fi
fi
if want probe-mlsae; then
  if [ "$OK_MLSAE" = 1 ]; then probe_chain mlsae "$SRC_MLSAE" "$REF_CKPT_MLSAE" "$REF_PROBE_MLSAE"
  else note "SKIP  probe mlsae (MLSAE sources gate failed)"; fi
fi

# --- B: the prefix sweep with a mean-initialised bias ----------------------------------
if want prefix; then
  if [ "$OK_MLSAE" = 1 ]; then
    [ -e "$PX_GATE/tables/embed_only.json" ] || step "B zero-init embedding-only fit (gate)" \
      py scripts/phase4_embed_only.py --config "$CFG" --out "$PX_GATE" --seed 0 \
        --sources-npz "$SRC_MLSAE" --cutoffs 0 "${PREFIX_GATE_EXTRA[@]}"
    if [ "$SMOKE" = 1 ] || gate "embedding-only fit vs its record" prefix "$PX_GATE" "$REF_PREFIX" --cutoffs 0; then
      [ -e "$PX/tables/embed_only.json" ] && note "SKIP  mean-init prefix sweep (already done)" || \
        step "B prefix sweep, bias initialised at the mean" py scripts/phase4_embed_only.py \
          --config "$CFG" --out "$PX" --seed 0 --sources-npz "$SRC_MLSAE" --bias-init mean \
          "${PREFIX_EXTRA[@]}"
    fi
  else
    note "SKIP  prefix sweep (MLSAE sources gate failed)"
  fi
fi

# --- C: the parameter-matched gap on the primary model --------------------------------
if want pm; then
  if [ -e "$CK_MLSAE/tables/candidate_pool.json" ] && [ "$OK_MLSAE" = 1 ]; then
    [ -e "$PM_CI/tables/r0_bootstrap_ci.json" ] && note "SKIP  rank-64 CI, MLSAE (already done)" || \
      step "C rank-64 baseline + paired CI, MLSAE seed 0" py scripts/r0_bootstrap_ci.py \
        --config "$CFG" --out "$PM_CI" --seed 0 --checkpoint-dir "$CK_MLSAE" \
        --sources-npz "$SRC_MLSAE" --rank 64 "${FIT_EXTRA[@]}"
    [ "$SMOKE" = 1 ] || gate "rank-64 CI, MLSAE vs its record" ci "$PM_CI" "$REF_PM_CI"
  else
    note "SKIP  rank-64 CI, MLSAE (no MLSAE checkpoint)"
  fi
  if [ "$OK_MLSAE" = 1 ]; then
    for s in 1 2; do
      d="$(rundir "pm-baseline-r64-mlsae-s$s")"
      [ -e "$d/tables/gate3_pareto.json" ] && { note "SKIP  rank-64 baseline, MLSAE seed $s (already done)"; continue; }
      step "C rank-64 baseline, MLSAE seed $s" py scripts/gate3_pareto.py --config "$CFG" \
        --out "$d" --sources-npz "$SRC_MLSAE" --rank 64 --lambdas 1 --variants baseline \
        "${FIT_EXTRA[@]}" --batch-size 128 --device cuda --seed "$s"
    done
  fi
  if [ -e "$CK_L1/tables/candidate_pool.json" ]; then
    [ -e "$PM_L1/tables/r0_bootstrap_ci.json" ] && note "SKIP  rank-64 CI, L1 (already done)" || \
      step "C rank-64 baseline + paired CI, L1 dictionary" py scripts/r0_bootstrap_ci.py \
        --config "$CFG" --out "$PM_L1" --seed 0 --checkpoint-dir "$CK_L1" \
        --sources-npz "$SRC_L1" --rank 64 "${FIT_EXTRA[@]}"
  else
    note "SKIP  rank-64 CI, L1 (no L1 checkpoint)"
  fi
fi

# --- C: the same gap on the untrained network -----------------------------------------
if want untrained; then
  if [ "$OK_UNSTORE" = 1 ]; then
    if [ -e "$UN_SRC" ] && [ -e "${UN_SRC%.npz}.weights.pkl" ] && [ -e "$UN_DICT/tables/sources_trained_l1.json" ]; then
      note "SKIP  untrained-network dictionary (complete; reused, never overwritten)"
    else
      step "C train the L1 dictionary on the untrained network" py scripts/train_410m_saes.py \
        --config "$CFG" --mode l1 --l1-coeff 0.5 --h5 "$UN_H5" --lr 1e-3 --seed 0 \
        "${DICT_EXTRA[@]}" --out-npz "$UN_SRC" --out "$UN_DICT"
    fi
    if [ -e "$UN_SRC" ] && { [ "$SMOKE" = 1 ] || gate "untrained dictionary vs its record" sources "$UN_SRC" "$REF_UN_DICT"; }; then
      [ -e "$UN_R32/tables/gate3_pareto.json" ] || step "C untrained network, rank-32 knee" \
        py scripts/gate3_pareto.py --config "$CFG" --out "$UN_R32" --h5 "$UN_H5" \
          --sources-npz "$UN_SRC" --rank 32 --lambdas 1 "${FIT_EXTRA[@]}" --batch-size 128 \
          --device cuda --seed 0
      if [ "$SMOKE" = 1 ] || gate "untrained knee vs its record" knee "$UN_R32" "$REF_UN_KNEE"; then
        [ -e "$UN_R64/tables/gate3_pareto.json" ] || step "C untrained network, rank-64 baseline" \
          py scripts/gate3_pareto.py --config "$CFG" --out "$UN_R64" --h5 "$UN_H5" \
            --sources-npz "$UN_SRC" --rank 64 --lambdas 1 --variants baseline "${FIT_EXTRA[@]}" \
            --batch-size 128 --device cuda --seed 0
      fi
    fi
  else
    note "SKIP  untrained parameter-matched gap (no verified untrained store)"
  fi
fi

# --- C: the same gap at 12 layers (Pythia-160m) ---------------------------------------
OK_160=0
if want depth; then
  [ -e "$P160_DIR/filtered_triples.parquet" ] || step "12L dataset" py scripts/build_dataset.py \
    --config "$CFG160" --out "$(rundir p160-dataset)" --seed 0
  [ -e "$P160_DIR/residual_stream.h5" ] || { [ -e "$P160_DIR/filtered_triples.parquet" ] && \
    step "12L activations" py scripts/collect_activations.py --config "$CFG160" \
      --out "$(rundir p160-collect)" --seed 0; }
  [ -e "$P160_SRC" ] || { [ -e "$P160_DIR/residual_stream.h5" ] && \
    step "12L MLSAE sources" py scripts/build_410m_sources.py --config "$CFG160" --suite mlsae \
      --sae-template "$TEMPLATE160" --out-npz "$P160_SRC" --out "$(rundir p160-sources)" --seed 0; }
  if [ -e "$P160_SRC" ] && gate "12L sources vs their record" sources "$P160_SRC" "$REF_160_SRC"; then
    [ -e "$P160_R32/tables/gate3_pareto.json" ] || step "12L rank-32 knee, seed 0" \
      py scripts/gate3_pareto.py --config "$CFG160" --out "$P160_R32" --sources-npz "$P160_SRC" \
        --rank 32 --lambdas 1 "${FIT_EXTRA[@]}" --device cuda --seed 0
    if [ "$SMOKE" = 1 ] || gate "12L knee vs its record" knee "$P160_R32" "$REF_160_S0"; then
      OK_160=1
      for s in 0 1 2; do
        d="$(p160_r64 "$s")"
        [ -e "$d/tables/gate3_pareto.json" ] && continue
        step "C 12L rank-64 baseline, seed $s" py scripts/gate3_pareto.py --config "$CFG160" \
          --out "$d" --sources-npz "$P160_SRC" --rank 64 --lambdas 1 --variants baseline \
          "${FIT_EXTRA[@]}" --device cuda --seed "$s"
      done
    fi
  else
    note "SKIP  12-layer parameter-matched gap (store or sources not rebuilt / gate failed)"
  fi
fi

# --- summary ---------------------------------------------------------------------------
if want summary; then
  ARGS=()
  [ -e "$PR_MLSAE/tables/state_probe_causal.json" ] && ARGS+=(--probe "mlsae=$PR_MLSAE")
  [ -e "$PR_L1/tables/state_probe_causal.json" ] && ARGS+=(--probe "l1=$PR_L1")
  [ -e "$PX/tables/embed_only.json" ] && ARGS+=(--prefix "$PX" --prefix-recorded "$REF_PREFIX")
  [ -e "$PM_CI/tables/r0_bootstrap_ci.json" ] && ARGS+=(--pm-ci "$PM_CI")
  [ -e "$PM_S1/tables/gate3_pareto.json" ] && ARGS+=(--pm-seed "$PM_S1" "$REF_SSM_S1")
  [ -e "$PM_S2/tables/gate3_pareto.json" ] && ARGS+=(--pm-seed "$PM_S2" "$REF_SSM_S2")
  [ -e "$PM_L1/tables/r0_bootstrap_ci.json" ] && ARGS+=(--pm-l1-ci "$PM_L1")
  [ -e "$UN_R64/tables/gate3_pareto.json" ] && [ -e "$UN_R32/tables/gate3_pareto.json" ] && \
    ARGS+=(--pm-untrained "$UN_R64" "$UN_R32")
  if [ -e "$(p160_r64 0)/tables/gate3_pareto.json" ] && [ -e "$(p160_r64 1)/tables/gate3_pareto.json" ] \
     && [ -e "$(p160_r64 2)/tables/gate3_pareto.json" ]; then
    ARGS+=(--pm-shallow "$(p160_r64 0)" "$REF_160_S0" --pm-shallow "$(p160_r64 1)" "$REF_160_S1"
           --pm-shallow "$(p160_r64 2)" "$REF_160_S2")
  fi
  [ -e "$TRANSFER/tables/kernel_transfer.json" ] && ARGS+=(--transfer "$TRANSFER")
  rm -rf "$SUMMARY"   # the summary is derived; always rebuilt from whatever exists
  step "summary" py scripts/v11_checks_summary.py --config "$CFG" --out "$SUMMARY" --seed 0 "${ARGS[@]}"
fi

if [ "${#FAILED[@]}" -gt 0 ]; then
  note "FINISHED WITH ${#FAILED[@]} FAILED STEP(S): ${FAILED[*]}"
  exit 1
fi
note "ALL DONE — see $SUMMARY/tables/checks_summary.json"
