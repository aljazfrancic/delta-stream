#!/usr/bin/env bash
# Overnight cross-model replication chain (Phase 2 Step 5).
#
# Runs the full pipeline — H1 (ridge -> AR/Granger) -> H2 (stationarity, k*) ->
# Phase-3 (SSM-vs-per-layer-baseline gap) — for GPT-2-small (cross-family, res-jb
# SAEs) then Pythia-160m-deduped (in-family depth midpoint, mlsae SAEs).
#
# Robust for unattended use: run dirs are resolved by globbing the newest match for
# each --desc (survives the midnight date rollover); each model runs in isolation so
# one failure does not abort the other; Phase-3 fits are gated on the source build's
# built-in reconstruction sanity (exit!=0 => skip the SSM fits, keep H1/H2).
#
# Usage:  scripts/run_crossmodel.sh            # both models
#         scripts/run_crossmodel.sh gpt2       # GPT-2 only
#         scripts/run_crossmodel.sh p160       # Pythia-160m only
#
# Logs to runs/crossmodel-overnight.log (plus each step's own run.log).

set -u
cd "$(dirname "$0")/.."
PY="uv run python"
LOG="runs/crossmodel-overnight.log"
RANK=32
LAMBDAS="0.3,1,3,10,30"
OUTER=10

say() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

# run a step; on failure log and return 1 (caller decides whether it is fatal)
step() {
  local name="$1"; shift
  say "START $name :: $*"
  if $PY "$@" >>"$LOG" 2>&1; then
    say "OK    $name"; return 0
  else
    say "FAIL  $name (exit $?)"; return 1
  fi
}

# newest run dir for a --desc
rundir() { ls -dt runs/*-"$1" 2>/dev/null | head -1; }

run_model() {
  local cfg="$1" tag="$2" suite="$3" model="$4" template="${5:-}"
  say "================  MODEL $tag ($model, suite=$suite)  ================"

  # --- Phase 1: dataset + collection ---
  step "$tag dataset" scripts/build_dataset.py --config "$cfg" --desc "$tag-dataset" --seed 0 \
    || { say "$tag: dataset failed -> skipping model"; return 1; }
  step "$tag collect" scripts/collect_activations.py --config "$cfg" --desc "$tag-collect" --seed 0 \
    || { say "$tag: collect failed -> skipping model"; return 1; }

  # --- Phase 1: H1 diagnostics ---
  step "$tag step3" scripts/step3_ridge.py --config "$cfg" --desc "$tag-step3" --seed 0 \
    || { say "$tag: step3 failed -> skipping rest"; return 1; }
  local d3; d3="$(rundir "$tag-step3")"
  step "$tag step5" scripts/step5_ar.py --config "$cfg" --desc "$tag-step5" \
    --step3-csv "$d3/tables/ridge_r2.csv" \
    || { say "$tag: step5 failed -> skipping H2/Phase3"; return 1; }
  local d5; d5="$(rundir "$tag-step5")"

  # --- Phase 2: H2 stationarity (+ k* best-effort) ---
  step "$tag ph2step1" scripts/step1_stationarity.py --config "$cfg" --desc "$tag-ph2step1" \
    --step3-csv "$d3/tables/ridge_r2.csv" --granger-csv "$d5/tables/granger_real.csv" \
    || say "$tag: stationarity failed (continuing)"
  local d1; d1="$(rundir "$tag-ph2step1")"
  if [ -n "$d1" ]; then
    step "$tag ph2step2" scripts/step2_kstar.py --config "$cfg" --desc "$tag-ph2step2" \
      --step1-stats "$d1/tables/step1_stats.json" --granger-csv "$d5/tables/granger_real.csv" \
      || say "$tag: k* failed (continuing)"
  fi

  # --- Phase 3: SSM-vs-baseline gap (gated on source recon sanity) ---
  local npz="data/$model/sources_${suite}.npz"
  local src_args=(scripts/build_410m_sources.py --config "$cfg" --desc "$tag-sources" --suite "$suite")
  [ "$suite" = "mlsae" ] && [ -n "$template" ] && src_args+=(--sae-template "$template")
  if step "$tag sources" "${src_args[@]}"; then
    step "$tag gate3" scripts/gate3_pareto.py --config "$cfg" --desc "$tag-gate3" \
      --sources-npz "$npz" --rank "$RANK" --lambdas "$LAMBDAS" --outer-iters "$OUTER" --device cuda \
      || say "$tag: gate3 pareto failed (continuing)"
    step "$tag step4" scripts/step4_sanity.py --config "$cfg" --desc "$tag-step4" \
      --sources-npz "$npz" --rank "$RANK" --lambda-l1 1.0 --outer-iters "$OUTER" --device cuda \
      || say "$tag: step4 sanity failed (continuing)"
  else
    say "$tag: source build suspect (recon R² < 0.4 or load error) -> SKIP Phase-3 fits; H1/H2 stand"
  fi
  say "================  MODEL $tag DONE  ================"
}

WHICH="${1:-both}"
say "##### cross-model overnight run start (which=$WHICH) #####"
if [ "$WHICH" = "gpt2" ] || [ "$WHICH" = "both" ]; then
  run_model configs/gpt2-small.yaml xm-gpt2 resjb gpt2
fi
if [ "$WHICH" = "p160" ] || [ "$WHICH" = "both" ]; then
  run_model configs/pythia-160m-deduped.yaml xm-p160 mlsae pythia-160m-deduped \
    "tim-lawson/sae-pythia-160m-deduped-x64-k32-layers-{layer}"
fi
say "##### cross-model overnight run COMPLETE #####"
