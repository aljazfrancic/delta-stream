# R9b — Param-matched baseline under the LASSO protocol (Gate-3 wording decider)

**Date:** 2026-07-03/04 (run dir dated by local midnight). **Script:**
`gate3_pareto.py --rank 64 --lambdas 1` (both variants, otherwise the exact
recorded Gate-3 protocol: mlsae sources, split 1244/311 seed 0, d_z=r, 10 outer
iters, batch 128). **Question (from R9):** R9 showed that in the *frozen-source*
protocol a rank-64 per-layer decoder (8.41M params) matches the rank-32 SSM
(8.36M) — does the **LASSO protocol** (per-example codes re-inferred against
test Δ with operators frozen; the Gate-3 protocol) also lose its +0.049 gap
when the baseline gets the same total parameters?

> **Verdict — the state-path edge SURVIVES at matched total params, but
> shrinks ~6×: from +0.049 (matched rank) to ≈+0.008 (matched params).**
> Under per-example code inference, cross-layer state buys a small real edge
> that decoder width does not fully replicate — unlike the frozen protocol,
> where R9 found a wash. Write-up wording: report the +0.049 [+0.0464,
> +0.0516] rank-matched gap with the ≈+0.008 param-matched qualifier beside
> it, and the R9 frozen-protocol wash alongside. Do not use "load-bearing"
> without the param-matched number in the same sentence.

## Result (LASSO protocol, λ_L1 = 1 knee)

| arm | rank / d_z | params | test R² | test L0 |
|---|---|---|---|---|
| per-layer baseline (this run) | 64 / — | 8.41M | **+0.849** | 0.0158 |
| SSM (recorded, R0 `runs/2026-07-03-r0-gate3-ci-mlsae`) | 32 / 32 | 8.36M | **+0.8571** | 0.0149 |
| per-layer baseline (recorded, R0) | 32 / — | 4.22M | +0.8081 | 0.0133 |

Matched-params gap **ssm(r32) − baseline(r64) ≈ +0.008**. Conservative in two
ways: the r64 baseline runs slightly *denser* (L0 0.0158 vs 0.0149 — more
active code, which helps it), and its train R² (+0.875) shows headroom rather
than saturation. **Pending refinement:** a paired bootstrap CI for the +0.008
needs a per-example-SS export refit of the r64 baseline arm (~15 min;
`r0_bootstrap_ci.py` has the machinery and the ssm-r32 arrays already exist).
For scale: the R0 CI half-width on the same 311 examples was ±0.0026, and S2
seed-sd on gate3 gaps was 0.001–0.003 — +0.008 is plausibly but not yet
demonstrably significant.

## Ops note — the rank-64 SSM arm DIVERGED (excluded)

The run's own `ssm` arm (r=64, d_z=64) exploded during alternating
minimisation: |D| 150 → 27k within two outer iterations, train R² −4.1e9,
test-time FISTA inference −1.4e13 with fully dense codes. This is the
documented Qwen heavy-λ divergence (blown operator norms feeding an unstable
24-layer unroll), here triggered at the λ=1 knee purely by the larger
rank/state at the default `operator_lr=1e-2`. It is an **optimizer artifact,
not a science number** — the frozen-protocol rank-64 SSM (Adam-only, R2) fit
fine at +0.8155 — and it is unneeded here (the comparison target is the
*recorded* rank-32 SSM). Anyone extending the LASSO-protocol rank series must
drop `operator_lr` (and/or raise λ_reg) at r ≥ 64. The `gate3_pareto.json`
"NOT met" flag for this run refers to the diverged ssm arm and should be
ignored; the run's purpose was the baseline arm.

## Where this leaves the §4.2 story (with R9)

- **Matched per-decoder rank:** SSM ≫ per-layer (+0.049 LASSO / +0.053 frozen,
  both CI-tight) — unchanged.
- **Matched total params, frozen sources (R9):** wash (±0.005, opposite signs
  at 8.4M and 16.8M).
- **Matched total params, LASSO protocol (this run):** small surviving SSM
  edge ≈ +0.008.
- The SSM's robust distinctions are therefore: (i) the small per-example-
  inference edge, (ii) sample efficiency vs CLT (R3), and (iii) **structure** —
  the interpretable contractive propagation operator (A_ℓ spectrum, τ ≈ 0.83
  layers) that a widened per-layer dictionary does not provide at any budget.
