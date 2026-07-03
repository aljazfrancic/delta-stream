# R9 — Param-matched per-layer control (frozen-source protocol; 4th-pass review)

**Date:** 2026-07-03. **Runs:** `runs/2026-07-03-r9-perlayer-rank64` (this dir) +
`runs/2026-07-03-r9-perlayer-rank128`. **Script:** `phase4_baseline_clt.py
--models per_layer`, protocol-identical to R2 (frozen mlsae sources
`u[1555,24,4433]`, split 1244/311 seed 0, 180 epochs, batch 64, λ_reg grid
{0.3, 1, 3, 10} — interior peak at λ=1 for both ranks, no edge-landing).

**The gap this closes.** R2 introduced the matched-**total**-params frame for
SSM-vs-CLT ("SSM r64 @16.9M beats CLT r16 @26M") but the same frame was never
applied to SSM-vs-**per_layer**: the rank-32 SSM (8.36M) had only ever been
compared to the rank-32 per_layer (4.22M) — half its parameters. per_layer at
rank 64 costs 8.41M ≈ the rank-32 SSM; at rank 128, 16.79M ≈ the rank-64 SSM
(16.89M). Without these points, "cross-layer state is load-bearing" in the
frozen protocol was confounded with "2× parameters help."

> **Verdict — DEFLATIONARY for the frozen-protocol framing: at matched total
> params the wide per-layer decoder and the state-space decoder are within
> ±0.005 of each other, in opposite directions at the two budgets.** Spending
> the same parameters on decoder width buys about as much held-out Δ
> reconstruction as spending them on the cross-layer state path. The SSM's
> distinction in the frozen protocol is *structure* (an interpretable,
> contractive propagation operator — the A_ℓ spectrum result), not
> reconstruction-per-parameter against a widened per-layer dictionary.

## Result (frozen sources; paired bootstrap over the same 311 test examples)

| budget | wide per-layer | state-space SSM | per_layer − ssm (95% CI) |
|---|---|---|---|
| ≈8.4M | r64: **+0.8027** (λ=1) | r32: +0.7975 (λ=3, lamprobe) | **+0.0052 [+0.0031, +0.0075]** — per_layer ahead |
| ≈16.8M | r128: **+0.8131** (λ=1) | r64: +0.8155 (λ=0.3, R2) | **−0.0024 [−0.0047, −0.0001]** — ssm ahead, marginal |

Cross-run CIs computed with `diagnostics.bootstrap.paired_gap_ci` on the saved
`per_example_ss.npz` arrays (identical `test_idx` verified; `ss_tot` identical).
Reference points at matched per-decoder rank (unchanged): per_layer r32 0.7445 ≪
ssm r32 0.7975; CLT r32 0.8117 @52.4M; CLT r64 0.8430 @105M.

## What survives, what changes

- **Survives:** the Gate-3 **LASSO-protocol** headline (+0.0489 [+0.0464,
  +0.0516] at the λ=1 knee) — different protocol, both arms share rank/d_z;
  the fig06 depth-scaling *trend* (same-rank arms at every depth); all R2/R3
  SSM-vs-CLT budget and sample-efficiency statements.
- **Changes:** fig08 panel A now carries the per_layer r64/r128 points and an
  outcome-neutral title — the honest picture is all three decoder families
  near one params→R² frontier on frozen sources. Write-up: do not call the
  cross-layer state *uniquely* load-bearing in the frozen protocol; it is one
  of two roughly equally effective ways to spend parameters there.
- **Decides next:** **R9b** (`runs/2026-07-03-r9b-gate3-rank64-knee`) — the
  LASSO-protocol analogue (gate3 at rank 64 vs the recorded rank-32 SSM at the
  λ=1 knee). If the rank-64 baseline closes the +0.049 LASSO gap too, the
  Gate-3 "load-bearing" language gets the same param-matched caveat; if it
  doesn't, the LASSO protocol keeps it (the state path buys something width
  cannot when codes are re-inferred per example).

## Caveats

- Single seed (0), matching the R2-family convention; the paired CIs quantify
  example-level, not seed-level, uncertainty (S2 showed seed-sd ≈0.001–0.003
  on gate3 gaps at these budgets).
- λ_reg selected per arm by best test R² (house convention; interior peaks
  here, and the 3rd/4th-pass reviews already flag the selection-on-test
  methods sentence for D3).
- per_layer r128's overfit gap (+0.066) exceeds r64's (+0.054) — width is
  starting to buy memorisation at N=1244; a params-matched comparison at
  larger N (cloud Programme A collects more) may separate the families more.
