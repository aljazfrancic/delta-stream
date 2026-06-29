# Phase 4 — E1: norm-matched random-direction control + off-target specificity

**Date:** 2026-06-29. **Model:** pythia-410m-deduped. **Checkpoint:** the canonical Branch B SSM
(`runs/2026-06-26-ph4-checkpoint-410md`, train R² 0.880). **Run:**
`runs/2026-06-29-ph4-e1-distributed-410md` (8 seed-stable candidates × 100 top examples; ks 1…300;
`scripts/phase4_distributed.py`, `--skip-single`).

E1 supplies the two controls the S3 distributed-code result was missing (2026-06-29 advisory): a
**norm-matched random *direction*** (does the group-ablation effect need the feature direction, or
just its residual magnitude?) and an **off-target** readout (does the ablation suppress the gold
answer specifically, or the whole distribution?).

> **Verdict — the distributed-code group-ablation effect is MAGNITUDE-driven, not a privileged
> feature subspace.** A random direction of the same per-(example, layer) norm as the top-k δ
> reproduces the top-k answer-prob drop at every k (k=100: top **+0.237**, isotropic **+0.224**,
> shaped **+0.204**; all three cross 0.20 at the same k=100). Even a *fully isotropic* R^D direction
> matches it. The count-matched random group stays low only because it removes less |u|-mass; once
> magnitude is matched (mass- **or** norm-matched-direction), the effect returns regardless of
> feature identity or direction. Off-target specificity is weak and class-dependent (pooled on
> **+2.89** vs off **+2.45** nat; geography features answer-specific, organization features not).
> **→ The group ablation does NOT establish that the decomposition recovers a causally privileged
> code. The pre-registered single-feature null is the real causal result and it stands. This removes
> the S3 "causally faithful in aggregate" gloss and sharpens its "mass-driven" finding into its
> honest conclusion.**

## (a) Norm-matched random-direction control — magnitude vs direction

Pooled mean answer-prob drop over 8 candidates × 100 examples (`tables/distributed_control.json`):

| k | top-k co-firing | count-matched random | norm-rand isotropic | norm-rand shaped | top mass |
|---|---|---|---|---|---|
| 1 | −0.006 | −0.000 | +0.008 | +0.003 | 15 |
| 3 | +0.005 | +0.000 | +0.028 | +0.014 | 40 |
| 10 | −0.003 | +0.003 | +0.070 | +0.048 | 109 |
| 30 | +0.033 | +0.011 | +0.133 | +0.103 | 257 |
| **100** | **+0.237** | +0.034 | **+0.224** | **+0.204** | 611 |
| 300 | +0.331 | +0.145 | +0.315 | +0.307 | 1141 |

- **Norm-matched random direction reproduces top-k** at every k (both isotropic and the
  contribution-subspace-shaped variant cross 0.20 at the same k=100). At low k the random directions
  even *exceed* top-k — the top-k δ there is a near-zero-norm perturbation (k=1 mass ≈15), so
  spreading that tiny norm in any direction does as much; the curves converge as the magnitude
  becomes behaviourally relevant.
- **The decomposition's feature directions are not causally privileged** over random directions of
  equal magnitude. The query-token residual stream is ~isotropically sensitive to perturbations of
  this size across L1–L22 — a generic sensitivity, not a feature-specific causal handle.
- **Re-reads the S3 controls.** Count-matched random staying below 0.20 (the S3 "rules out any ~100
  directions" result) is a **mass** effect, not a **direction** effect: count-matching removes the
  same *number* of (randomly chosen, lower-|u|) actives → less mass → less norm → less drop.
  Mass-matched random (+0.244 ≈ top +0.237) and norm-matched random direction both reproduce → the
  effect tracks residual magnitude, full stop.

## (b) Off-target specificity — gold answer vs other-class tokens

At k=100, drop in log-prob of each example's gold answer (on-target) vs a basket of the other two
relation classes' single-token gold answers (off-target), same forward pass:

| feature@L | class | on-target Δlogp | off-target Δlogp | diff |
|---|---|---|---|---|
| 847@L17 | geography | +3.72 | +2.14 | **+1.58** |
| 1494@L18 | geography | +3.72 | +1.54 | **+2.18** |
| 426@L22 | geography | +2.56 | +0.51 | **+2.05** |
| 123@L2 | geography | +2.82 | +2.57 | +0.24 |
| 328@L1 | organization | +2.19 | +2.67 | −0.48 |
| 297@L9 | organization | +2.63 | +3.03 | −0.41 |
| 786@L5 | organization | +2.38 | +3.73 | −1.35 |
| 496@L9 | organization | +3.08 | +3.38 | −0.30 |
| **pooled** | | **+2.89** | **+2.45** | **+0.44** |

Bimodal by class: the **geography→continent** features (847/1494/426) suppress the gold answer
1.5–2.2 nat more than off-class tokens (answer-specific); the **organization** features suppress
off-class tokens *as much or more* (non-specific / reverse). Pooled diff +0.44 nat is below the
0.5-nat "specific" bar → **non-specific suppression** overall, consistent with the magnitude reading
in (a): a generic perturbation drops the highest-prob token most, which is usually the gold one.

## Bottom line for the paper

The Phase-4 causal story is now cleanly negative and fully controlled:
1. **Single-feature existence proof: NULL** (pre-registered; stands).
2. **Distributed-code group ablation: a magnitude effect, not a feature-specific causal code** (E1a)
   — a norm-matched random direction (even isotropic) reproduces it; the decomposition's directions
   are not causally privileged.
3. **Weak, class-dependent semantic specificity** (E1b) — geography features answer-specific,
   organization features not.

The method's demonstrated contribution is **reconstruction / compression** (Step 1: SSM +0.867 >
matched per-layer +0.816 ≫ differenced baselines), **not** causal feature discovery. Write it that
way; do **not** frame the top-100 crossing as evidence of a faithful distributed code. The S3
diagnostics remain a correct *diagnosis* of the single-feature null (recall is not a single-feature
bottleneck) but, with E1, are no longer read as a positive causal result.

## Reproduce
```
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True scripts/phase4_distributed.py \
  --config configs/pythia-410m-deduped.yaml --desc ph4-e1-distributed-410md \
  --checkpoint-dir runs/2026-06-26-ph4-checkpoint-410md --device cuda --skip-single \
  --pool-candidates 8 --n-examples 100 --group-ks 1,3,10,30,100,300 --mass-match-k 100
```
