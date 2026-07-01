# Phase 4 S1 — gauge-invariant seed-stability (410m mlsae) — memo

**Run:** `runs/2026-06-29-ph4-gauge-410md/` · seeds 0/1/2 (seed 0 = canonical
`ph4-checkpoint` fit; 1/2 refit with the Gate-3 knee config, λ_L1=1, r=d_z=32, 10 outer) ·
top-200 features by full-dataset |u| mass · all three seeds reach **train R² 0.880–0.881**.

## Question (plan Fix 5)
The recorded Gate-3 limitation is that the **raw decoder column** `D_{ℓ*}[:, j]` matches
across seeds at only ~0.45 (< 0.7), despite identical reconstruction. Fix 5 asked whether
that is a **cosmetic gauge artefact**: do *gauge-invariant* quantities recover seed stability?
We measured three (the SSM is linear in `u`, so each feature's data-averaged Δ-contribution is
exactly its **impulse response** `g_j ∈ R^{L×D}` — feedthrough `D_{ℓ*}` **plus** the downstream
state path `C_ℓ A…A B_{ℓ*}`; its home-layer slice equals the raw `D` column).

## Result (pairwise over seeds 0/1/2)

| quantity | gauge status | cross-seed stability |
|---|---|---|
| raw decoder column `D[:,j]` | **variant** | match@0.7 **0.36**, mean cos 0.53 |
| per-feature Δ-contribution `g_j` (impulse) | **invariant** | match@0.7 **0.29**, mean cos 0.51 |
| top-100 decoder **subspace** overlap | invariant | raw-D 0.62, impulse 0.58 |
| `A_ℓ` eigenvalue-modulus spectrum | invariant (Symmetry A) | **cos 0.96** |
| contribution mass beyond home layer | — | 26% |

(The raw-D 0.36 here is the top-200 read; it includes lower-mass, less-stable features than
the seed-stable ~45% subset behind the recorded 0.45 — same ballpark, slightly lower by design.)

## Reading — the limitation is **not cosmetic, but it is correctly scoped**
- **Fix 5's optimistic hope is refuted at the per-feature level.** Making the metric
  gauge-invariant does **not** rescue per-feature stability — the impulse response is *as
  unstable* (0.29) as the raw column (0.36). So the instability is **not** merely the
  state↔feedthrough gauge split (Symmetry B): the per-*single-feature* function genuinely
  differs across seeds. Mechanically the feedthrough dominates (74% of each feature's
  contribution is the home-layer `D` write), so adding the 26% state path can't make an
  unstable object stable.
- **Why this is expected, not a failure.** A single-feature impulse is **off the data
  manifold** (features co-fire in correlated combinations; recall is a *distributed*,
  mass-driven code — Step-3 + S3). The fit is only constrained on realized combinations, so
  which redundant TopK atom carries an example's mass is interchangeable across seeds →
  per-feature attribution is underdetermined while the *aggregate* map is pinned. This is the
  signature of a redundant TopK dictionary, tying S1 directly to **Fix 6** (mlsae-TopK source
  is the culprit; L1 sources are seed-stable at 0.81/0.95) and **Step 3** (no privileged
  single-feature subset).
- **Everything the paper actually claims is seed-stable.** Reconstruction R² (0.880, all
  seeds), the cross-layer **state dynamics** (`A` eigenvalue spectrum **cos 0.96**), and the
  top-feature **subspace** (~0.6 shared) are all stable. The paper claims (a) reconstruction
  above the per-layer baseline, (b) structured cross-layer propagation, and (c) a *distributed*
  causal code — **none** rests on single-feature decoder identifiability, which is the only
  thing S1 finds unstable.

## Verdict
S1 **closes** the seed-stability story — not via "it was an illusion", but by localising the
instability to single-feature *attribution* (an intrinsic property of a redundant distributed
TopK code) and showing the system-level invariants the claims depend on are seed-stable. The
honest write-up line: *per-feature decoder directions are not seed-identifiable on the 410m
mlsae source (a gauge-invariant fact, consistent with the distributed code and the TopK-source
diagnosis), but reconstruction, state-dynamics spectrum, and the feature subspace are.*

Artifacts: `tables/gauge_stability.json`, `tables/gauge_pairs.csv`,
`figures/gauge_stability.svg`, refit checkpoints `checkpoints/ssm_seed{1,2}.pt`.
