# Gate 1 decision memo — Phase 1 Diagnostic

**Model:** Pythia-410m-deduped (L=24, D=1024). **Dataset:** 1,555 factual-recall
prompts, 3 relation classes (CounterFact, top-1==object, prob>0.1), residual stream
collected at the final prompt token → `residual_stream[1555, 25, 1024]`.
**Date:** 2026-06-22. **Seed:** 0. **Split:** 1,244 train / 311 test (shared across
Steps 3–5). Figures in `figures/`, all numerics in `tables/`.

> **⚠ STATUS UPDATE (2026-07-06 — annotated after the H1 pressure-test battery; the body
> below is the dated 2026-06-22 record, unedited).** The Gate-1 decision and route stand,
> but three later results correct how this memo's numbers must be *read* (D3 leads with
> the corrected story, per the review hazards):
> **(1) The raw AR R² levels ARE partly the ±v_ℓ tautology.** Point 7's "not explained by
> the residual-stream tautology" was too strong: the per-depth intercept removes only the
> *mean* persistent component, not the example-specific one. T5 (2026-06-30) splits the
> headline exactly: **factual 0.65 = 0.52 tautological + 0.13 genuine** (generic 0.935 ≈
> all tautological). The tautology-robust reads are T1 (R² **0.44** after stripping 87% of
> state variance ≫ scramble ≈0), T2 (persistence-matched surrogate ≈0), and T3 (lag≥2
> Granger **89/89** vs 0/89) — the binary H1 claim survives; quote those, not 0.65 alone.
> **(2) The surviving structure is substantially ARCHITECTURAL, not learned** (T4
> 2026-06-30: a random-init 410m matches the persistence-stripped R², 0.50/0.22 vs
> 0.44/0.29; R6 2026-07-03 replicates on GPT-2, learned share ~5%) — H1's depth structure
> is a property of the architecture, general across tasks (E4/E7) and training. The
> *learned* content lives in the Branch-B reconstruction win (~half learned, T4-recon/R10).
> **(3) Robustness later added:** object-disjoint split (R5: AR 0.561, Granger 112/112
> identical), tautology-free sublayer streams (R8: attn/mlp writes carry no ±v_ℓ term,
> both reproduce H1; k\* contrast mlp=1 vs attn→10), and a Freedman–Lane permutation check
> calibrating the Granger F thresholds (2026-07-05). See progress.md (T1–T5 · T4 · R5 ·
> R6 · R8) + `runs/2026-06-30-ph4-pressure-410md{,-general}`, `runs/2026-07-05-r8-*`.

> Headline: **H1 (depth-local predictability of Δ_ℓ) is cleanly supported.** The
> layer-to-layer update Δ_ℓ = v_{ℓ+1} − v_ℓ is linearly predictable far above trivial
> and calibrated-lens baselines, has autoregressive structure over depth reaching back
> ≥15 layers, and both signals survive the scrambled null **and** the context-
> perturbation control. **Route: proceed to Phase 2.**

---

## The seven Gate-1 questions

**1. Is Δ_ℓ predictable from v_ℓ above the trivial baselines (zero / mean / identity)?
— YES.** (Step 3) Low-rank ridge gives layer-averaged held-out **R² = 0.602** (best
rank 128; full-rank 0.921) against mean-prediction R² = 0 and zero-prediction
R² = −1.54. The identity transparency baseline (regress v_{ℓ+1} on v_ℓ) sits at
R² = 0.965 as expected and is reported for transparency only, not as a comparator.

**2. Does the ridge fit beat the Tuned-Lens-implied update? — YES.** (Step 3, the
proposal-mandated precondition for Step 5.) Ridge **0.602** vs Tuned-Lens-implied
**−0.691**, layer-averaged on the same split. The cheap diagnostic separates cleanly
from a calibrated per-layer lens, so the AR test is justified on the budget.

**3. Is there shared subspace structure between v_ℓ and Δ_ℓ? — YES.** (Step 4) Shrinkage
CCA: held-out leading canonical correlation ≈ **0.9997** at every layer, robust to
γ→0.99, collapsing to ≈0 under the scrambled null (real−null = 1.006). *Caveat (logged
at Step 4):* the ≈1 is substantially the definitional −v_ℓ ⊂ Δ_ℓ term amplified by a
few high-variance residual dims, not structure the ridge missed — so the substantive
predictability claim stays the Step 3 R² vs mean/TL, and CCA only confirms Q3.

**4. Is there AR structure over depth? — YES.** (Step 5) A single low-rank kernel
shared across depth with per-depth intercept μ_ℓ predicts Δ_ℓ from the k preceding
updates with held-out **R² = 0.65** at k=20 (rank 128; ≥0.47 at every k in
{1,2,3,5,10,15,20}). This is measured against the per-depth mean, so it is structure
*beyond* the per-depth bias, and it is comparable to (slightly above) the Step-3
predict-from-v_ℓ R² of 0.602. The per-(depth,lag) nested-F (Wilks Λ, Rao's F, on a
top-32 PCA projection) is Bonferroni-significant in **112/112** feasible cells
(α=4.5e-4); significance reaches the longest lag tested (lag 20, −log10 p ≈ 15–18),
so the depth dependence is long-range, not just nearest-neighbour.

**5. Do the Granger structures survive the scrambled null? — YES.** (Step 5, Null 1)
Permuting the example axis independently per depth collapses the AR fit to
**R² ≈ 0.0000** at every k (best 7e-5) and the Granger grid to **0/112** significant
cells. The real signal is example-specific cross-depth structure, not a marginal
artefact.

**6. Do they survive the context-perturbation control? — YES.** (Step 5, Null 2 — the
critical control.) Token-level construction: a 6-token neutral preamble ("In school I
learned that ") vs the same length with its first 5 content words replaced by random
single-token function words, the boundary whitespace and the subject-relation span held
at identical absolute positions (alignment 1000/1000). On the 863/1000 examples where
the wrapped baseline still recovers the correct top-1: AR R² is **unchanged** (wrapped
0.639 → perturbed 0.696) and the Granger −log10 p heatmaps correlate at **0.964**
(wrapped 112/112 significant, perturbed 109/112 — the only losses are three marginal
lag-20 cells at depths 21–23). The cross-layer dynamics at the query token are a
property of subject-relation processing, **depth-intrinsic**, not a leaked-context
artefact.

**7. Honest conclusion about H1.** All six diagnostic questions are YES, including both
mandatory nulls. Δ_ℓ has structured, linearly-predictable, autoregressive cross-depth
dynamics that are not explained by the residual-stream tautology (ruled out by the
TL baseline and the per-depth-intercept AR target), not a finite-sample artefact (ruled
out by the scrambled null), and not context leakage (ruled out by the perturbation
control). The pooled AR R² is high and the non-stationary-H1 escape flag is **false**,
so we are *not* in the "low pooled R² + wide Granger significance" non-stationary
corner — the structure is consistent with approximate AR-stationarity over depth, which
Phase 2 Step 1 will test directly.

---

## Pre-registered routing

**☑ Proceed to Phase 2 — (1,2,4,5,6) all yes, H1 cleanly supported.**

The other branches (context-dependent pivot; non-stationary-H1; no-TL-separation paper;
negative-result paper) do not apply — each requires at least one of the above to fail.

**Caveats carried into Phase 2.**
- *Dataset scale.* 1,555 examples / 3 classes is below the proposal's ≥1000-per-class
  target (410m confidently knows only ~2,000 CounterFact facts). Phase-1 diagnostics
  are robust at this size; Phase 2 Step 4 (cross-relation generalisation) needs scaling
  (union Pararel/T-REx or a larger model) before its per-class claims hold.
- *Compute approximations.* The AR predictors are compressed to the top-256 Δ PCs
  (≥ Step-3's saturating rank 128; target stays full-D, so R² is in full Δ space) to
  keep the k·D SVD tractable on the dev GPU; the Granger nested-F runs on a top-32 PCA
  projection because D=1024 ≫ N makes the full-D Wilks Λ singular. Both are documented,
  conservative reductions; neither changes the qualitative result.
- *Single token position.* All trajectories are the final prompt token. Phase 2 Step 3
  tests position-invariance explicitly — not assumed here.

**Phase 2 expectation.** Pooled AR R² high + non-stationary flag false ⇒ enter Phase 2
Step 1 stationarity test without pre-committing the branch; block-Toeplitz (Branch A) is
plausible but the principal-angle / cross-band tests decide it.

## Reproduce

```
scripts/step3_ridge.py        --config configs/pythia-410m-deduped.yaml --desc ph1-step3-410md
scripts/step4_cca.py          --config configs/pythia-410m-deduped.yaml --desc ph1-step4-410md   --step3-csv <step3>/tables/ridge_r2.csv
scripts/step5_ar.py           --config configs/pythia-410m-deduped.yaml --desc ph1-step5-410md    --step3-csv <step3>/tables/ridge_r2.csv
scripts/step5_context_null.py --config configs/pythia-410m-deduped.yaml --desc ph1-step5ctx-410md --n-context 1000
```
