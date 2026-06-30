# Phase 4 — E6: effective depth k* stratified by relation class

**Date:** 2026-06-30. **Model:** pythia-410m-deduped. **Run:**
`runs/2026-06-30-ph4-e6-kstar-byclass-410md` (`scripts/step2_kstar.py --by-class --rank 256`).

[phase-2 Step 2 item 4] and [proposal §5.4] both call for reporting the effective kernel depth k*
**as a function of relation class** ("compare it across relation types … to see whether it is a
property of transformers in general or of specific circuits"). The pooled Phase-2 result (k*≈1) was
never stratified. E6 refits the per-depth and per-band k* sweep within each relation class.

> **Verdict — k*≈1 is class-invariant.** Every relation class gives **median k*=1** at every depth
> band (early/middle/late all k*=1), with the scrambled null collapsing to ≈0. The reconstruction-
> effective depth — a single preceding update captures essentially all held-out predictable Δ_ℓ
> variance — is a **general property of the residual stream at the query position, not a
> circuit-specific quantity.** This complements the cross-class *reconstruction* finding (circuit-
> specific kernels, Phase-4 Step-2): the kernels differ across classes, but their **order** (k*≈1,
> first-order Markov-in-depth) is the same.**

## Result — per-class k* (rank 256, same sweep as pooled)

| class | n | median k* | max k* | band k* (early/mid/late) | median +lags≥2 R² | null |
|---|---|---|---|---|---|---|
| pooled | 1555 | 1 | 2 | 1 / 1 / 1 | +0.018 | ≈0 |
| geography | 511 | **1** | 3 | 1 / 1 / 1 | +0.037 | +0.000 |
| language | 333 | **1** | 3 | 1 / 1 / 1 | +0.025 | +0.000 |
| organization | 711 | **1** | 2 | 1 / 1 / 1 | +0.038 | +0.000 |

`tables/kstar_by_class.csv`, `tables/step2_stats.json` (`by_class`).

## Interpretation

- **First-order is universal across these circuits.** k*=1 holds in every class and every depth band.
  Lags ≥ 2 add only a small median increment (+0.025…+0.038 R² per class, marginally above the pooled
  +0.018), and the scrambled null collapses to ≈0 throughout, so the k*=1 reads are real
  example-specific structure, not noise. Branch B's first-order (Markov-in-depth) state is the right
  order for every relation class, not just on average.
- **Order vs kernel — the honest distinction.** E6 says the *effective depth* is class-invariant; it
  does **not** say the *dynamics* are class-general. Phase-4 Step-2 (cross-class reconstruction) and
  Phase-2 Step-1/3 (cross-band, cross-position) already established the kernels are circuit-specific
  and non-stationary. So: same model **order** (k*≈1) everywhere, different **kernels** per class —
  consistent with the paper's "structured but circuit-specific, non-stationary" characterisation.
- **Same ±v_ℓ tautology caveat as the pooled result.** k*≈1 is substantially the mechanical
  Δ_ℓ ⊃ −v_ℓ ⊂ Δ_{ℓ-1} overlap (cca-vΔ); it is a property of the AR-on-Δ parameterisation, and the
  class-invariance of k* inherits that caveat. The robust, tautology-free statement is the null
  collapse (real structure) + the class-invariant *order*.

## Caveat

Per-class fits use rank 256 (the pooled working rank) on smaller N (geography 511 / language 333 /
organization 711), so per-class absolute R² is noisier than pooled; k* (a saturation-point statistic,
robust to rank and N) is the reported quantity and is stable at 1 across all classes and bands.

## Bottom line

E6 closes the pre-registered "k* across relation types" check: the first-order effective depth is
class-invariant. The paper can state k*≈1 as a property of query-position factual recall in general,
with circuit-specificity living in the *kernels*, not the *order*.
