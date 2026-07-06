# Phase 4 — E4: H1 diagnostics on generic (non-factual-recall) text

**Date:** 2026-06-30. **Model:** pythia-410m-deduped (L=24, D=1024). **Runs:**
`runs/2026-06-30-ph4-e4-collect-410md` (collection) + `runs/2026-06-30-ph4-e4-general-410md`
(analysis). **Baseline:** the Phase-1 factual-recall Step-5 run `runs/2026-06-22-ph1-step5-410md`.

E4 asks the question §7 of the proposal concedes is open: is the above-null **cross-layer Δ
structure (H1) specific to the narrow factual-recall circuit, or general to transformer depth?**
It re-runs the *identical* Phase-1 Step-5 analysis — pooled AR(k) on Δ over depth + the
per-(depth, lag) Granger nested-F + scrambled Null 1 — on generic **Pile** text instead of factual
prompts. Same code (`scripts/step5_ar.py --h5 …`), same config (k∈{1,2,3,5,10,15,20}, rank sweep,
256 predictor PCs, Granger q=32), same N and the same train/test split sizes.

> **⚠ STATUS UPDATE (2026-07-06 — annotated after T1–T5/T4; the body below is the dated
> 2026-06-30 record, unedited).** The verdict's binary claim stands, but its "*in fact
> stronger* (0.935 vs 0.650)" reading was quantified away the same week: **the generic R²
> level is ≈ all ±v_ℓ persistence** (T5: generic 0.935 ≈ 0.94 tautological + ≈0 genuine;
> factual 0.65 = 0.52 + 0.13; T1: after stripping ~all state variance the genuine
> residual is **0.29 generic vs 0.44 factual** — i.e. *weaker*, not stronger). And T4:
> the tautology-free structure is substantially **architectural** on both stores (random
> init matches). The task-generality conclusion itself is *reinforced* by both (E7
> extends it to H2 + k\*), but quote the persistence-stripped numbers, not 0.935. See
> `runs/2026-06-30-ph4-{pressure,t5-taut-floor}-410md{,-general}` + progress.md (T1–T5).

> **Verdict — H1 GENERALISES. The cross-layer Δ structure is a general property of transformer
> depth, not an artefact of factual recall.** On generic Pile text the pooled AR(k) explains
> **R² = 0.935** of held-out Δ variance (best k=20; k=1 alone already 0.882), the scrambled null
> collapses to **≈0** (−0.0006), and the Granger grid is **112/112** Bonferroni-significant vs
> **0/112** under the null — the same clean H1 signature as factual recall, and in fact *stronger*
> (factual AR R² 0.650). The two tautology-free reads (null collapse + Granger localisation)
> replicate **identically**. This directly answers the "narrow circuit" critique: the structured
> propagation the paper characterises is not specific to the factual-recall query position.

## Result — generic text vs factual recall (identical pipeline)

Pooled AR(k) held-out R² on Δ_ℓ (vs the per-depth mean μ_ℓ), best rank, real vs scrambled null;
Granger = # of (depth × lag) cells Bonferroni-significant out of 112.

| | N (train/test) | AR best R² (k) | k=1 R² | scrambled null | real − null | Granger sig | non-stat flag |
|---|---|---|---|---|---|---|---|
| **Factual recall** (Ph1) | 1244 / 311 | 0.650 (k=20) | — | ≈0 (7e-5) | 0.650 | **112/112** vs 0 | False |
| **Generic Pile text** (E4) | 1244 / 311 | **0.935** (k=20) | 0.882 | ≈0 (−6e-4) | **0.935** | **112/112** vs 0 | False |

Full per-k curve (generic, real vs null): k=1 0.882/−0.002 · k=2 0.900 · k=3 0.909 · k=5 0.922 ·
k=10 0.887 · k=15 0.911 · k=20 0.935/−0.001. Every k clears the null by ~0.9. See
`tables/ar_r2.csv`, `tables/granger_real.csv`, `figures/{ar_r2_vs_k,granger_real,granger_null}.svg`.

## Interpretation — what replicates, and the honest read on the *level*

- **The binary H1 conclusion replicates exactly.** Above-null, example-specific, depth-localised
  cross-layer Δ structure is present in generic next-token prediction: the null collapses both AR
  (≈0) and Granger (0/112), and Granger is 112/112 significant out to lag 20 — the identical
  stationarity-free signature recorded for factual recall and now replicated across 6 models / 3
  families (cross-model memo). So H1 is **not** a property of the factual-recall circuit.
- **The AR R² is *higher* on generic text (0.935 vs 0.650) — do not over-read this as "more
  structure."** Per the standing tautology caveat (`cca-vdelta` memory; Phase-2 Step-2 note), the
  AR-on-Δ R² is inflated by the mechanical overlap Δ_ℓ ⊃ −v_ℓ ⊂ Δ_{ℓ−1}: a larger, more stable
  −v_ℓ component raises the score without implying more genuine computation. Generic last tokens
  (common words, punctuation, code) plausibly have **more stereotyped, lower-entropy** residual
  trajectories than the high-information factual query position — exactly the regime that enlarges
  the tautological term. The robust, tautology-free reads (null collapse, Granger localisation) are
  what carry the H1 claim, and they replicate. So the defensible statement is *"H1 holds on generic
  text"*, **not** *"Δ is more predictable on generic text in a computationally meaningful sense."*
- **Consistent with the token-position result (Ph2 Step 3).** That result showed the *kernel* is
  query-position-specific (cross-position R² negative); E4 shows the *existence* of above-null
  structure is position-general. No contradiction — a different kernel, but cross-layer structure
  either way. (E4 did not refit a generic-vs-factual cross-kernel-transfer test; the token-position
  study already establishes kernels do not transfer across positions.)

## Method notes (provenance / why the comparison is clean)

- **Length-matched, N-matched.** The generic windows reuse the factual content-token length multiset
  exactly (1555 windows; lengths 4–24, median 9), one window per **distinct** Pile document at a
  random in-document offset (`data/general_text.length_matched_windows`, 6 unit tests). So the only
  thing that differs from Phase 1 is factual-recall vs generic text — **not** context length, N, the
  split, or the analysis code.
- **Source.** `NeelNanda/pile-10k` — the Pile is Pythia-410m-deduped's training distribution, so
  this is in-distribution generic text. Sample windows: `" do I code the clientvalidation"`,
  `"y) = -y**2. Let"`, `" 2 percent of survey respondents said"` — clearly non-factual mid-stream
  contexts.
- **Sanity check (generic analog of the Step-2 top-1 check).** For 5 random windows, ln_final +
  unembed of v_L recovers the model's **own argmax next-token** — **5/5 OK** — validating the
  extraction without a gold label.
- **Reproduce.** `scripts/collect_general_text.py --config configs/pythia-410m-deduped.yaml` then
  `scripts/step5_ar.py --config … --h5 data/pythia-410m-deduped-general/residual_stream.h5`.

## Bottom line for the paper

E4 closes the narrow-circuit hole: H1 (structured, above-null, depth-localised cross-layer Δ
dynamics) is a **general property of transformer depth**, demonstrated on generic Pile text with the
identical pipeline that established it on factual recall, and at least as strong. Phase 2's
characterization (non-stationary, query-position-specific *kernel*) scopes *which* kernel applies
where; E4 establishes the *phenomenon* is not factual-recall-specific. Both outcomes were
pre-judged publishable; this is the generality-confirming one, and it strengthens the spine of the
paper (reconstruction + characterization of a general structure) rather than a narrow circuit.
