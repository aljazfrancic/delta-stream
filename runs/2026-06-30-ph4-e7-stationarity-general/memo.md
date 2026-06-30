# Phase 4 — E7: H2 (stationarity) + k* on generic text (completing E4)

**Date:** 2026-06-30. **Model:** pythia-410m-deduped. **Runs:**
`runs/2026-06-30-ph4-e7-stationarity-general` (H2, `step1_stationarity.py --h5 <general> --k 5
--rank 256`) + `runs/2026-06-30-ph4-e7-kstar-general` (k*, `step2_kstar.py --h5 <general> --rank
256`). **Store:** the E4 length-matched generic Pile residuals `data/pythia-410m-deduped-general/`.

E4 showed **H1** (above-null cross-layer Δ structure) generalises from factual recall to generic
text. E7 completes that: does the *rest* of the characterisation — **H2 (non-stationarity)** and the
**low effective depth k*** — also generalise, or is only H1's *existence* general? Identical Phase-2
Step-1/Step-2 pipelines, same (k=5, r=256) as the factual runs.

> **Verdict — the WHOLE characterisation generalises, not just H1.** On generic text the AR-on-Δ
> kernel is **non-stationary** (within-band R² positive 0.98/0.42/0.91, but **every cross-band R²
> negative**, worst −144; rank-robust 32→256; bands bootstrap-separated 39–188 SD, between-band
> Frobenius ≈√2 = near-orthogonal) — the same H2-fails signature as factual recall, and *more*
> emphatic. Effective depth is **low-order** (median k*=2, vs factual 1; null collapses to 0). So
> the structured-but-non-stationary, low-order, query-position cross-layer dynamics the paper
> characterises are a **general property of transformer depth**, demonstrated on non-factual text
> with the identical pipeline.**

## E7a — stationarity (H2) on generic text

Within-band AR(k=5, r=256) R² (positive) vs cross-band transfer R² (fit kernel on row band, evaluate
on column band):

| fit ↓ / eval → | early | middle | late |
|---|---|---|---|
| **early** | **0.982** | −144.5 | −0.07 |
| **middle** | −0.16 | **0.423** | −0.044 |
| **late** | −2.40 | −33.9 | **0.911** |

- **Every off-diagonal is negative** — kernels do not transfer across depth bands at all (worse than
  the per-depth mean). Rank-robust: worst off-diagonal cross-band R² = −137 (r32) → −144 (r256).
- **Bands are bootstrap-separated** (n_boot=200): between-band Frobenius 1.41–1.75 (≈√2, near-
  orthogonal); between-vs-within z = 48.7 (early-middle), 188 (middle-late), 39 (early-late).
- **Same signature as factual, amplified.** Factual H2 cross-band R² were ≈−1 to −3; generic text is
  far more negative (the middle band's kernel is wildly mis-scaled for the other bands), so generic
  text is *even less* stationary across depth. H2 fails decisively on generic text too.

(The summary `stationarity_stats.json` + figure were skipped by a since-fixed bug — `ksel` was
unbound when `--granger-csv` is omitted, the sibling of the `sat` fix; the CSVs `cross_band_r2.csv`,
`rank_robustness.csv`, `bootstrap.csv`, `per_lag_frobenius.csv` were all written before it and carry
the full result above.)

## E7b — effective depth k* on generic text

| | median k* | max k* | band k* (early/mid/late) | null best R² | median +lags≥2 |
|---|---|---|---|---|---|
| factual (pooled) | 1 | 2 | 1 / 1 / 1 | ≈0 | +0.018 |
| **generic text** | **2** | 10 | 1 / 3 / 1 | ≈0 (0.000) | +0.040 |

- **Low-order on generic text too**, slightly higher and noisier than factual: median k*=2 (vs 1),
  driven by the middle band (k*=3) and a few high-variance depths (k* spikes to 5/10). The scrambled
  null collapses to 0, so the structure is real.
- **Honest nuance (per-depth vs pooled).** Per-depth single-target reconstruction R² is much lower on
  generic text (≈0.1–0.5) than factual (0.83–0.94), even though E4's *pooled* shared-kernel AR R² was
  *higher* on generic text (0.935 vs 0.65). These are consistent: the pooled fit shares one kernel +
  per-depth intercept across all depths (much more data, dominated by the ±v_ℓ tautology that is
  large on stereotyped generic last tokens), while each single-depth fit has only N=1244 rows for 256
  predictor PCs and the generic per-depth Δ is genuinely noisier. The robust reads (null collapse,
  low k*, non-stationarity) all hold; the higher k* and lower per-depth R² are the honest "generic
  text is noisier per depth" caveat.

## Bottom line for the paper

With E4, E7 establishes that **the full Phase-1/2 characterisation — above-null cross-layer structure
(H1), non-stationarity across depth (H2-fails), and low effective depth (k*≈1–2) — is general to
transformer depth, not an artefact of the factual-recall circuit.** The factual-recall scope applies
to the *specific kernels* (circuit-specific, query-position-specific — Phase-2 Step-3/4, Phase-4
Step-2), not to the *phenomenon*. This is the strongest available answer to the §7 "narrow circuit"
critique.
