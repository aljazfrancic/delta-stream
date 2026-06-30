# T5 — analytic tautology floor (Phase 4 H1-validity)

**Date:** 2026-06-30. **Model:** pythia-410m-deduped. Code:
`autoregress.tautology_floor_decomposition` (+2 tests), `scripts/phase4_t5_taut_floor.py`.
Runs: factual `runs/2026-06-30-ph4-t5-taut-floor-410md/`, generic
`…-410md-general/` (store `data/pythia-410m-deduped-general/residual_stream.h5`).

> **What T5 is.** The explicit closed-form companion to T1. T1 *refits* the AR after
> stripping the persistent state directions and reports the surviving R² (0.44 factual
> / 0.29 generic). T5 instead decomposes the **single headline AR(k=20) fit**: its
> held-out explained variance splits **orthogonally (Pythagorean)** over the top-m
> persistent state subspace (`par`, tautological) and its complement (`perp`, genuine),
> and the two add back to the headline R² *exactly*:
>
>     R²_full = frac_par + frac_perp        (exact at every m)
>
> A first wrong turn worth recording: the naive "predict −v_ℓ" floor is **scale-
> degenerate** — ‖v_ℓ‖≈140 ≫ ‖Δ_ℓ‖, so Δ̂=−v̂_ℓ scores wildly negative. The tautology
> lives in the persistent *directions* (smooth running-sum drift), not in Δ≈−v, which
> is why the subspace projection — not a −v_ℓ regression — is the right closed form.

## Results

**Factual (R²_full = 0.6497):**

| m | Δ-var in subspace | frac_par (tautology floor) | frac_perp (genuine) | r2_par | r2_perp |
|---|---|---|---|---|---|
| 1 | 0.044 | 0.041 | 0.609 | 0.933 | 0.637 |
| 5 | 0.269 | 0.254 | 0.395 | 0.947 | 0.540 |
| 20 | 0.455 | 0.420 | 0.229 | 0.924 | 0.421 |
| 50 | 0.580 | **0.520** | **0.130** | 0.896 | 0.309 |

**Generic (R²_full = 0.9346):**

| m | Δ-var in subspace | frac_par (tautology floor) | frac_perp (genuine) | r2_par | r2_perp |
|---|---|---|---|---|---|
| 1 | 0.843 | **0.842** | 0.092 | **0.999** | 0.588 |
| 5 | 0.918 | 0.900 | 0.035 | 0.981 | 0.420 |
| 20 | 0.957 | 0.931 | 0.004 | 0.972 | 0.095 |
| 50 | 0.968 | **0.937** | **−0.003** | 0.969 | −0.078 |

## What it shows

- **The decomposition is exact** (sum = R²_full at every m, both stores) — the
  invariant the unit test pins. So `frac_par` is a clean, explicit subtracted baseline.
- **The persistent subspace is near-trivially predictable** — `r2_par` ≈ **0.90–0.95**
  (factual) and **0.97–0.999** (generic). That near-perfect predictability *is* the
  ±v_ℓ tautology: consecutive Δ's agree along the smooth running-sum drift for free.
- **Factual has substantial genuine structure; generic is almost all tautology.** At
  m=50 the factual headline 0.65 splits 0.52 tautological + 0.13 genuine (single fit),
  whereas the generic headline 0.935 is **0.94 tautological + ≈0 genuine** — its single
  top persistent direction alone (m=1) already accounts for **0.84** of the headline at
  r2_par **0.999**. This is the exact quantification of the standing E4/T1 caveat: the
  generic R² *level* (0.935 > factual 0.65) is **persistence, not more computation**.

## Reconciling T5 with T1 (report both, they answer different questions)

- **T1** *refits* a kernel on the persistence-removed Δ → genuine **0.44 factual / 0.29
  generic** (within-complement, re-optimised). The "irreducibly tautological" share is
  `full − T1` = **~0.21 factual / ~0.64 generic**.
- **T5** attributes the *one* headline fit, no refit → single-fit complement
  `frac_perp` = **0.13 factual / ≈0 generic**. Lower than T1 because the headline kernel
  is not re-optimised for the complement; the gap (T1 − T5_perp) is what refitting
  recovers, and it is large for generic (the single fit dumps ~all generic predictability
  into the persistent subspace).
- **Robust joint conclusion:** the **binary** H1 claim is tautology-robust on both
  stores (T2 surrogate ≈0, T3 lag≥2 Granger 89/89), and genuine non-persistent depth
  structure is **substantial for factual** (T1 0.44; T5 frac_perp 0.13) and **marginal
  for generic** (T1 0.29; T5 frac_perp ≈0). Quote the residual/decomposed numbers, never
  the raw 0.935, as "more computation."
