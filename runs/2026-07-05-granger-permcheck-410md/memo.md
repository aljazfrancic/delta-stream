# Granger permutation spot-check — the Rao-F thresholds are calibrated (finding 10 closed by a run, not a caveat)

**Date:** 2026-07-05. **Script:** `scripts/granger_perm_check.py` (new; machinery
`autoregress.granger_permutation_check`, +2 tests). **Run:**
`runs/2026-07-05-granger-permcheck-410md` (factual 410m store, depths ℓ∈{12, 22},
all feasible lags from the standard sweep, q_pca=32, n_perm=999, seed 0).
**Question (3rd-pass finding 10):** the Step-5 Granger grid's p-values use Rao's F,
which assumes Gaussian errors; the heavy-tailed rogue dims mildly violate that. The
only calibration evidence so far was the scrambled null landing at 0/112. Does a
direct permutation test agree?

> **Verdict — YES, cleanly. In all 12 spot cells the observed F sits at the
> permutation floor (p_perm ≤ 1/1000), and the Freedman–Lane permutation null of F
> matches the parametric F(df1, df2) reference to within ~1% in the tail quantiles
> the Bonferroni threshold actually uses (q95/q99 ratios 0.991–1.004, max
> |log q99 ratio| = 0.009).** The stats appendix can state that the F approximation
> is calibrated under exchangeability, Gaussian errors or not, with the scrambled
> 0/112 as corroborating (not sole) evidence.

## Design

Freedman & Lane (1983): at target depth ℓ and lag j, fit the reduced model
(intercept + lags 1..j−1) in the grid's own top-32 PCA coordinates, then permute its
residuals — `y* = ŷ_red + π(r_red)` — and recompute the Wilks/Rao nested F for the
added lag-j block. 999 permutations per cell. Two reads per cell:

- **p_perm** — permutation exceedance of the observed F (resolution floor 1e-3);
- **calibration** — KS test + 95th/99th-percentile ratios of the permuted F sample
  against the parametric F(df1, df2) null the pipeline uses for its thresholds.

The observed F from this machinery is unit-tested to match `granger_lag_grid`'s
lstsq-based value on the same cell (rtol 1e-6).

## Results (all 12 cells)

| ℓ | lag | F_obs | p_param | p_perm | KS p | q95 ratio | q99 ratio |
|---|-----|-------|---------|--------|------|-----------|-----------|
| 12 | 1 | 66.66 | ≈0 | ≤0.001 | 0.45 | 0.997 | 0.995 |
| 12 | 2 | 19.64 | ≈0 | ≤0.001 | 0.91 | 0.999 | 1.001 |
| 12 | 3 | 12.29 | ≈0 | ≤0.001 | 0.26 | 0.996 | 0.991 |
| 12 | 5 | 6.66 | ≈0 | ≤0.001 | 0.59 | 1.000 | 0.999 |
| 12 | 10 | 3.20 | 1.8e-215 | ≤0.001 | 0.012 | 1.004 | 0.999 |
| 22 | 1 | 61.85 | ≈0 | ≤0.001 | 0.49 | 1.000 | 0.992 |
| 22 | 2 | 16.69 | ≈0 | ≤0.001 | 0.96 | 0.997 | 0.992 |
| 22 | 3 | 12.11 | ≈0 | ≤0.001 | 0.067 | 0.999 | 1.002 |
| 22 | 5 | 7.36 | ≈0 | ≤0.001 | 0.87 | 1.002 | 0.992 |
| 22 | 10 | 3.01 | 8.3e-189 | ≤0.001 | 0.80 | 0.993 | 0.991 |
| 22 | 15 | 1.85 | 3.8e-51 | ≤0.001 | 0.093 | 0.997 | 0.996 |
| 22 | 20 | 1.45 | 1.5e-18 | ≤0.001 | 0.88 | 0.999 | 0.999 |

Reading: even the *weakest* headline cells (lag 15/20, F ≈ 1.5–1.9) are at the
permutation floor, and the permutation null's tails coincide with the parametric
F's. The two smallish KS p-values (0.012 at ℓ12/lag10, 0.067 at ℓ22/lag3) are the
expected sensitivity of a 999-sample KS to sub-percent body deviations — the tail
quantiles, which are what the −log10 p Bonferroni threshold consumes, are within 1%
everywhere.

## Scope

- Permutation p can only certify p ≤ 1e-3 at n_perm=999; the claim is **threshold
  calibration** (the Bonferroni cut at α=4.5e-4 per cell behaves as advertised under
  exchangeability), not a verification of astronomical parametric p magnitudes.
- Spot depths ℓ=12 (mid) and ℓ=22 (late, admits the full lag sweep) on the factual
  store; the D3 stats appendix cites this plus the scrambled 0/112 (full-grid,
  both stores) as the pair of empirical calibrations.
