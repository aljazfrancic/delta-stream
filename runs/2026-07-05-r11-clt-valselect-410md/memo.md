# R11 — validation-selected λ for the CLT family: the SSM−CLT CI survives clean selection (finding 4 closed by a run)

**Date:** 2026-07-05. **Script:** `phase4_baseline_clt.py` with the new `--val-frac`
(λ_reg selected per family on a train-internal validation carve-out; test untouched
by selection). **Run:** `runs/2026-07-05-r11-clt-valselect-410md` — 410m frozen mlsae
sources, rank 32, 180 epochs, grid λ∈{0.3, 1, 3, 10, 30}, split **1057 fit / 187 val
/ 311 test** (val carved from the canonical 1244-train; canonical test unchanged),
seed 0. **Question (4th-pass finding 4):** the CLT-family scripts picked each arm's
λ_reg by best *test* R² and quoted the paired CI at those points. Does the one CI
that matters (SSM−CLT) survive selection that never sees the test split?

> **Verdict — YES. Every arm's validation-selected λ\* is interior (λ\*=3 for all
> three), and the paired CIs at the validation-selected operating points reproduce
> the test-selected conclusions: SSM−CLT −0.0090 [−0.0136, −0.0045] (CLT keeps a
> small real edge at 6.27× the params), SSM−per_layer +0.0518 [+0.0485, +0.0552],
> CLT−per_layer +0.0608 [+0.0563, +0.0655].** The R2 merged test-selected gap was
> −0.0142 [−0.0185, −0.0099] and R1's grid-fixed value −0.0105 [−0.0148, −0.0061] —
> the validation-selected −0.0090 sits inside/adjacent to those intervals, slightly
> *smaller* in magnitude, i.e. test-selection had (mildly) flattered the CLT, not the
> SSM. The D3 methods sentence can now cite this run instead of carrying a caveat.

## Results (each arm at its validation-selected λ\*)

| arm | params | λ\* (val) | train R² | val R² | test R² |
|---|---|---|---|---|---|
| per_layer | 3.15M | 3 | +0.7827 | +0.7386 | +0.7360 |
| ssm | 8.36M | 3 | +0.8486 | +0.7902 | +0.7879 |
| clt_full | 52.4M | 3 | +0.8935 | +0.7924 | +0.7969 |

Paired bootstrap over the 311 test examples (10k resamples), at the
validation-selected points:

- **ssm − clt_full: −0.0090 [−0.0136, −0.0045]** (significant; the §4.2 CI)
- ssm − per_layer: +0.0518 [+0.0485, +0.0552]
- clt_full − per_layer: +0.0608 [+0.0563, +0.0655]

Full per-(arm, λ) train/val/test table in `tables/lambda_grid.csv`.

## Notes

- **Protocol variant, not a re-estimate:** fits use 1057 examples (1244 − 187 val),
  so absolute R² sits slightly below the R2 runs (which fit on the full 1244); the
  paired contrasts are internal to this protocol and that is what the finding-4
  check requires. No refit-on-full after selection (both arms treated identically).
- R2's test-selected CLT λ was 10; validation picks 3 (val nearly ties 3 vs 10) —
  the direction of the difference (−0.0142 → −0.0090) is exactly the mild
  test-selection optimism the review predicted, and it does not move any conclusion:
  the three-axis §4.2 story (R2/R3: CLT edge is small, data-bought, and
  budget-dominated by the SSM at ≤52M params) is unchanged.
- AR best-(k, rank) selection on test (the other half of finding 4) remains a
  methods sentence: it is symmetric real-vs-null, and the real−null contrast is the
  claim; no CI is quoted at a test-selected AR point.
