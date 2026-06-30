# Phase 4 — T1–T3 pressure-tests: generic-text replication

**Date:** 2026-06-30. **Model:** pythia-410m-deduped. **Run:**
`runs/2026-06-30-ph4-pressure-410md-general` (`scripts/phase4_pressure_tests.py`).
**Store:** the E4/E7 length-matched generic **Pile** residuals
`data/pythia-410m-deduped-general/residual_stream.h5` (1555 windows, query position).
**Companion:** factual run `runs/2026-06-30-ph4-pressure-410md` (memo there).

Generic text is where the ±v_ℓ tautology worry was **strongest**: E4 reported AR-on-Δ
R² **0.935** (vs factual 0.65) and explicitly cautioned not to over-read the higher
level, because more-stereotyped generic last tokens enlarge the mechanical −v_ℓ⊂Δ
overlap. T1–T3 quantify exactly how much of the 0.935 is that tautology.

> **Verdict — the worry is CONFIRMED for the *level* and the H1 *existence* claim is
> VINDICATED on both stores.** (T1) The generic store is **96.1 %** dominated by its
> single top state direction (vs 32 % factual), so removing it drops R² 0.935→0.671;
> stripping the top-50 (**98.5 %** of state variance) lands at **0.290** — a steeper
> decay than factual (→0.44), so *more* of the raw generic number was persistence
> padding, exactly as E4 predicted. **But the surviving signal is real on both:** 0.29
> (generic) / 0.44 (factual) ≫ scramble ≈0. (T2) The persistence-matched surrogate
> floors at **−0.0001** (`real − surrogate = 0.935`). (T3) The tautology-free **lag≥2
> Granger is 89/89** significant vs 0/89 null — *identical* to factual. T2 and T3 are
> indistinguishable across the two stores; only T1's *level* differs.

## T1 — persistence-stratified AR (generic vs factual)

| m removed | state var removed (generic) | **generic R²** | factual R² |
|---|---|---|---|
| 0  | 0.000 | **0.9346** | 0.6497 |
| 1  | **0.961** | 0.6711 | 0.6369 |
| 5  | 0.977 | 0.5780 | 0.5483 |
| 20 | 0.983 | 0.3716 | 0.4535 |
| 50 | **0.985** | **0.2900** | 0.4361 |

- **The single top direction is 96.1 % of all state variance on generic text** (top-20
  = 98.3 %), vs 32 % / 79 % on factual. Stereotyped Pile last tokens (common words,
  punctuation, code syntax) have a near-one-dimensional persistent trajectory — the
  textbook regime that inflates the ±v_ℓ overlap.
- **Removing it drops R² 0.935→0.671** — so ~0.26 of the generic headline was the
  persistence tautology, and the part of 0.935 that *exceeded* the factual 0.65 is
  almost entirely it. After the top direction, generic and factual are close
  (0.671 vs 0.637 at m=1).
- **Genuine structure survives:** at m=50 (98.5 % of variance gone) generic R² = 0.290,
  thousands of times the scramble floor (≈0). The decay is steeper than factual and
  saturates lower, the honest signature of "more tautology in the level, same real
  phenomenon underneath".

## T2 — persistence-matched surrogate null

| | best held-out AR R² |
|---|---|
| real | **0.9346** |
| persistence-matched surrogate | **−0.0001** |
| scramble null | −0.0006 |

`real − surrogate = 0.9348`. Identical behaviour to factual: independent matched steps
reproduce **none** of the AR-on-Δ predictability, so the headline cannot be a state-
persistence artefact (the level inflation is a *lag-1 linear* effect that T1 isolates,
not something the surrogate floor captures — both nulls correctly sit at 0).

## T3 — lag≥2 Granger (tautology-free)

| variant | lag 1 | lag≥2 |
|---|---|---|
| real | 23/23 | **89/89** |
| scramble null | — | 0/89 |

Identical to factual: every lag≥2 cell significant, none under the null. The depth
dependence beyond the mechanical lag-1 overlap is total on generic text too.

## Combined read (with the factual companion) — for the paper

1. **H1 (existence of above-null cross-layer Δ structure) is tautology-robust on both
   factual and generic text:** survives removing 87 %/98.5 % of state variance (T1),
   sits entirely above a parametric persistence-matched surrogate (T2), and is 89/89 in
   the tautology-free lag≥2 Granger (T3).
2. **The E4 caveat is now quantified, not just flagged:** the generic R² *level* (0.935)
   is inflated by its 96 %-dominant persistent direction; ~0.26 of it is the ±v_ℓ
   tautology, and stripping the dominant subspace brings generic (0.67@m=1, 0.29@m=50)
   into the same regime as factual. So report the **binary H1 conclusion** as general,
   and the **R² level** with the T1 caveat — never the raw 0.935 as "more computation".
3. **The honest headline numbers** to quote alongside the raw R²: factual survives at
   **0.44** / generic at **0.29** after removing ~all state variance; surrogate floors
   **≈0**; lag≥2 Granger **89/89** both. These pre-empt the "you only rediscovered the
   running sum" critique with three independent controls.

## Reproduce
```
scripts/phase4_pressure_tests.py --config configs/pythia-410m-deduped.yaml \
  --desc ph4-pressure-410md-general --seed 0 \
  --h5 data/pythia-410m-deduped-general/residual_stream.h5
```
