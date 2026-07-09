# R13 — the pressure battery at a common depth set (referee-pass correction C2)

**Date:** 2026-07-09. **Script:** `scripts/r13_common_depth_pressure.py`.
**Run:** `runs/2026-07-09-r13-common-depth-pressure`. **Trigger:** 6th-pass referee read
of the D3 draft.

`fit_ar_pooled` scores only the depths where all `k` lags exist (`ℓ ≥ k`). At `L=24`
that is 23/22/21/19/14/9/**4** target depths for `k = 1/2/3/5/10/15/20` — the count was
in `ar_r2.csv` all along (`n_depths`). So the Phase-1 "best over k" selection is partly
a selection of the **easiest depth subset**, and any comparison between two fits that
chose different `k` is a comparison across different evaluation sets.

> **Verdict.** Three published readings were confounded. Fixing them **strengthens** the
> reconstruction-shallow headline, **narrows** T4's architectural claim to Pythia, and
> **widens** the factual-vs-generic gap. No gate outcome changes.

## 1 · The lag effect reverses under depth control

`tables/ar_lag_vs_depth_subset.csv` — same fits, two evaluation sets:

| k | own depths (ℓ≥k) | R² own | common depths ℓ∈[20,23] | R² common |
|---|---|---|---|---|
| 1 | 23 | 0.5206 | 4 | **0.6839** |
| 2 | 22 | 0.4915 | 4 | 0.6545 |
| 3 | 21 | 0.4736 | 4 | 0.6472 |
| 5 | 19 | 0.4666 | 4 | 0.6481 |
| 10 | 14 | 0.5018 | 4 | 0.6516 |
| 15 | 9 | 0.5513 | 4 | 0.6508 |
| 20 | 4 | **0.6497** | 4 | 0.6497 |

Own-depths gain k1→k20 **+0.129**; depth-controlled **−0.034**. **AR(1) is the best fit
at fixed depths and every longer lag order is worse.** The published "R² rises to 0.65 at
k=20" is a depth-subset effect, not a lag effect. This agrees with Step-2's per-depth
result (lags≥2 add a median +0.018) and makes the "Granger reach 20 vs reconstruction
k\*≈1" headline stronger, not weaker.

## 2 · The T1/T5 battery at a common k=1 (all feasible depths)

`tables/t1_common_depth.csv`, `tables/t5_common_depth.csv`:

| store | depths | R² (m=0) | R² (m=50, T1) | T5 taut. | T5 genuine |
|---|---|---|---|---|---|
| 410m factual, trained | 23 | 0.521 | **0.436** | 0.272 | 0.248 |
| 410m factual, untrained | 23 | 0.543 | **0.503** | 0.116 | 0.427 |
| 410m generic, trained | 23 | 0.882 | **0.069** | 0.879 | 0.003 |
| 410m generic, untrained | 23 | 0.063 | **0.061** | 0.007 | 0.056 |
| GPT-2-small factual, trained | 11 | 0.626 | **0.439** | 0.490 | 0.136 |
| GPT-2-small factual, untrained | 11 | 0.246 | **0.233** | 0.046 | 0.200 |

Scrambled nulls ≤ 7e-3 everywhere.

### 2a · T4's GPT-2 replication was an artefact → "architectural" is Pythia-only

The published comparison was trained **0.439** (`k=1`, 11 depths) vs untrained **0.420**
(`k=10`, **2 depths**) → "learned share ~5%". At a common `k=1`:

- **GPT-2-small: 0.439 trained vs 0.233 untrained → 47% learned.**
- **Pythia-410m: 0.436 trained vs 0.503 untrained → untrained *exceeds* trained.**

So the architectural deflation stands on the primary model and **does not replicate**
cross-family. Consequently the AR-vs-reconstruction **dissociation** (R10) is a
one-family observation: on GPT-2 the AR learned share (47%) and the reconstruction
learned share (56%) agree. R10's reconstruction split itself (untrained carries 53% at
410m, 44% at GPT-2) is untouched — it never used the AR sweep.

### 2b · The headline 0.65 is a four-depth number

T5 decomposes the `k=20` fit, so `0.65 = 0.52 taut + 0.13 genuine` describes ℓ∈[20,23],
where the stream drifts most. Whole-stack (`k=1`): **`0.52 = 0.27 taut + 0.25 genuine`**.
The tautology is the majority share either way; the genuine share is ~2× larger than the
headline suggests.

### 2c · The factual/generic T1 overlay compared different depth sets

fig14 plotted factual (which selects `k=1`, 23 depths) against generic (which selects
`k=20`, 4 depths). At a common `k=1`, generic persistence-stripped structure is **0.069**,
not 0.29 — an **order of magnitude below** factual's 0.436, not "comparable". The raw
generic level of 0.882 is 99.7% drift (T5 genuine = 0.003). `make_paper_figures.py` now
builds fig14 from `t1_common_depth.csv`.

## Scope

Analysis-only; no new activations, no refit of any SSM. Same stores, same 80/20 split
(seed 0), same ranks/PCs as the Phase-1 sweep. The pre-registered headline numbers are
reported unchanged in the paper alongside the depth set they were scored on.
