# Phase 4 — T1–T3: H1-validity / tautology pressure-tests (factual recall)

**Date:** 2026-06-30. **Model:** pythia-410m-deduped (L=24, D=1024). **Run:**
`runs/2026-06-30-ph4-pressure-410md` (`scripts/phase4_pressure_tests.py`,
`diagnostics/autoregress.py`). **Store:** the Phase-1 factual residual stream
`data/pythia-410m-deduped/residual_stream.h5` (1555 prompts, query position).

T1–T3 stress the paper's **most contestable headline** — the pooled AR-on-Δ R²
(**0.65** factual) — against the standing ±v_ℓ running-sum tautology (Δ_ℓ ⊃ −v_ℓ ⊂
Δ_{ℓ-1}; `cca-vdelta` memory). The scramble null (Null 1) destroys the genuine
structure **and** the tautology together (after a per-depth permutation the −v_ℓ in
Δ_ℓ and the +v_ℓ in Δ_{ℓ-1} come from different examples), so it proves only "the
predictability is example-specific" — and trivial persistence is *also*
example-specific. T1–T3 are the three data-only tests that actually separate "real
depth computation" from "the residual stream persists".

> **Verdict — H1 SURVIVES ALL THREE DECISIVELY. The headline AR-on-Δ predictability
> is NOT the persistence tautology.** (T1) Projecting out the dominant persistent
> state directions barely dents it — removing the **single top direction (32 % of all
> state variance, the giant rogue dim)** drops R² only 0.650→0.637, and removing the
> **top-50 (87 % of state variance)** still leaves **R² 0.436 ≫ scramble 0.0001**; the
> predictability lives in the broad subspace, not the few persistent directions. (T2)
> A persistence-matched **independent-step surrogate** floors at **0.0000**
> (`real − surrogate = 0.6496`): no amount of state persistence reproduces AR-on-Δ —
> it requires genuine step-to-step structure, exactly as the cancellation argument
> predicts. (T3) The tautology-free **lag≥2 Granger is 89/89 Bonferroni-significant**
> (vs 0/89 null). So the honest, tautology-robust H1 number is large, not a sliver.

## T1 — persistence-stratified AR (project out top-m state directions)

Project every Δ_ℓ onto the orthogonal complement of the top-m principal directions of
the **states** v_ℓ, then re-run the full AR(k)×rank sweep (best held-out R²).

| m removed | state variance removed | real best R² (k, rank) | scramble null |
|---|---|---|---|
| 0  | 0.000 | **0.6497** (k=20, r128) | 0.0001 |
| 1  | **0.320** | **0.6369** (k=20, r128) | 0.0001 |
| 5  | 0.605 | 0.5483 (k=20, r128) | 0.0001 |
| 20 | 0.790 | 0.4535 (k=1, r128) | 0.0001 |
| 50 | **0.871** | **0.4361** (k=1, r128) | 0.0001 |

- **The top direction alone is 32 % of all state variance** (top-20 = 79 %) — the
  persistent "rogue" dimension that the ±v_ℓ tautology is built on. Removing it costs
  **0.013 R²**. If the AR score were the tautology, this would have gutted it.
- **Decay is gradual and saturates ~0.44**, not a collapse: even after stripping 87 %
  of state variance the held-out predictability is 0.44, ~4000× the scramble floor.
  (The best k drops 20→1 as the dominant directions go, consistent with k\*≈1: the
  surviving structure is first-order, the long-lag tail lived partly in the persistent
  subspace.)
- **Reading:** the AR-on-Δ predictability is genuine depth structure spread across the
  residual subspace, **not** the mechanical persistence of the dominant directions.

## T2 — persistence-matched surrogate null (independent matched steps)

Independent per-layer Gaussian steps matched to each Δ_ℓ's mean + (rank-256)
covariance — a random walk whose increments carry **no** cross-layer dependence but
whose every per-layer step marginal (incl. the persistent directions) is preserved.

| | best held-out AR R² |
|---|---|
| real | **0.6497** |
| persistence-matched surrogate (T2) | **0.0000** |
| scramble null (Null 1) | 0.0001 |

`real − surrogate = 0.6496`. The surrogate collapses to 0 as the cancellation argument
predicts (with independent steps the Var(v_ℓ) contributions to Cov(Δ_ℓ, Δ_{ℓ-1}) net to
zero), so **the honest floor for the headline is essentially the scramble floor**, and
the full 0.65 sits above it. This *parametrically* corroborates Null 1 and closes the
"is it just persistence?" question in the favourable direction: AR-on-Δ predictability
**cannot** be manufactured by state persistence alone.

## T3 — lag≥2 Granger (intrinsically tautology-free)

The shared ±v_ℓ term couples **only** lag 1 (Δ_{ℓ-2}=v_{ℓ-1}−v_{ℓ-2} shares no v_ℓ/v_{ℓ+1}
term with Δ_ℓ), so the lag≥2 significant count is tautology-free by construction.

| variant | lag 1 sig | lag≥2 sig |
|---|---|---|
| real | 23/23 | **89/89** |
| scramble null | — | 0/89 |

**Every** lag≥2 cell is Bonferroni-significant (real) and **none** under the null. The
depth dependence beyond the mechanical lag-1 overlap is total and clean.

## Honest framing for the paper

- The *level* of the AR R² is still partly tautological at lag 1 (T1 shows ~0.21 R² is
  attributable to the persistent subspace: 0.650→0.436 as 87 % of state variance is
  removed). The defensible claims are the **tautology-robust** ones: **(i)** R² 0.44
  survives removing 87 % of state variance; **(ii)** the surrogate floor is 0; **(iii)**
  lag≥2 Granger is 89/89. Report these next to the raw 0.65, not the raw number alone.
- This **upgrades** the H1 evidence: the scramble null is now backed by a parametric
  surrogate (T2) and a subspace-stratified control (T1) and a lag-split Granger (T3),
  so a referee's "you only rediscovered the running sum" critique is pre-empted with
  three independent controls.

## Reproduce
```
scripts/phase4_pressure_tests.py --config configs/pythia-410m-deduped.yaml \
  --desc ph4-pressure-410md --seed 0
```
Generic-text replication (E4/E7 store) is the companion run `ph4-pressure-410md-general`.
