# Gate 2 decision memo — Phase 2 Characterization

**Model:** Pythia-410m-deduped (L=24, D=1024). **Dataset:** 1,555 factual-recall
prompts, 3 relation classes (CounterFact, top-1==object, prob>0.1), residual stream
at the final prompt token → `residual_stream[1555, 25, 1024]`. **Date:** 2026-06-22.
**Seed:** 0. **Split:** 1,244 train / 311 test (shared with Phase 1). Runs:
`runs/2026-06-22-ph2-step1-410md/` (stationarity), `runs/2026-06-22-ph2-step2-410md/`
(effective depth).

> Headline: **H2 (approximate stationarity across depth) is rejected.** The AR-on-Δ
> kernel that predicts Δ_ℓ from its recent past is **strongly depth-dependent** — a
> kernel fit on one depth band predicts another band's updates *worse than that band's
> own mean* — and the dependence is **pervasive** (no clean phase boundary), not a
> gentle drift. The reconstruction-effective depth is **k\* ≈ 1** at every depth. This
> is the pre-registered "H2 fails" outcome (proposal §4.2): H1 still holds (Gate 1),
> the dynamics are genuinely non-stationary, and CLT-style unconstrained per-depth
> parameterisation is *validated*. **Architecture: Phase 3 Branch B — a first-order,
> fully depth-indexed state-space model** (see `phase3_architecture.md`).

---

## The five Gate-2 questions

**1. Does stationarity hold? — NO (decisively, rank-robustly).** (Step 1) One
band-shared low-rank AR(k=5) kernel was fit on each of the early (ℓ5–7), middle
(ℓ8–15), and late (ℓ16–23) depth bands against a **shared input PCA basis**, then
compared four ways:
- **Cross-band prediction (the functional test):** within-band held-out R² is
  positive (early 0.74 / middle 0.58 / late 0.67) but **every** off-diagonal is
  **negative** (early→middle −1.00, late→middle −1.89, …). A kernel from one band is
  worse than the per-depth mean on another band. This holds at **every** truncation
  rank 32→256 (worst off-diagonal −0.32 even at r=32), so it is not a high-rank tail
  artefact (`tables/rank_robustness.csv`).
- **Frobenius distance:** between-band kernel distance ≈ **1.39–1.42** ≈ √2 (the
  orthogonal-of-equal-norm ceiling).
- **Principal angles:** mean output-subspace angle **57–62°** per pair.
- **Bootstrap (n=200/band):** the between-band distance sits **80–104 SD** above the
  within-band sampling spread; all three pairs are cleanly separated.
- **No clean phase boundary:** the three pairwise distances are ≈ equal (ratio 1.03),
  so this is *not* a segmented (Branch C) structure — the depth-dependence is
  pervasive (adjacent bands are about as different as distant bands).

*k was capped* at the early-band feasibility limit ⌊L/3⌋−3 = 5 (the Step-5 Granger
profile is significant out to the maximum lag tested, 20), so Step 1 tests
**short-horizon** stationarity; long-horizon stationarity is deferred to the Phase 3
Step 4 MA recheck on the fitted generative model.

**2. What is the effective depth k\*? — k\* ≈ 1 at every depth.** (Step 2) Per-depth
single-target AR(k) fits over k ∈ {1,2,3,5,10,15,20} (capped at k ≤ ℓ), at the Step-1
working rank r=256: per-depth k\* has **median 1, max 2** (only ℓ=17), and all three
bands give k\*=1. A single preceding update Δ_{ℓ-1} captures essentially all held-out
predictable variance (per-depth best R² 0.83–0.94, dipping to 0.64 at ℓ17); lags ≥ 2
add a **median +0.018 R² (max +0.05)**. The scrambled null collapses the whole curve
(median best R² **0.000**), confirming real example-specific structure.
- **Reconstruction-vs-significance gap (a headline number):** reconstruction k\* ≈ 1
  versus the Step-5 **Granger reach of lag 20**. Long-range depth structure is
  *statistically detectable but reconstruction-negligible*.
- **Honest caveat (the ±v_ℓ tautology):** consecutive updates share ∓v_ℓ
  (Δ_ℓ = v_{ℓ+1}−v_ℓ, Δ_{ℓ-1} = v_ℓ−v_{ℓ-1}), so the lag-1-dominated reconstruction is
  partly mechanical. k\*≈1 is therefore a property of the AR-on-Δ parameterisation,
  **not** proof the transformer's genuine computational dependence is one layer deep.
  Operationally it means a **first-order** state-space model (Markov in the depth
  latent) is the right order for Phase 3.

**3. Is the AR structure token-position-invariant, or query-specific? — NOT YET
TESTED.** (Step 3) Requires re-collecting residual-stream trajectories at ~3 extra
token positions (~4 GB) and refitting the AR(k\*) kernel per position. Outstanding.
This question gates the *scope* of the claim (depth-intrinsic vs context-aggregated
factual recall at the query token) — it **cannot flip the architecture choice**,
which the plan's decision matrix derives from Step 1 alone.

**4. Is the structure circuit-general or circuit-specific? — BLOCKED on dataset
scale.** (Step 4) The per-class cross-relation test needs ≥ 1,000 examples in each of
≥ 3 classes; the current set is 1,555 examples across 3 classes (organization 711,
geography 511, language 333), below the bar (Pythia-410m confidently knows only ~2,000
CounterFact facts). Unblocking requires the Pararel/T-REx union (stubbed in
`data/sources.py`) or a larger model. Outstanding; refines generality, not
architecture.

**5. Does any of this replicate cross-model? — DEFERRED.** (Step 5) Gemma-2-2B and
Llama-3.1-8B do not fit the 3 GB dev GPU; cross-model replication is deferred to a
larger/cloud GPU (a standing constraint recorded since Phase 1). Outstanding; refines
generality, not architecture.

---

## Architecture decision

**☑ Branch B — depth-indexed / state-space model.** The plan's decision matrix keys the
Phase-3 architecture off Step 1: stationarity holds → Branch A (block-Toeplitz);
fails smoothly → Branch B (state-space); clean phase transition → Branch C (segmented).
Step 1 rejects stationarity and finds **no clean phase boundary**, so neither A nor C
applies → **Branch B**.

Two refinements beyond the bare branch label, both load-bearing for Phase 3 and both
justified by Steps 1–2 (spelled out in `phase3_architecture.md`):
- **Fully depth-indexed, not depth-smoothed.** The plan's default Branch B stores
  operator factors at L_c ≪ L control depths and *interpolates* (assuming smooth
  drift). Step 1 found the depth-dependence is **pervasive** — adjacent bands are about
  as different as distant ones — so the smoothness the interpolation assumes is not
  supported at band scale. Default to **free per-layer operators**; the depth-smoothed
  variant is an ablation, to be reconsidered only if a finer per-depth kernel-similarity
  analysis later reveals exploitable smoothness.
- **First order.** k\*≈1 ⇒ a first-order Markov-in-depth SSM (z_{ℓ+1}=A_ℓ z_ℓ+B_ℓ u_ℓ)
  is the right order; no long depth memory needs to be engineered in.

**Parsimony claim is off.** Because H2 failed, we are explicitly in proposal §4.2's
"H2 fails → CLT's free parameters are load-bearing → CLT validated" branch. We do
**not** claim a parameter win over CLT; the fully-depth-indexed Branch B is *larger*
than a stationary kernel by design. Report Branch B's parameter count honestly against
CLT in Phase 4 (do not equate a reconstruction win with a parsimony win).

---

## Status, caveats, and what remains

This is an **architecture-locking** Gate 2: Q1–Q2 are answered definitively and fix the
Phase 3 branch; **Q3 (position), Q4 (cross-class), Q5 (cross-model) are outstanding**
(feasible-later / blocked-on-data / deferred-to-cloud, respectively). Per the plan,
those three refine the *scope and generality* of the characterisation and feed the
paper's claims, but none can overturn the Step-1-driven architecture choice. They
should be completed (or their absence stated as a limitation) before the Phase 4
write-up.

Carried-forward caveats: (1) k capped → short-horizon stationarity only; the MA-form
recheck is Phase 3 Step 4. (2) AR-on-Δ stationarity is a proxy, not a logical
equivalent of H2's MA/generative stationarity. (3) the ±v_ℓ shared term biases the
kernel comparison *toward* stationarity, so rejecting it is the conservative direction;
it also dominates the k\*≈1 read. (4) single token position (final prompt token) —
Step 3 tests this. (5) dataset below the ≥1000/class target — Step 4.

## Reproduce

```
scripts/step1_stationarity.py --config configs/pythia-410m-deduped.yaml --desc ph2-step1-410md \
    --step3-csv   runs/2026-06-21-ph1-step3-410md/tables/ridge_r2.csv \
    --granger-csv runs/2026-06-22-ph1-step5-410md/tables/granger_real.csv
scripts/step2_kstar.py        --config configs/pythia-410m-deduped.yaml --desc ph2-step2-410md \
    --step1-stats runs/2026-06-22-ph2-step1-410md/tables/step1_stats.json \
    --granger-csv runs/2026-06-22-ph1-step5-410md/tables/granger_real.csv
```

Figures in `figures/`; all numerics in the two run directories' `tables/`.
