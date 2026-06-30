# T4 — untrained-model control (Phase 4 H1-validity). RESULT: DEFLATIONARY.

**Date:** 2026-06-30. **Model:** pythia-410m-deduped, **random init** (TL default init,
seed 0; `load_untrained_model`, `init_weights` forced on). Same prompts/windows as the
trained stores (only the weights differ). Stores: `data/pythia-410m-deduped-untrained{,-general}/`
via `scripts/phase4_t4_untrained.py`. Analysis = the *identical* pressure-test pipeline
(`scripts/phase4_pressure_tests.py --h5`). Runs: `runs/2026-06-30-ph4-t4-pressure-untrained-410md{,-general}/`.

> **What T4 asks.** T1–T3 hardened the AR-on-Δ headline against the ±v_ℓ *tautology*.
> T4 asks the orthogonal *plumbing* question: is the (tautology-free) depth structure a
> product of **learning**, or of the transformer **architecture** (residual + per-layer
> nonlinear maps + LayerNorm)? Pre-registered logic: if trained persistence-removed R²
> ≫ untrained, it is learned. **The data say it is NOT — the structure is architectural.**

## Result

Persistence-stratified AR R² (T1), random-init vs trained:

| m removed | state-var removed | **untrained** | **trained** | best-k (untr / tr) |
|---|---|---|---|---|
| **factual** | | | | |
| 0 | 0.00 | 0.543 | 0.650 | 1 / 20 |
| 1 | 0.20 / 0.32 | 0.543 | 0.637 | 1 / 20 |
| 5 | 0.56 / 0.61 | 0.533 | 0.548 | 1 / 20 |
| 20 | 0.73 / 0.79 | 0.519 | 0.454 | 1 / 1 |
| 50 | 0.81 / 0.87 | **0.503** | **0.436** | 1 / 1 |
| **generic** | | | | |
| 0 | 0.00 | 0.210 | 0.935 | 20 / 1 |
| 50 | 0.50 / 0.87 | **0.216** | **0.290** | 20 / 20 |

Both untrained stores: scramble null ≈0, T2 surrogate ≈0 (real−surrogate = 0.543 / 0.210),
**T3 lag≥2 Granger 89/89** (vs 0/89 null) — i.e. the untrained structure passes *every*
tautology-free control, exactly like the trained model.

## Reading (honest, deflationary)

- **The tautology-free AR-on-Δ predictability is predominantly ARCHITECTURAL.** A
  random-init transformer already shows persistence-removed R² **0.50 (factual) / 0.22
  (generic)** — comparable to the trained model (**0.44 / 0.29**), and in the factual
  case *higher*. In neither store is trained ≫ untrained, so the structure H1 measures is
  a property of the architecture (fixed per-layer nonlinear maps + residual + LN), **not**
  of the learned weights. This is the pre-registered "plumbing" outcome.
- **What training *does* change** is the *persistent/drift* component, not the
  tautology-free one: it lifts raw R² (factual 0.54→0.65; generic 0.21→**0.935**) and adds
  a long-range k=20 term — but both vanish under persistence removal (trained drops to k=1
  by m=20). So training amplifies the running-sum drift structure, the part T1–T3/T5
  already flagged as the ±v_ℓ tautology, and adds little tautology-free depth structure.
- **Coheres with E4.** E4 found H1 generalises to generic text; T4 finds it is present at
  random init. Together: **H1's depth structure is a generic property of the transformer
  architecture, independent of task (E4) and of training (T4).** H1's *literal* claim
  (predictable above the scramble null) still holds — but it should be framed as
  architectural depth structure, not learned depth-computation.

## What is and isn't touched

- **Reframes:** H1's interpretation (architectural, not learned). The "honest" headline
  number (T1 0.44 factual / 0.29 generic) is *not* a learned-computation signature.
- **Untouched by T4:** the reconstruction win (SSM > per-layer baseline; ties CLT at 16%
  params), the contractive A_ℓ operator, k*≈1, the causal null — T4 tests AR-on-Δ
  predictability, not SSM reconstruction. **Open follow-up:** is the *reconstruction* win
  also architectural? (Fit the SSM-vs-per-layer Pareto on an untrained model's
  activations.) That is the experiment that decides whether the paper's *headline* claim
  is learned or architectural — recommended before write-up.
- **Caveat:** single random init. The qualitative conclusion (untrained ~0.5/0.2, not
  near-zero; never trained≫untrained) is robust to the draw, but multiple inits would
  firm up the exact gap.
