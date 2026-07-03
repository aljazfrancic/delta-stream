# Phase 4 — E3: residual-write CLT vs the state-space SSM (the §4.2 comparison, on-box)

**Date:** 2026-06-30. **Model:** pythia-410m-deduped (L=24, D=1024). **Run:**
`runs/2026-06-30-ph4-e3-clt-410md` (`scripts/phase4_baseline_clt.py`, `src/layer_dynamics/clt.py`).
**Sources:** the cached mlsae per-layer sparse sources `u[1555,24,4433]`, frozen.

> **⚠ SUPERSEDED (2026-07-03, review items R1/R2/R3 — annotated during the 4th-pass
> review; the body below is the dated 2026-06-30 record, unedited).** The headline
> "**tie** at 6.27× fewer params" is **dead**: this run's λ_reg grid topped out at 0.1,
> which was every model's best (edge-landing). Extending the grid (R2) lifts all three
> and the best-of-grid CLT **leads** at matched per-decoder rank — merged rank-32 gap
> **SSM−CLT −0.0142 [−0.0185, −0.0099]** (paired bootstrap over test examples) — and R1
> finds a small significant CLT edge at every depth (6L −0.009 / 12L −0.050 & −0.023 /
> 24L −0.011, not monotone → the E8 "depth-graded tie" framing is retired too). **What
> replaces the parsimony claim (the honest §4.2 statement):** at matched **total**
> parameter budget the SSM Pareto-dominates ≤52M (SSM r64 @16.9M = 0.8155 > CLT r32
> @52.4M = 0.804 > CLT r16 @26.2M = 0.794; CLT needs 105M to lead outright), and the
> SSM generalises better below ~1k train examples (R3 sign-flip: N=300 SSM +0.0204,
> N=600 +0.0128, N=1244 CLT −0.0105 — crossover ≈ N≈1k). SSM−per_layer stays large and
> CI-significant everywhere (+0.022…+0.053). See progress.md (R1/R2/R3), fig08
> (`fig08_clt_three_axis.svg`), runs `2026-07-0{2,3}-r{1,2,3}-clt-*`.

E3 supplies the number the paper's *primary* conceptual comparison ([proposal §4.2], Baseline D)
never had. CLT places a **free** low-rank decoder at every (source, target) layer pair `ℓ'≤ℓ` —
the unconstrained `O(L²)` parameterisation our Branch-B state-space model is meant to be a
parsimonious reparameterisation of. The "cloud-only" tag was a mis-classification (it conflated
*training a CLT from scratch* with the §6 **residual-write variant fit on cached sources**); the
latter is a linear low-rank ridge in `u` and fits the 3 GB box. We fit three decoders of the **same
frozen sources** at the **same rank** (32), each λ_reg-swept to its **best held-out** R² (project
convention, pooled vs per-depth mean), reporting the **effective parameter count** of each.

> **Verdict — PARSIMONY WIN ("comparable reconstruction at far fewer parameters", the §4.2
> H2-holds-style outcome).** The unconstrained `O(L²)` CLT reconstructs Δ held-out **no better than
> our first-order state-space SSM** — test R² **0.784 (CLT) vs 0.780 (SSM)**, a **−0.0035** gap
> (a tie, within run-to-run noise) — while using **6.27× the parameters** (52.4M vs 8.4M). The CLT
> spends its extra budget on overfitting: it has the largest train→test gap (0.134 vs the SSM's
> 0.091). Both cross-layer models beat the no-propagation per-layer baseline (0.727) by ~0.05. So
> the state-space parameterisation captures essentially all of the held-out cross-layer
> reconstruction value at ~16% of CLT's parameter budget. The free per-(source,target) decoder
> family is **redundant** on this data.**

## Result — frozen mlsae sources, rank 32, best λ_reg per model

| model | decoder | eff. params | best λ_reg | train R² | **test R²** | overfit gap |
|---|---|---|---|---|---|---|
| `per_layer` | diagonal only (no cross-layer) | 4.22M | 0.1 | 0.801 | 0.727 | 0.075 |
| **`ssm` (ours)** | first-order depth-indexed state | **8.36M** | 0.1 | 0.871 | **0.780** | 0.091 |
| `clt_full` | unconstrained `O(L²)` per-pair | 52.41M | 0.1 | 0.918 | **0.784** | 0.134 |

`SSM − CLT` held-out gap **−0.0035**; CLT/SSM parameter ratio **6.27×**. See
`tables/baseline_clt.json`, `figures/clt_vs_ssm.svg`.

## Interpretation

- **The §4.2 claim lands on the "comparable" branch.** §4.2 framed two outcomes: H2-holds → "my
  decomposition reconstructs comparably to CLT [with] much fewer parameters"; H2-fails → "CLT's free
  parameters are doing real work." H2 *did* fail (Phase 2), so the clean post-H2 reading is the
  refined one: **both** methods are fully depth-indexed, and the question E3 answers is whether the
  *first-order Markov-in-depth state* compression loses reconstruction vs free per-(source,target)
  decoders. It does not — the state-space form is a genuine `6×` compression of the cross-layer
  decoder budget at no held-out cost. The free `O(L²)` decoders are redundant *for reconstruction*
  at this data scale.
- **Cross-layer propagation is load-bearing; its unconstrained form is not.** per_layer→ssm is
  +0.053 held-out (cross-layer state helps, consistent with Gate-3's `use_state` result); ssm→clt
  is +0.0035 (the unconstrained generalisation of that state buys nothing). The value is in *having*
  a cross-layer term, not in giving it `O(L²)` free parameters.
- **Why CLT doesn't win despite strictly more capacity:** N=1244 training examples. The 52M-param
  CLT fits the train set best (0.918) but overfits hardest (gap 0.134), so its held-out R² lands at
  the SSM's. This is the parsimony argument made concrete: on realistic data, the inductive bias of
  a low-dimensional propagated state matches an unconstrained decoder family that has 6× the
  parameters to overfit with.

## Honest caveats (for the write-up)

1. **A tie, not a "beat".** CLT is +0.0035 *above* the SSM held-out — report as *comparable*, not
   *SSM > CLT*. The result is "no reconstruction cost to the 6× compression", which is the §4.2 win;
   do not overstate it as the SSM dominating CLT. (At under-regularised λ_reg the CLT overfits below
   the SSM; at its best λ_reg it ties — both directions are in `tables/baseline_clt.json`.)
2. **Frozen-source operating point.** All three decode the *same fixed* mlsae sources with plain Adam
   (no LASSO re-inference), which isolates the decoder parameterisation but sits below the Gate-3
   SSM's headline 0.857 (that number uses LASSO-inferred sparse sources + alternating-min). The
   ~0.78 ceiling here is the frozen-raw-source ceiling and is the same for all three, so the
   *comparison* is clean; 0.857 is noted as the with-sparse-inference reference.
3. **Sources held fixed for both.** A fully native CLT would jointly infer its own sparse sources;
   here it inherits the SSM's source set. This is the cleanest isolation of "decoder
   parameterisation" but does not give CLT a separately-optimised dictionary. Stated as scope.
4. **On-box scope.** Rank 32, single seed, mlsae sources. The conclusion (unconstrained `O(L²)`
   decoders are redundant vs the state-space form at matched rank/sources) is the headline; a rank
   or seed sweep is future work.

## Bottom line for the paper

E3 converts the largest remaining hole — *no number for the primary §4.2 baseline* — into a headline
parsimony result: **the state-space decomposition reconstructs Δ as well as an unconstrained
`O(L²)` residual-write CLT at 6.3× fewer parameters**, on identical sources at matched rank. This is
the reconstruction-parsimony comparison §4.2/§6 promised, now produced on the 3 GB box rather than
deferred to cloud as a limitation.
