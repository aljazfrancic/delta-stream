# Gate 3 decision memo — Phase 3 Decomposition

**Model:** Pythia-70m-deduped (L=6, D=512) — the **debug lane**, not the scientific
primary. **Sources:** public per-layer residual SAEs `EleutherAI/sae-pythia-70m-deduped-32k`
(M=32768, TopK k=16; `u_0` zero-init — `v_0` has no embedding SAE). **Dataset:** top-1000
confidence-ranked CounterFact prompts, residual stream at the final prompt token →
`residual_stream[1000, 7, 512]` → `Δ[1000, 6, 512]`. **Split:** 800 train / 200 test,
seed 0. **Date:** 2026-06-25. Runs: `runs/2026-06-24-ph3-gate3-70md/` (Pareto),
`runs/2026-06-24-ph3-step4-70md/` (stability/sanity), `runs/2026-06-24-ph3-decomp-70md/`
(optimiser debug).

> Headline: **On the 70m debug lane the Branch B SSM clears the Gate-3 go criterion.**
> Held-out reconstruction sits **on or above** the matched per-layer sparse-dictionary
> baseline at every sparsity level (R² gap **+0.008 → +0.018** across the L0 range), and
> the recovered features are **seed-stable (match-rate@0.7 = 0.99 ≫ 0.7)** and
> dataset-stable (top-100 Jaccard 0.89). **But the margin is small** — the cross-layer
> state contributes `|C|≈14` against a feedthrough `|D|≈170`, i.e. most of `Δ_ℓ` is
> already explained by the current-layer source (consistent with k\*≈1 and the ±v_ℓ
> tautology). So this is a **method-validating Go on the debug model**; the load-bearing
> test of whether structured cross-layer propagation *matters* is the 24-layer **410m
> primary lane, which is blocked on the SAE-source decision** (below).

---

## The five Gate-3 questions

**1. What fraction of Δ_ℓ variance does the decomposition reconstruct, per λ_L1?**
Held-out R² (vs the per-depth-mean baseline), λ_L1 sweep ∈ {0.1, 0.3, 1, 3, 10}:

| λ_L1 | SSM test R² | SSM test L0 | baseline test R² | baseline L0 |
|---|---|---|---|---|
| 0.1 | **+0.863** | 0.0086 | +0.856 | 0.0088 |
| 0.3 | **+0.862** | 0.0045 | +0.854 | 0.0050 |
| 1   | **+0.858** | 0.0015 | +0.850 | 0.0018 |
| 3   | **+0.848** | 0.0008 | +0.841 | 0.0009 |
| 10  | **+0.820** | 0.0006 | +0.814 | 0.0006 |

The knee is λ_L1 ≈ 0.3–1 (R² ≈ 0.86 at L0 ≈ 0.0015–0.0045), matching the
`ph3-altmin-tuning` calibration. The train–test gap is small and stable (≈ +0.018
throughout) — the operators generalise, they are not memorising.

**2. How does this compare to a per-layer SAE at the same L0? — SSM ON/ABOVE the
baseline at every operating point.** The baseline is the **maximally matched** per-layer
sparse dictionary: the *same* model with the cross-layer state ablated
(`use_state=False` → `Δ_ℓ ≈ b_ℓ + D_ℓ u_ℓ`), fit with the identical optimiser, source
init, and L1 sweep. It is a per-layer SAE on `Δ_ℓ` in everything but the state term, so
the SSM−baseline gap **isolates exactly the reconstruction value of cross-layer
propagation** (`C_ℓ z_ℓ`). Interpolating the baseline curve onto the SSM's L0 grid:

| matched L0 | SSM R² | baseline R² | gap |
|---|---|---|---|
| 0.0008 | +0.848 | +0.831 | **+0.018** |
| 0.0015 | +0.858 | +0.847 | **+0.011** |
| 0.0045 | +0.862 | +0.853 | **+0.009** |
| 0.0086 | +0.863 | +0.855 | **+0.008** |

Every gap is positive → **go criterion (recon on/above baseline Pareto) MET**. The gap
*widens* toward higher sparsity (the state earns more of its keep when each layer's own
source is starved), but it is small in absolute terms on this 6-layer model. See
`figures/pareto.png`.

**3. How does it compare to a crosscoder baseline? — NOT RUN (debug lane).** A
crosscoder/MLSAE baseline is a Phase 4 Step 1 deliverable (the full baseline suite:
per-layer SAE, crosscoder/MLSAE, Tuned Lens, CLT). Out of scope for this debug Gate; the
per-layer-SAE baseline (Q2) is the one the go criterion is keyed to.

**4. Are the features seed- and dataset-stable? — YES** (Phase 3 Step 4,
`runs/2026-06-24-ph3-step4-70md/`). Seed stability (3 seeds, identity-matched decoder
directions `D_ℓ[:,j]`, the fixed-dictionary specialisation of the plan's max-cosine):
**match-rate@0.7 = 0.99** (mean cos 0.92) — clears the ≥0.7 bar by a wide margin.
Dataset stability (disjoint halves): top-100 **Jaccard 0.89** (94/100 shared),
shared-direction match@0.7 0.84. No collapse-to-bias (‖ops‖=173 ≫ ‖b‖=4.7, L0≈0.001≠0).

**5. Do top features look interpretable? — YES.** Top-20 features by |u| mass have mean
relation-class **purity 0.93** with clean themes (feat 8180→religion, 27754→language/
French, 4110→geography). Reconstruction eyeball: median per-example R²=1.00 over 10
random examples (`figures/reconstruction.png`).

---

## Decision

**☑ Go → Phase 4 — on the 70m debug lane**, with one explicit caveat carried forward.

The pre-registered go criterion ("reconstruction on or above the per-layer SAE Pareto at
comparable sparsity **and** seed stability ≥ 0.7 for the top-100 features") is **met on
both clauses**. The method is validated end-to-end: the optimiser descends, sparsifies
with exact zeros, generalises to held-out examples, does not collapse, produces stable
interpretable features, and beats the matched per-layer baseline at every sparsity.

**Caveat — small margin, debug model.** The reconstruction win over independent
per-layer dictionaries is **+0.008 to +0.018 R²**. On a 6-layer model with k\*≈1 and a
±v_ℓ tautology that loads most of Δ_ℓ onto the current-layer source, that is about what
the cross-layer state *can* contribute. This Gate-3 Go therefore certifies the **method**,
not yet the **scientific claim**. Whether structured cross-layer propagation is a
materially better account of the residual updates than independent per-layer SAEs is a
question for the **24-layer 410m primary lane**, where the depth axis is long enough for
the state to carry signal the feedthrough cannot.

**Blocker on the primary lane (the next action).** 410m-deduped has **no public per-layer
residual SAE** (`sae-pythia-410m-65k` is non-deduped + MLP-only; `*-deduped-*` 404 — see
the Phase-3-scoping log). The 410m run requires either (a) **training minimal per-layer
residual SAEs** on 410m-deduped (plan-sanctioned fallback, embarrassingly parallel, the
cheapest thing in the pipeline) or (b) a cheaper sparse-init substitute. This is the
gating decision for the primary Phase-3 fit and the honest Phase-4 baseline comparison.

**Parsimony framing (unchanged from Gate 2).** Because H2 failed, the fully-depth-indexed
Branch B is *larger* than a stationary kernel by design; we do **not** claim a parameter
win. Report Branch B's parameter count honestly against CLT in Phase 4. The Gate-3 win is
a reconstruction-at-matched-sparsity win over a per-layer SAE, not a parsimony win.

---

## Reproduce

```
scripts/step3_decompose.py --config configs/pythia-70m.yaml --desc ph3-decomp-70md \
    --rank 32 --lambda-l1 1.0 --outer-iters 10 --device cuda
scripts/step4_sanity.py    --config configs/pythia-70m.yaml --desc ph3-step4-70md \
    --rank 32 --lambda-l1 1.0 --outer-iters 10 --device cuda
scripts/gate3_pareto.py    --config configs/pythia-70m.yaml --desc ph3-gate3-70md \
    --rank 32 --lambdas 0.1,0.3,1,3,10 --outer-iters 10 --device cuda
```

Pareto figure `figures/pareto.png`; numerics `tables/pareto.csv`, `tables/gate3_pareto.json`;
stability/sanity in `runs/2026-06-24-ph3-step4-70md/tables/`.

## Caveats carried forward
1. **Debug lane only.** 70m/6-layer is too shallow to be a scientific target; the Gate-3
   Go certifies the method. Primary = 410m (24 layers), pending the SAE-source decision.
2. **Small SSM-over-baseline margin** (+0.008…+0.018 R²) — the state is a minor term on
   6 layers; re-evaluate magnitude on 410m.
3. **±v_ℓ tautology** loads most of Δ_ℓ onto the current-layer feedthrough; the headline
   "competitive reconstruction" is partly mechanical (shared with the baseline, so it
   cancels in the *gap*, which is the load-bearing number).
4. **No crosscoder/CLT baseline yet** (Phase 4 Step 1).
5. **Checkpoints not persisted** — the debug fits are cheap to regenerate (≈5 min/fit);
   the 410m primary fit will populate `runs/gate3/checkpoints/`.
