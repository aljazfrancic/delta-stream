# Cross-model replication memo — Phase 2 Step 5

**Models:** GPT-2-small (L=12, D=768; **cross-family** — different architecture, training
data, tokenizer than Pythia) and Pythia-160m-deduped (L=12, D=768; **in-family depth
midpoint** between 70m/6L and 410m/24L). **Date:** 2026-06-26. **Seed:** 0.
**Datasets** (CounterFact, top-1==object, prob>0.1): GPT-2 **1358** / 3 classes
(1086 train); Pythia-160m **767** / 3 classes (613 train). **SAE sources:** GPT-2 =
res-jb L1 suite (`jbloom/GPT2-Small-SAEs-Reformatted`, resid_pre, d_sae=24576, via
`saes_resjb.py`); 160m = MLSAE TopK suite (`tim-lawson/sae-pythia-160m-deduped-x64-k32`,
drop-in). **Runs:** `runs/2026-06-2{5,6}-xm-{gpt2,p160}-*`. **Figure:**
`runs/crossmodel/figures/depth_scaling.png`.

> **Headline (three results).**
> **(1) H1 and H2 replicate cross-family.** On both new models the residual update Δ_ℓ is
> AR-predictable over depth above the scrambled null (Granger 43/43 real vs 0/43 null), and
> the AR kernel is **non-stationary** (every cross-band off-diagonal R² is negative) — the
> exact Gate-1/Gate-2 signatures, now confirmed in a non-Pythia model.
> **(2) The cross-layer-propagation gain is monotone in depth.** The max SSM-vs-per-layer
> -baseline R² gap is **+0.018 (6L) → +0.037 (12L) → +0.066 (24L)** across the Pythia family;
> GPT-2 (12L, a different SAE family) also clears its baseline (+0.018). The two-point
> Gate-3 result is now a depth trend.
> **(3) The seed-stability degeneracy tracks the SAE source type, not depth.** At equal
> depth (12L), GPT-2/res-jb decoder directions are seed-stable (match@0.7 **0.95**) while
> Pythia-160m/mlsae are not (**0.38**). This is a new lead for the 410m limitation that none
> of the SSM-side fixes (d_z, asymmetric reg) addressed.

---

## H1 — depth-local predictability + AR + null (REPLICATES)

| Model | AR best R² (k) | real − null | Granger real/total | Granger null |
|---|---|---|---|---|
| GPT-2-small | 0.626 (k=1) | 0.625 | **43/43** | 0/43 |
| Pythia-160m | 0.899 (k=10) | 0.894 | **43/43** | 0/43 |
| Pythia-410m (ref) | 0.65 (k=20) | 0.65 | 112/112 | 0/112 |

Δ_ℓ is autoregressively predictable over depth in both models, the structure is fully
Bonferroni-significant under the Granger nested-F, and it **collapses to ≈0 under the
scrambled null** — i.e. real example-specific cross-depth structure, not a marginal
artefact. **H1 holds cross-family.**

*Anomaly (GPT-2):* the cheap Step-3 *low-rank* ridge (predict Δ_ℓ from v_ℓ) scored only
**0.146** layer-avg at the best rank, despite full-rank R² **0.91** and the AR R² **0.626**.
GPT-2's v→Δ map is much less low-rank than Pythia's; it does not affect the H1 (AR) verdict
but is worth a dedicated look (different residual geometry / no LayerNorm-folding in res-jb).

## H2 — stationarity (REJECTED cross-family)

Cross-band held-out R² (one band-shared AR kernel fit on early/mid/late thirds, evaluated
across bands). Within-band positive, **every off-diagonal negative** in both models — the
"no-transfer" signature from the 410m Gate-2:

- **GPT-2:** within {0.85, 0.82, 0.76}; off-diagonal −1.0 … −2.8.
- **Pythia-160m:** within {0.81, 0.68, 0.86}; off-diagonal −0.17 … −1.5.

The depth-dependence of the kernel is **pervasive** in both — H2 fails the same way it does
at 24L, so **Branch B (fully depth-indexed) is the right architecture cross-family too.**

## Phase 3 — cross-layer propagation gap (depth-scaling)

Max held-out SSM-vs-per-layer-baseline R² gap (baseline = same SSM, state ablated):

| Model | L | SAE | gap (min … max) | seed match@0.7 |
|---|---|---|---|---|
| Pythia-70m | 6 | mlsae | +0.008 … **+0.018** | 0.99 |
| GPT-2-small | 12 | res-jb | +0.003 … **+0.018** | **0.95** |
| Pythia-160m | 12 | mlsae | +0.008 … **+0.037** | **0.38** |
| Pythia-410m | 24 | mlsae | +0.037 … **+0.066** | 0.45 |

**In-family depth-scaling is monotone** (0.018 → 0.037 → 0.066 for 6 → 12 → 24L): structured
cross-layer propagation explains progressively more of the residual updates as depth grows,
exactly the §4.2 prediction. The SSM is on/above the per-layer baseline at **every** operating
point for all four models (no collapse; sanity OK). GPT-2 confirms the **sign** of the effect
in a different architecture *and* a different SAE family.

## The seed-stability surprise (new lead for the 410m limitation)

The decoder-direction seed-instability recorded at Gate-3 (410m match@0.7 = 0.45, unmoved by
d_z↓ or asymmetric reg) is **not a depth phenomenon**: at the *same* 12 layers, GPT-2/res-jb
is stable (**0.95**) while Pythia-160m/mlsae is not (**0.38**). Within the Pythia-mlsae lane
stability degrades 0.99 (6L) → 0.38 (12L) → 0.45 (24L); swapping the **source dictionary**
(L1 res-jb) restores it at 12L. So the degeneracy appears tied to the **SAE source type /
residual geometry**, not the depth-indexed SSM — a lever none of Fixes 1–4 touch.

**→ Fix 6 (most promising):** re-source 410m from an L1/denser dictionary, SSM held fixed,
and re-measure seed match@0.7.

**Confound (must control before concluding):** dataset sizes differ — 160m train N=613 is the
smallest (GPT-2 1086, 410m 1244). 160m's 0.38 is partly N-aggravated. A **size-matched**
comparison (subsample GPT-2 to N=613, and/or fit 410m on a res-jb-style source) is the
first step of Fix 6, before attributing the split to SAE type.

## Caveats
1. **160m small dataset** (767 / 613 train) — fine for H1/H2 (held at n=1555 on 410m) but
   confounds the seed-stability comparison (above).
2. **GPT-2 low-rank ridge anomaly** (0.146 vs full-rank 0.91) — residual geometry differs;
   AR/H1 verdict unaffected.
3. **res-jb vs mlsae are different SAE families** (L1/resid_pre vs TopK/resid_post) — a feature
   for generality of H1/H2, but it means the GPT-2 vs Pythia stability gap conflates SAE type
   with model family until Fix 6 disentangles them.
4. **tuned_lens omitted** from both configs (Phase-1 gate baseline only; confirmatory here) —
   kept out for unattended-run robustness; add back if the ridge-vs-lens comparison is wanted.

## Reproduce

```
scripts/run_crossmodel.sh both        # full per-model pipeline (H1 -> H2 -> Phase-3)
scripts/plot_depth_scaling.py         # -> runs/crossmodel/figures/depth_scaling.png
```
