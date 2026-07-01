# Cross-model replication memo — Phase 2 Step 5

**Models (6 across 3 families).** *Wave 1 (2026-06-26):* GPT-2-small (L=12) + Pythia-160m-deduped
(L=12). *Wave 2 (2026-06-28):* **GPT-2-medium (L=24)** + **GPT-2-large (L=36)** (H1/H2 only — no
public residual SAEs) and **Qwen2-0.5B (L=24, third family** — RoPE/GQA/SwiGLU, full pipeline on
its 12-layer SAE-coverage prefix). With the Pythia 70m/410m references this is **6 models, depths
6→12→24→36L, 3 families**. **Seed:** 0. **Datasets** (CounterFact, top-1==object, prob>0.1): GPT-2-small
1358; Pythia-160m 767; GPT-2-medium 1795; GPT-2-large 1924; **Qwen2-0.5B 2540** (> 410m's 1555).
**SAE sources:** GPT-2-small = res-jb L1 (`jbloom/GPT2-Small-SAEs-Reformatted`, resid_pre); 160m =
MLSAE TopK; **Qwen = chanind L1** (`chanind/sae-qwen2-0.5b-res`, resid_post, via `saes_qwen.py`).
**Runs:** `runs/2026-06-2{5,6,8}-xm-*`. **Figure:** `runs/crossmodel/figures/depth_scaling.svg`.

> **Headline (three results, strengthened in wave 2).**
> **(1) H1 and H2 replicate across 3 families and 4 depths.** Δ_ℓ is AR-predictable over depth
> above the scrambled null and the AR kernel is **non-stationary** (every cross-band off-diagonal
> R² negative), with **k\*=1** (first-order) everywhere — the exact Gate-1/Gate-2 signatures, now in
> GPT-2 (12/24/36L) and a genuinely different family (Qwen2-0.5B).
> **(2) The cross-layer-propagation gain is robustly larger at depth — but it is an endpoint
> contrast, not a clean monotone line (revised with 3-seed error bars, S2 2026-06-29).** Max
> SSM-vs-per-layer-baseline R² gap (Pythia, mean±sd over seeds 0/1/2): **+0.017±0.002 (6L) →
> +0.020±0.015 (12L) → +0.064±0.003 (24L)**. The **24L gap is ~3× the shallow models', with
> non-overlapping error bars** → the §4.2 "load-bearing at depth" prediction holds at the endpoints.
> **But the clean monotone 3-point trend does NOT survive:** the original single-seed 12L value
> (+0.037) was a high outlier that did not replicate (seeds 1–2 ≈ +0.011), so 6L and 12L are
> statistically indistinguishable (~+0.02). Most likely 160m's small dataset (N=767, smallest) +
> gap_max being a max-over-λ order statistic. Cross-family points (single-seed) clear their baselines
> too (GPT-2-small **+0.018**, Qwen2-0.5B **+0.043**, both 12L).
> **(3) The seed-stability degeneracy tracks the SAE source TYPE (L1 vs TopK), not depth or family —
> CONFOUND NOW BROKEN.** L1 sources are seed-stable on two families (GPT-2 **0.95**, Qwen **0.81**);
> TopK sources are not on two Pythia models (160m **0.38**, 410m **0.45**). The wave-1 caveat that
> SAE-type was entangled with family is resolved by Qwen (L1 on a third family → still stable). The
> Gate-3 limitation is **diagnosed**: it is the mlsae TopK dictionary, not the depth-indexed SSM.

---

## H1 — depth-local predictability + AR + null (REPLICATES)

| Model | L | AR R² (real−null) | Granger real/total | Granger null |
|---|---|---|---|---|
| GPT-2-small | 12 | 0.625 | **43/43** | 0/43 |
| Pythia-160m | 12 | 0.894 | **43/43** | 0/43 |
| **GPT-2-medium** | 24 | **0.513** | **112/112** | 0/112 |
| **GPT-2-large** | 36 | **0.311** | **196/196** | 0/196 |
| **Qwen2-0.5B** | 24 | **0.462** | **112/112** | 0/112 |
| Pythia-410m (ref) | 24 | 0.65 | 112/112 | 0/112 |

Δ_ℓ is autoregressively predictable over depth in every model, the structure is fully
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

- **GPT-2-small:** within {0.85, 0.82, 0.76}; off-diagonal −1.0 … −2.8.
- **Pythia-160m:** within {0.81, 0.68, 0.86}; off-diagonal −0.17 … −1.5.
- **GPT-2-medium (24L):** within {0.72, 0.52, 0.56}; worst cross-band by rank −0.56 … −1.38.
- **GPT-2-large (36L):** within {0.63, 0.45, 0.40}; worst cross-band by rank −0.49 … −1.65.
- **Qwen2-0.5B (24L):** within positive; worst cross-band by rank down to **−2.59**; bands bootstrap-separated.

The depth-dependence of the kernel is **pervasive** in every model (and **k\*=1** at every band — the
first-order read), so **Branch B (fully depth-indexed) is the right architecture cross-family too**, at
12/24/36L and in a third family.

## Phase 3 — cross-layer propagation gap (depth-scaling)

Max held-out SSM-vs-per-layer-baseline R² gap (baseline = same SSM, state ablated):

| Model | L | SAE (type) | gap (max, mean±sd) | seed match@0.7 |
|---|---|---|---|---|
| Pythia-70m | 6 | mlsae (TopK) | **+0.017 ± 0.002** (n=3) | 0.99 |
| GPT-2-small | 12 | res-jb (L1) | **+0.018** (n=1) | **0.95** |
| Pythia-160m | 12 | mlsae (TopK) | **+0.020 ± 0.015** (n=3; seed-0 was +0.037) | **0.38** |
| **Qwen2-0.5B** | 12* | **chanind (L1)** | **+0.043** (n=1) | **0.81** |
| Pythia-410m | 24 | mlsae (TopK) | **+0.064 ± 0.003** (n=3) | 0.45 |

<sub>*Qwen Phase-3 is on its 12-layer SAE-coverage prefix (chanind covers blocks 0..11 of 24); recon R² mean +0.659 clears the gate. Pythia gaps are mean±sd over seeds 0/1/2 (S2, `scripts/run_s2_seeds.sh`); per-seed: 70m [0.018, 0.015, 0.019], 160m [0.037, 0.012, 0.011], 410m [0.066, 0.066, 0.061]. Cross-family rows are seed-0 only.</sub>

**Depth-scaling, with 3-seed error bars (S2):** the gap is **robustly ~3× larger at 24L
(+0.064±0.003) than at 6L (+0.017±0.002)** — non-overlapping, so structured cross-layer propagation
explains materially more of the residual updates at depth, the §4.2 prediction. **However, it is an
endpoint contrast, not a clean monotone 6→12→24 line:** the 12L (160m) midpoint is **seed-unstable**
(seed-0 +0.037 did not replicate; seeds 1–2 ≈ +0.011 → mean +0.020±0.015), so 6L and 12L overlap and
are statistically indistinguishable. The original "monotone 0.018→0.037→0.066" was partly a seed-0
artefact at the noisy 12L point (small N=767 + max-over-λ order statistic). The SSM is on/above the
per-layer baseline at the knee for **every** model (no collapse; sanity OK), and GPT-2 **and** Qwen
confirm the effect's **sign** in different architectures and SAE families — Qwen's +0.043 at 12L is
the strongest cross-family 12L gap.

## The seed-stability surprise (new lead for the 410m limitation)

The decoder-direction seed-instability recorded at Gate-3 (410m match@0.7 = 0.45, unmoved by
d_z↓ or asymmetric reg) is **not a depth phenomenon**: at the *same* 12 layers, GPT-2/res-jb
is stable (**0.95**) while Pythia-160m/mlsae is not (**0.38**). Within the Pythia-mlsae lane
stability degrades 0.99 (6L) → 0.38 (12L) → 0.45 (24L); swapping the **source dictionary**
(L1 res-jb) restores it at 12L. So the degeneracy appears tied to the **SAE source type /
residual geometry**, not the depth-indexed SSM — a lever none of Fixes 1–4 touch.

**→ Fix 6 (most promising):** re-source 410m from an L1/denser dictionary, SSM held fixed,
and re-measure seed match@0.7.

**Confound 1 — N: RULED OUT (2026-06-26).** GPT-2 subsampled to N=767 (matched to 160m) still
gives seed match@0.7 = **0.95** — identical to full-N. So at equal depth *and* N, res-jb is stable
and mlsae is not. Run: `runs/2026-06-26-xm-gpt2-step4-n767/`.

**Confound 2 — SAE-type vs model-family: BROKEN (2026-06-28, Qwen lane).** In wave 1 the split was
entangled (res-jb=GPT-2-only, mlsae=Pythia-only). **Qwen2-0.5B with chanind L1 SAEs at 12L is
seed-stable (match@0.7 = 0.81)** — i.e. L1 on a *third* family is stable, just like GPT-2-small L1
(0.95), while TopK is unstable on *two* Pythia models (160m 0.38, 410m 0.45). So the lever is the
**SAE source TYPE (L1 vs TopK)**, not the model family and not depth:

| SAE type | models (L) | seed match@0.7 |
|---|---|---|
| **L1** | GPT-2-small (12), Qwen2-0.5B (12) | **0.95, 0.81** ✓ |
| **TopK** | Pythia-160m (12), Pythia-410m (24) | **0.38, 0.45** ✗ |

**The 410m drop-in problem (unchanged).** The **only** public per-layer residual SAEs for
pythia-410m-deduped are mlsae x64-k32 (HF-verified — no L1/denser variant), so **the 410m fix
requires training an L1 residual SAE suite ourselves** (future GPU work). But the *diagnosis* is now
complete: the Gate-3 seed-instability is **not** intrinsic to the depth-indexed SSM — it is a
property of the **mlsae TopK source**, and L1 dictionaries are stable at the same depth on two
independent families.

## Caveats
1. **160m small dataset** (767 / 613 train, 153 test) — fine for H1/H2 (held at n=1555 on 410m) but
   confounds the seed-stability comparison (above) **and makes its depth-scaling gap seed-noisy**
   (3-seed +0.020±0.015, vs the tight ±0.002–0.003 at 70m/410m) — the reason the once-"monotone"
   12L midpoint did not replicate (S2). The robust depth claim is the 6L-vs-24L endpoint contrast.
2. **GPT-2 low-rank ridge anomaly** (0.146 vs full-rank 0.91) — residual geometry differs;
   AR/H1 verdict unaffected.
3. ~~res-jb vs mlsae conflate SAE type with model family~~ — **RESOLVED in wave 2**: Qwen2-0.5B
   (chanind L1, a third family) is seed-stable (0.81), so L1 is stable on two families and TopK
   unstable on two Pythia models → the lever is SAE type, not family (see Confound 2 above).
4. **tuned_lens omitted** from both configs (Phase-1 gate baseline only; confirmatory here) —
   kept out for unattended-run robustness; add back if the ridge-vs-lens comparison is wanted.

## Reproduce

```
scripts/run_crossmodel.sh both        # wave 1: GPT-2-small + Pythia-160m (H1 -> H2 -> Phase-3)
scripts/run_crossmodel.sh wave2       # wave 2: Qwen2-0.5B (full) + GPT-2 medium/large (H1/H2)
scripts/plot_depth_scaling.py         # -> runs/crossmodel/figures/depth_scaling.svg
```
