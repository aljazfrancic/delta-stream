# Gate 3 decision memo — Phase 3 Decomposition

**Decision: ☑ GO → Phase 4 (on the Pythia-410m-deduped PRIMARY lane), with a recorded
seed-stability limitation.** The reconstruction clause of the go criterion passes
decisively and seed-robustly; the seed-stability clause fails *structurally* and is
recorded as a limitation (per the plan's "reconstruction fine, seed unstable → record as a
limitation" branch). **Date:** 2026-06-25.

**Primary model:** Pythia-410m-deduped (L=24, D=1024). **Sources:** public per-layer
**residual** SAEs from the MLSAE suite `tim-lawson/sae-pythia-410m-deduped-x64-k32-layers-{0..23}`
(Lawson et al. 2024; M=65536, TopK k=32; `u_0` zero-init — `v_0` has no embedding SAE).
A dense `u[N,L,65536]` is ~10 GB, so each layer is pruned to its **alive** latents
(active on ≥1 example — standard dead-latent removal; Branch B permits per-layer width
`M_ℓ`) and padded to `M*=4433` → `u[1555,24,4433]` (662 MB). Matched SAE reconstruction
R² mean **+0.598** across 24 layers (loader sanity; two weak public SAEs at v_18/v_19,
R²=0.21/0.31). **Dataset:** 1555 confidence-filtered CounterFact prompts (organization 711
/ geography 511 / language 333), residual stream at the final prompt token → `Δ[1555,24,1024]`.
**Split:** 1244 train / 311 test, seed 0. **Fit:** r=32, d_z=32, λ_L1 swept, λ_reg=1e-4,
10 outer iters, GPU (GTX 1050, ~1.5 GB). Runs: `runs/2026-06-25-ph3-gate3-410md/` (Pareto),
`runs/2026-06-25-ph3-step4-410md/` (stability/sanity), `runs/2026-06-25-ph3-step4-410md-reg2/`
(λ_reg=1e-2 retune), `runs/2026-06-25-ph3-decomp-410md/` (optimiser debug).

> **Headline (two parts, both honest):**
> **(1) The central claim holds, with margin.** On the 24-layer primary model the Branch B
> SSM reconstructs Δ_ℓ **above** the matched per-layer baseline at every sparsity (R² gap
> **+0.037 → +0.066**, vs only +0.008…+0.018 on the 6-layer 70m) — structured cross-layer
> propagation is a *materially* better account of the residual updates than independent
> per-layer SAEs, and this is fully **seed-robust** (every seed → R²=0.880, same gap;
> `|C|` *grows* under sparsity, 27.9→50.0). This is the proposal §4.2 "H2 fails → CLT's
> free parameters are load-bearing" branch shown to be reconstruction-positive.
> **(2) The recovered decoder directions are not seed-unique.** Top-100 feature
> decoder-direction stability across re-inits is **match-rate@0.7 = 0.45** (mean cos 0.60),
> **below the 0.7 bar and unmoved by 100× stronger λ_reg** (0.45 at both 1e-4 and 1e-2 while
> R² is identical) — a *structural* equivalent-minima degeneracy from the SSM's
> state/feedthrough trade-off, **recorded as a limitation**. The sparse features themselves
> (fixed pretrained SAE latents) are dataset-stable (Jaccard **0.89**); only the learned
> Δ-mapping per feature is non-unique, and ~45% of top features clear the bar.

---

## The five Gate-3 questions (primary lane, 410m)

**1. What fraction of Δ_ℓ variance does the decomposition reconstruct, per λ_L1?**
Held-out R² (vs the per-depth-mean baseline), λ_L1 ∈ {0.3, 1, 3, 10, 30}:

| λ_L1 | SSM test R² | SSM test L0 | baseline test R² | baseline L0 |
|---|---|---|---|---|
| 0.3 | **+0.867** | 0.0452 | +0.816 | 0.0382 |
| 1   | **+0.857** | 0.0149 | +0.808 | 0.0133 |
| 3   | **+0.847** | 0.0110 | +0.802 | 0.0104 |
| 10  | **+0.786** | 0.0091 | +0.746 | 0.0088 |
| 30  | **+0.698** | 0.0098 | +0.653 | 0.0098 |

**2. How does this compare to a per-layer SAE at the same L0? — SSM ABOVE the baseline at
every operating point.** Baseline = the *same* SSM with the cross-layer state ablated
(`use_state=False` → `Δ_ℓ ≈ b_ℓ + D_ℓ u_ℓ`), identical optimiser / source init / L1 sweep,
so the gap isolates exactly the value of cross-layer propagation (`C_ℓ z_ℓ`). Baseline R²
interpolated onto the SSM L0 grid:

| matched L0 | SSM R² | baseline R² | gap |
|---|---|---|---|
| 0.0091 | +0.786 | +0.720 | **+0.066** |
| 0.0098 | +0.698 | +0.662 | **+0.037** |
| 0.0110 | +0.847 | +0.803 | **+0.044** |
| 0.0149 | +0.857 | +0.809 | **+0.048** |

Every gap is strongly positive → **reconstruction clause MET, with margin** (3–6× the 70m
debug gap). `ssm_on_or_above_baseline = True`. See `figures/pareto.png`.

**3. How does it compare to a crosscoder baseline? — NOT RUN.** Crosscoder / MLSAE / CLT
are the Phase 4 Step 1 baseline suite; the go criterion is keyed to the per-layer-SAE
baseline (Q2).

**4. Are the features seed- and dataset-stable? — DATASET yes, SEED no (the limitation).**
*Dataset stability* (disjoint halves): top-100 **Jaccard 0.89** (94/100 shared),
shared-direction match@0.7 0.86. *Seed stability* (3 seeds, identity-matched decoder
directions `D_ℓ[:,slot]`): **match-rate@0.7 = 0.45** (mean cos 0.60) — **below the ≥0.7
bar.** Diagnosis: all three seeds reach the *same* reconstruction (R²=0.880/0.881/0.880)
but different decoder directions → **equivalent-minima degeneracy**, not under-convergence.
The plan's prescribed remedy (stronger regularisation) was tried — λ_reg 1e-4 → 1e-2 (100×)
left seed stability **unchanged at 0.45** while reconstruction was identical — so the
degeneracy is **structural** (the SSM's state `C z` and feedthrough `D u` trade off to fit
the same Δ via different directions). The plan's other remedy (dictionary-alignment) does
not apply: the SAE dictionary is fixed, so the features are the same atoms across re-fits;
the gauge freedom lives in the operators, not the dictionary. **Recorded as a limitation.**
No collapse (‖ops‖=252 ≫ ‖b‖=11, L0≈0.009).

**5. Do top features look interpretable? — YES.** Top-20 features by |u| mass have mean
relation-class **purity 0.83**. Reconstruction eyeball: median per-example R² **0.98** over
10 random examples (0.94–0.99; `figures/reconstruction.png`).

---

## Decision

**☑ Go → Phase 4 — on the 410m primary lane, with the seed-stability limitation recorded.**

Clause-by-clause against the pre-registered go criterion ("reconstruction on/above the
per-layer SAE Pareto at comparable sparsity **and** seed stability ≥ 0.7 for the top-100
features"):

- **Reconstruction clause — MET, with margin.** +0.037…+0.066 R² over the matched per-layer
  baseline at every operating point; seed-robust (identical R²/gap across seeds). This is
  the load-bearing scientific result: at 24 layers, structured cross-layer propagation
  materially beats independent per-layer SAEs — the §4.2 "H2-fails → CLT parameters are
  load-bearing" branch, shown reconstruction-positive.
- **Seed-stability clause — NOT met (0.45 < 0.7), recorded as a structural limitation.**
  Per the plan's "reconstruction fine / seed unstable" branch ("record this as a
  limitation"), with both pre-registered remedies exhausted/inapplicable (stronger reg
  tried+failed; dictionary-alignment N/A for a fixed dictionary).

**Why GO and not a hard no-go.** The project's central claim (proposal §4.2) is a
*reconstruction/parsimony* claim, and it is solid and seed-robust. The instability is
confined to the **identifiability of the learned decoder directions**, not to the claim and
not to the sparse feature set: the features are fixed pretrained SAE latents and are
dataset-stable (0.89). Phase 4's causal existence proof already filters candidate features
by seed/dataset consistency, so it will draw from the **seed-stable subset** (~45% of top
features clear ≥0.7) — the instability narrows the *yield* of causally-testable features,
it does not invalidate them. The hard-pivot branch ("both fail") does not apply because
reconstruction does not fail.

**Phase-4 carry-over.** (a) Restrict the causal-existence-proof candidate pool to
seed-stable (match@0.7) features; (b) report the seed-stability degeneracy as a named
limitation of depth-indexed Branch B and discuss the state/feedthrough trade-off as its
mechanism; (c) consider a d_z-capacity sweep or a cross-seed decoder-alignment penalty as
future work to improve operator identifiability.

**Parsimony framing (unchanged).** H2 failed, so the fully depth-indexed Branch B is
*larger* than a stationary kernel by design; this is a reconstruction-at-matched-sparsity
win over a per-layer SAE, **not** a parameter-parsimony win. Report Branch B's parameter
count honestly against CLT in Phase 4.

---

## Caveats carried forward
1. **Seed-stability degeneracy (the headline limitation).** Decoder directions are not
   seed-unique (match@0.7 = 0.45); structural (state/feedthrough equivalent minima),
   unmoved by 100× λ_reg. Features and reconstruction are stable; the learned Δ-mapping is
   not. Phase 4 uses the seed-stable subset.
2. **Weak public SAEs at v_18 / v_19** (matched recon R² 0.21 / 0.31; alive sets ~3× denser).
   A property of those two MLSAE checkpoints, shared by SSM and baseline, so it **cancels in
   the gap**. Re-training those two SAEs is a possible Phase-4 polish.
3. **Per-layer alive compression** (497–4433 of 65536 latents kept per layer) — standard
   dead-latent removal of a fixed pretrained SAE, exact for reconstruction; the held-out
   LASSO re-infers within the alive column space. Branch B's per-layer `M_ℓ` makes this
   principled.
4. **±v_ℓ tautology** loads part of Δ_ℓ onto the feedthrough; shared by the baseline, cancels
   in the gap.
5. **No crosscoder / CLT baseline yet** (Phase 4 Step 1).
6. **Single rank / d_z point** (r=32, d_z=32). Rank sweep {16…256} + d_z sensitivity are
   Phase-4 items; a smaller d_z is also a candidate fix for the seed-stability degeneracy.
7. **Checkpoints not yet persisted** — fits are ~16 min each to regenerate; Phase-4 ablation
   work will populate `runs/gate3/checkpoints/`.

## Reproduce

```
scripts/build_410m_sources.py --config configs/pythia-410m-deduped.yaml --desc ph3-sources-410md
scripts/gate3_pareto.py       --config configs/pythia-410m-deduped.yaml --desc ph3-gate3-410md \
    --sources-npz data/pythia-410m-deduped/sources_mlsae.npz --rank 32 --lambdas 0.3,1,3,10,30 --outer-iters 10 --device cuda
scripts/step4_sanity.py       --config configs/pythia-410m-deduped.yaml --desc ph3-step4-410md \
    --sources-npz data/pythia-410m-deduped/sources_mlsae.npz --rank 32 --lambda-l1 1.0 --outer-iters 10 --device cuda
# seed-stability retune (showed stronger reg does not recover it):
scripts/step4_sanity.py       --config configs/pythia-410m-deduped.yaml --desc ph3-step4-410md-reg2 \
    --sources-npz data/pythia-410m-deduped/sources_mlsae.npz --rank 32 --lambda-l1 1.0 --lambda-reg 1e-2 --outer-iters 10 --device cuda
```

---

# Appendix — 70m-deduped debug lane (method validation)

The 70m-deduped debug lane (L=6, D=512; public EleutherAI `sae-pythia-70m-deduped-32k`,
M=32768) validated the Branch B method end-to-end before the primary run. It cleared **both**
go-criterion clauses but with a **small reconstruction margin** (SSM−baseline +0.008…+0.018;
`|C|≈14` vs `|D|≈170`) and high seed stability (match@0.7 = 0.99) — the latter because the
6-layer model has few equivalent minima, which the 24-layer primary does not. So the debug
lane was a **method-validating** Go; the primary lane above is the scientific finding (large
reconstruction margin) *and* surfaces the seed-stability degeneracy that only appears at
depth. Runs: `runs/2026-06-24-ph3-gate3-70md/`, `runs/2026-06-24-ph3-step4-70md/`,
`runs/2026-06-24-ph3-decomp-70md/`. Debug-lane numbers: seed match@0.7 = 0.99, dataset
Jaccard 0.89, feature purity 0.93, eyeball median R² 1.00.
