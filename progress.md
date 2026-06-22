# Progress Tracker

Control panel for [*Structured Cross-Layer Propagation of Features in Transformers*](docs/README.md).
The phase docs under [docs/implementation-plan/](docs/implementation-plan/README.md) are the **pre-registration** and stay frozen — this file is the only place progress is recorded, so updating it never touches a pre-registered bar.

**Legend:** `[ ]` todo · `[~] ` in progress · `[x]` done · `[-]` skipped / N-A · `[!]` blocked

**Current focus:** _Phase 2 — Steps 1–2 done 2026-06-22 (**H2 FAILS → Branch B**; reconstruction **k\*≈1**). Next: Step 3 (token-position invariance) or scale dataset for Step 4_
**Start date:** 2026-06-20

---

## Dashboard

| Phase | Doc | Gate | Status |
|---|---|---|---|
| 1 · Diagnostic | [phase-1](docs/implementation-plan/phase-1-diagnostic.md) | **Gate 1** — is H1 supported above null? | `[x]` **PASS** |
| 2 · Characterization | [phase-2](docs/implementation-plan/phase-2-characterization.md) | **Gate 2** — does H2 (stationarity) hold? → picks Phase 3 branch | `[~]` **H2 NO → Branch B** (S1–2 done; S3–5 open) |
| 3 · Decomposition | [phase-3](docs/implementation-plan/phase-3-decomposition.md) | **Gate 3** — reconstruction competitive with baselines? | `[ ]` |
| 4 · Evaluation | [phase-4](docs/implementation-plan/phase-4-evaluation.md) | — deliverable: paper + code | `[ ]` |

**Phase 3 architecture branch (locked at Gate 2):** **B — state-space (fully depth-indexed, first-order)** `[x]` · `[ ]` A stationary convolutive · `[ ]` C segmented — see `runs/gate2/phase3_architecture.md`

> A phase that fails its gate **does not silently continue** — it pivots to writing up the negative result. Record the routing in the gate block below.

---

## Phase 1 — Diagnostic

> Goal: does `Δ_ℓ = v_{ℓ+1} − v_ℓ` have structured cross-layer dynamics at all, above null, after controlling for context leakage?

- [x] **Step 0** — [Environment](docs/implementation-plan/phase-1-diagnostic.md#step-0--environment) (env pinned, `layer_dynamics/` skeleton, run-logging w/ git commit + seed)
- [x] **Step 1** — [Assemble factual-recall dataset](docs/implementation-plan/phase-1-diagnostic.md#step-1--assemble-the-factual-recall-dataset) → `filtered_triples.parquet` (built; **smaller than target — see note**)
  - [x] CounterFact downloaded, reformatted to s-r-o convention (`NeelNanda/counterfact-tracing`, already s-r-o). Pararel/T-REx **not yet unioned**.
  - [x] Filtered (top-1 == object, prob > **0.1**), stratified with floor **300** → **1555 examples, 3 classes** (organization 711, geography 511, language 333; occupation/religion dropped)
  - [x] Sanity check: read 50 random rows (clean)
  > **Note:** Primary model is **`pythia-410m-deduped`** (still "Pythia-410m"; chosen for its public Tuned Lens). It confidently knows only ~1800 CounterFact facts, so the 5,000 / ≥1000-per-class targets are **not reached**. Threshold tuned 0.3→0.1 and class floor 500→300 (plan-sanctioned) to keep 3 classes. Phase 1 diagnostics are fine on 1555; **Phase 2 (≥1000/class) still needs scaling** (union Pararel/T-REx or larger model).
- [x] **Step 2** — [Collect residual-stream trajectories](docs/implementation-plan/phase-1-diagnostic.md#step-2--collect-residual-stream-trajectories) → HDF5 `residual_stream[1758, 25, 1024]` (`data/pythia-410m/residual_stream.h5`)
  - [x] Sanity check: `ln_final` + unembed of `v_L` recovers top-1 token (**5/5**)
- [x] **Step 3** — [Linear depth-local predictability](docs/implementation-plan/phase-1-diagnostic.md#step-3--linear-depth-local-predictability-h1-weak-form) (ridge, low-rank, vs zero / mean / identity / **Tuned-Lens** baselines) — **ridge R²=0.602** (full 0.921) ≫ mean 0, zero −1.54; identity 0.965. **GATE PASS: ridge 0.602 ≫ Tuned Lens −0.691** → Step 5 AR justified. Best rank 128 (not saturated).
- [x] **Step 4** — [CCA backstop](docs/implementation-plan/phase-1-diagnostic.md#step-4--cca-backstop) (shrinkage CCA, first 10 canonical correlations) — **held-out canonical corr ≈ 1.000 at every layer** (top-1 avg 0.9997), robust to shrinkage up to γ=0.99, **collapses to ~0 under scrambled null** (real−null = 1.006). Plan interpretation key → high-CC everywhere (17 layers high-R²/high-CC, 7 low-R²/high-CC). **Caveat:** the ≈1 is largely the definitional −v_ℓ ⊂ Δ_ℓ term (same tautology as Step 3 identity baseline, amplified by rogue residual dims: v_ℓ top-1 dim = 28% of variance, top-20 = 85%). CCA confirms **shared subspace = yes** (Gate-1 Q3) but exposes no structure ridge missed; substantive claim stays the Step 3 ridge R² vs mean/TL.
- [x] **Step 5** — [AR(k) on Δ + Granger causality](docs/implementation-plan/phase-1-diagnostic.md#step-5--ark-on-delta-and-granger-causality-h1-full-form) (`scripts/step5_ar.py`, `scripts/step5_context_null.py`, `diagnostics/autoregress.py`) — gate from Step 3 satisfied
  - [x] AR(k) fits with per-depth intercept μ_ℓ; k ∈ {1,2,3,5,10,15,20} → **best held-out R²=0.65** (k=20, rank 128; ≥0.47 all k), vs per-depth mean. (Predictors compressed to top-256 Δ PCs; target full-D.)
  - [x] Nested-F lag test (Wilks Λ + Rao F, top-32 PCA) → (depth × lag) heatmap → **112/112 cells Bonferroni-significant** (α=4.5e-4), significant out to lag 20
  - [x] **Null 1 — scrambled activations** → AR R²≈**0.0000** (best 7e-5), Granger **0/112** significant. Real−null R²=0.6496.
  - [x] **Null 2 — context perturbation** → AR R² unchanged (wrapped 0.639 → perturbed 0.696), Granger heatmap corr **0.964** (112→109/112 sig). **Depth-intrinsic** (Q6 survives). 863/1000 kept (wrapped top-1 recovered).
- [x] **Step 6** — [Gate 1 decision](docs/implementation-plan/phase-1-diagnostic.md#step-6--gate-1-decision) memo → `runs/gate1/memo.md` (route: **proceed to Phase 2**)

### ◆ Gate 1 — decision record
Answer (Y/N): (1) Δ predictable above trivial **Y** (ridge R²=0.602 ≫ mean 0, zero −1.54) · (2) ridge beats Tuned Lens **Y** (0.602 vs −0.691) · (3) shared subspace **Y** (CCA ≈0.9997, null≈0) · (4) AR structure **Y** (AR R²=0.65, Granger 112/112) · (5) survives scrambled null **Y** (AR R²≈0, Granger 0/112) · (6) survives context perturbation **Y** (heatmap corr 0.964, 109/112). Non-stationary-H1 flag: **false**.

Pre-registered routing — **tick the one that happened:**
- [x] **Proceed to Phase 2** — (1,2,4,5,6) all yes, H1 cleanly supported
- [ ] **Pivot — context-dependent** — (1,2,4,5) yes, (6) no → characterise context-reconstruction dynamics
- [ ] **Proceed to Phase 2 — non-stationary H1** — (1,2) yes, pooled AR R² low *but* Granger widely Bonferroni-significant under scrambled null (expect smoothly-varying / segmented branch)
- [ ] **Phase 1 is the paper — no TL separation** — (1) yes, (2) no → AR skipped
- [ ] **Phase 1 is the paper — negative result** — any of (1,4,5) fail and no escape route applies

Outcome: **Proceed to Phase 2 (H1 cleanly supported)** · Date: **2026-06-22** · Memo: `runs/gate1/memo.md`
- [x] `runs/gate1/figures/` (8 figures: R² heatmaps, CCA, Granger real/null profiles, context comparison)
- [x] `runs/gate1/tables/` (11 CSV/JSON of all numerics)
- [ ] Shared with mentors before committing to Phase 2

---

## Phase 2 — Characterization

> Goal: what *kind* of structure? Stationary? How deep? Generalises across relations / models?

- [x] **Step 1** — [Stationarity test](docs/implementation-plan/phase-2-characterization.md#step-1--stationarity-test) (`scripts/step1_stationarity.py`, `diagnostics/stationarity.py`) — **H2 (stationarity) FAILS, decisively & rank-robustly.** `k=5` (Granger sig to lag 20 but capped at ⌊L/3⌋−3=5 → **short-horizon only**), `r=256` (Step-3 saturation 128; spot-check promoted 128→256, pooled AR R² 0.467→0.591). Early/mid/late band kernels: within-band R² **0.74 / 0.58 / 0.67** (positive) but **every cross-band R² is negative** (early→mid −1.00, late→mid −1.89, etc.) — kernels don't transfer at all. Holds at every rank 32→256 (worst off-diag −0.32 even at r=32). Between-band ‖ΔW‖ ≈ **1.39–1.42** (≈ √2 orthogonal ceiling), principal angles **57–62°** mean. Bootstrap (n=200/band): between-band distance **80–104 SD** above within-band spread, all 3 pairs separated. No clean phase boundary (3 pair-distances ≈ equal, ratio 1.03) → **not segmented**; depth-dependence is pervasive → **Branch B (state-space / depth-indexed)**, likely *fully* depth-indexed not slowly-varying. Run: `runs/2026-06-22-ph2-step1-410md/`.
  > **Reconciles with Gate 1, not contradicts it:** Gate 1's high *pooled* AR R² (one compromise kernel + per-depth μ_ℓ) was explicitly only "consistent with" stationarity and deferred the direct test here; a generalist pooled kernel can score well while band-specific kernels fail to transfer. This is the pre-registered **"H2 fails"** outcome (proposal §4.2): H1 still holds, the dynamics are genuinely non-stationary, and CLT-style unconstrained per-(source,target) parameterisation is *validated* — "shared kernel along depth" is the wrong abstraction.
  > **Caveats carried forward:** (1) k capped → short-horizon only; long-horizon stationarity deferred to Phase 3 Step 4 MA recheck. (2) AR-on-Δ ≠ MA stationarity (proxy). (3) the ±v_ℓ shared term biases *toward* stationarity, so failing is the conservative direction. (4) 3 classes/1555 ex — per-class test is Step 4.
- [x] **Step 2** — [Effective kernel depth k\*](docs/implementation-plan/phase-2-characterization.md#step-2--effective-kernel-depth) (`scripts/step2_kstar.py`, `diagnostics/effective_depth.py`) — **reconstruction k\* ≈ 1 at every depth** (per-depth k\* median 1, max 2 (only ℓ=17); band k\* = 1/1/1). A single preceding update Δ_{ℓ-1} captures essentially all held-out predictable variance (per-depth best R² 0.83–0.94, dipping to 0.64 at ℓ17); **lags ≥ 2 add a median +0.018 R² (max +0.05)**. Scrambled null collapses the whole curve (median best R² **0.000**) → real example-specific structure. **Headline gap:** reconstruction k\*≈1 vs **Granger reach lag 20** (Step 5) — long-range depth structure is *statistically real but reconstruction-negligible*. Run: `runs/2026-06-22-ph2-step2-410md/`.
  > **Tautology caveat (per `cca-vdelta` memory):** k\*≈1 is substantially the mechanical Δ_ℓ ⊃ −v_ℓ ⊂ Δ_{ℓ-1} overlap, so it is a property of the AR-on-Δ parameterisation, **not** proof the transformer's genuine computational dependence is one layer deep. k\*≈1 *is* consistent with Branch B being a **first-order (Markov-in-depth) SSM** — good news for Phase 3 model order.
- [ ] **Step 3** — [Token-position invariance](docs/implementation-plan/phase-2-characterization.md#step-3--token-position-invariance) (4×4 / 3×3 cross-position confusion) — needs re-collecting activations at 3 extra token positions (~4 GB)
- [!] **Step 4 (blocked)** — [Generalisation across relation types](docs/implementation-plan/phase-2-characterization.md#step-4--generalisation-across-relation-types) — **needs dataset scaled to ≥1000/class first** (currently 3 classes / 1555 ex; see `counterfact-410m-yield`)
- [ ] **Step 5** — [Cross-model replication](docs/implementation-plan/phase-2-characterization.md#step-5--cross-model-replication) (Gemma-2-2B; needs Phase 1 Steps 3+5 on Gemma first; Llama optional)
- [~] **Step 6** — [Gate 2 decision](docs/implementation-plan/phase-2-characterization.md#step-6--gate-2-decision) memo — **architecture LOCKED (Branch B) from Steps 1–2**; Q3–5 outstanding (`runs/gate2/memo.md`)

### ◆ Gate 2 — decision record
Record: stationary? **NO** (Step 1) · k\* = **≈1** (Step 2; vs Granger reach 20) · token-position-invariant? **outstanding** (Step 3, ~4 GB re-collection) · circuit-general or -specific? **blocked** (Step 4, dataset <1000/class) · cross-model replicates? **deferred** (Step 5, Gemma/Llama don't fit 3 GB)

Architecture decision — **tick one** (copy into `runs/gate2/phase3_architecture.md`):
- [ ] **Branch A** — stationarity holds → block-Toeplitz convolutive kernel
- [x] **Branch B** — stationarity fails → state-space model (**fully depth-indexed**, not smoothed — drift is pervasive not gradual; **first-order**, k\*≈1)
- [ ] **Branch C** — clean phase transition → one kernel per segment (boundaries: _____)

Date: **2026-06-22** · Memo: `runs/gate2/memo.md` (architecture-locking; Q3–5 are scope/generality cross-checks that cannot flip the Step-1 branch)
- [x] `runs/gate2/figures/` (stationarity cross-band/bootstrap/angles, k\* curves; cross-class & cross-model N/A — Steps 4–5 outstanding)
- [x] `runs/gate2/phase3_architecture.md` (one-paragraph exact decomposition spec)

---

## Phase 3 — Decomposition

> Goal: fit the generative model — sparse sources + depth kernel (or SSM). The build step.

- [ ] **Step 1** — [Lock architecture from Gate 2](docs/implementation-plan/phase-3-decomposition.md#step-1--lock-the-architecture-based-on-gate-2) (copy branch into phase README)
- [ ] **Step 2** — [Define sparse sources {s_ℓ'}](docs/implementation-plan/phase-3-decomposition.md#step-2--define-sparse-source-representations-lbrace-s_ellrbrace) (per-layer SAE latents; single shared width M for Branch A)
- [ ] **Step 3** — [Loss + optimisation](docs/implementation-plan/phase-3-decomposition.md#step-3--loss-and-optimisation-branch-a--c) (alternating min; λ_L1 sweep; rank sweep; K = k\*)
- [ ] **Step 4** — [Sanity checks before Gate 3](docs/implementation-plan/phase-3-decomposition.md#step-4--sanity-checks-before-gate-3) — **all mandatory**
  - [ ] 1 · trivial-solution / collapse check
  - [ ] 2 · reconstruction eyeball (10 examples)
  - [ ] 3 · feature activations have expected structure (top-20)
  - [ ] 4 · seed stability (3 seeds)
  - [ ] 5 · dataset stability (disjoint halves)
  - [ ] 6 · MA-stationarity recheck (Branch A only)
- [ ] **Step 5** — [Gate 3 decision](docs/implementation-plan/phase-3-decomposition.md#step-5--gate-3-decision) memo

### ◆ Gate 3 — decision record
**Go criterion:** reconstruction on/above per-layer SAE Pareto at comparable sparsity **and** seed stability ≥ 0.7 (top-100 feature match).

Tick one:
- [ ] **Go → Phase 4**
- [ ] **No-go — under-expressive** → one round of retune (rank↑ / longer kernel / Branch C), then re-decide
- [ ] **No-go — unstable** → stronger regularisation / dictionary-alignment, record as limitation
- [ ] **No-go — both fail** → pivot: write up Phase 1+2 characterisation, decomposition becomes future-work

Recon R² (avg over layers): _____ · seed stability: _____ · Date: _____ · Memo: `runs/gate3/memo.md`
- [ ] `runs/gate3/figures/` (Pareto vs baselines, reconstruction plots, seed-stability matrices)
- [ ] `runs/gate3/checkpoints/` (trained decomposition, ready for Phase 4 ablation)

---

## Phase 4 — Evaluation and write-up

> Goal: baseline comparisons, the causal existence proof, the paper.

- [ ] **Step 1** — [Baselines, precisely specified](docs/implementation-plan/phase-4-evaluation.md#step-1--baselines-precisely-specified)
  - [ ] A · per-layer SAE · [ ] B · crosscoder / MLSAE · [ ] C · Tuned Lens · [ ] D · CLT (primary conceptual baseline; native + residual-write variants)
- [ ] **Step 2** — [Stability + robustness metrics](docs/implementation-plan/phase-4-evaluation.md#step-2--stability-and-robustness-metrics) (seed, dataset, cross-class)
- [ ] **Step 3** — [Causal existence proof](docs/implementation-plan/phase-4-evaluation.md#step-3--the-causal-existence-proof-the-critical-result) — **the critical result**
  - [ ] Find candidate novel features (both nulls, CLT excluded from matching)
  - [ ] Filter by seed/dataset consistency
  - [ ] Choose + describe specific candidate(s)
  - [ ] Ablate (re-solve LASSO, subtract δ_ℓ)
  - [ ] Matched random-feature ablation (N_rand = 10)
  - [ ] Symmetric activation test
  - [ ] All four pre-registered thresholds hold? → positive / null
- [ ] **Step 4** — [Paper structure](docs/implementation-plan/phase-4-evaluation.md#step-4--paper-structure) (draft all sections)
- [ ] **Step 5** — [Code release](docs/implementation-plan/phase-4-evaluation.md#step-5--code-release) (`scripts/reproduce.sh` works on fresh install)
- [ ] **Step 6** — [What "done" looks like](docs/implementation-plan/phase-4-evaluation.md#step-6--what-done-looks-like)

### ◆ Final deliverables
- [ ] Paper draft circulated for review ahead of release
- [ ] All figures regenerate from saved intermediates via `scripts/reproduce.sh`
- [ ] Public code release (MIT / Apache-2.0) + model card
- [ ] Short retrospective (internal) — seeds the next iteration

---

## Cross-cutting reminders

- [x] **Null tests are mandatory, not decorative** — run scrambled + context-perturbation before any Phase 1 claim. (Both run at Step 5; H1 survives both.)
- [ ] Every run logs config + git commit + seed to `runs/<date>-<desc>/`; notebooks are never source of truth.
- [ ] Share each gate memo with mentors **at** the gate, not all at the end.

---

## Log

| Date | Note |
|---|---|
| 2026-06-20 | Tracker created. |
| 2026-06-20 | Phase 1 Step 0 done — `pyproject.toml` (pinned), `src/layer_dynamics/` skeleton, run-logging backbone (config + git commit + seed → `runs/<date>-<desc>/`). Smoke run + 6 unit tests green on a bare env. |
| 2026-06-20 | Retargeted to dev box (GTX 1050 / 3 GB / Pascal sm_61). uv + Python 3.12 venv; torch 2.5.1+cu124 (GPU verified), transformers `<5`. **Pythia-410m stays primary** (fits in fp32, peak ~2 GB @ batch 4); **Pythia-70m** added as fast dev lane; collection path + Step 2 sanity check (ln_final+unembed of v_L) verified on both. Gemma/Llama cross-model (Ph2 S5, Ph4) deferred to cloud. |
| 2026-06-20 | Phase 1 Step 1 — dataset built (`scripts/build_dataset.py`, `src/layer_dynamics/data/`). CounterFact via `NeelNanda/counterfact-tracing`. Finding: 410m confidently knows only ~2000/21601 facts → tuned threshold 0.3→0.1, got **1758 examples / 3 classes**. Scored candidates cached for instant re-filtering. 10 tests green. **Open: scale dataset before Phase 2.** |
| 2026-06-20 | Decision: proceed Phase 1 on 410m (get Gate-1 signal first; scale dataset only if warranted). Step 2 — collected `residual_stream`, Step 2 sanity 5/5. |
| 2026-06-21 | Switched primary to **`pythia-410m-deduped`** (public Tuned Lens exists only for deduped; SAE coverage a wash). Re-ran Steps 1–2 → 1555 ex / 3 classes (floor 300). Step 3 done (`scripts/step3_ridge.py`, `src/layer_dynamics/diagnostics/`): ridge R²=0.602 ≫ mean/zero; **TL gate PASS** (0.602 vs −0.691). 16 tests green. Next: Step 4 CCA, Step 5 AR/Granger. |
| 2026-06-21 | Step 4 CCA done (`scripts/step4_cca.py`, `diagnostics/cca.py`): shrinkage CCA, held-out canonical corr on the same split as Step 3. Result: held-out top-1 ≈ **0.9997** at every layer, robust to γ→0.99, **scrambled null ≈ 0** (real−null 1.006) → **shared subspace = yes** (Gate-1 Q3). Honest caveat logged: ≈1 is mostly the −v_ℓ ⊂ Δ_ℓ tautology + rogue dims, not structure ridge missed. 20 tests green. Gate-1 status so far: Q1 yes, Q2 yes (TL), Q3 yes. Next: Step 5 AR(k)+Granger (both nulls). |
| 2026-06-22 | Step 5 done (`diagnostics/autoregress.py`, `scripts/step5_ar.py`, `scripts/step5_context_null.py`; `activations.collect_trajectories_from_ids`). Pooled AR(k) with per-depth intercept μ_ℓ (within-transform), GCV-α, low-rank coef; Granger = Wilks Λ + Rao F nested-F on top-32 PCA. **AR R²=0.65** (k=20) ≫ scrambled null ≈0; **Granger 112/112** Bonferroni-sig (to lag 20) vs **0/112** null. **Null 2 (context perturbation, token-level, alignment 1000/1000):** AR R² unchanged (0.639→0.696), Granger heatmap corr **0.964** → **depth-intrinsic**. Predictors compressed to top-256 Δ PCs (full-D target) to keep the k·D SVD tractable on the dev GPU. 7 new tests; **27 green**. **Gate 1: all six Q yes → PROCEED TO PHASE 2.** Memo + figures/tables in `runs/gate1/`. |
| 2026-06-22 | **Phase 2 Step 1 — stationarity test** (`scripts/step1_stationarity.py`, `diagnostics/stationarity.py`; extended `autoregress.fit_ar_pooled` with band-restricted depths + returned kernel, `eval_kernel_on_band`). Band-shared AR(k=5) kernels on early/mid/late thirds, shared input PCA basis; Frobenius + principal angles + 3×3 cross-band R² + within-band bootstrap + a rank-robustness sweep. **H2 FAILS decisively:** within-band R² positive (0.74/0.58/0.67), **all cross-band R² negative** at every rank 32→256; between-band ‖ΔW‖≈√2, angles 57–62°, bootstrap 80–104 SD separated, no clean phase boundary → **Branch B (state-space/depth-indexed)**. Pre-registered "H2 fails" route (proposal §4.2); does *not* contradict Gate 1 (pooled R² ≠ stationarity). k capped → short-horizon only. 18 new tests; **44 green**, ruff clean. Run: `runs/2026-06-22-ph2-step1-410md/`. Next: Step 2 (k\* by depth) on the depth-indexed branch. |
| 2026-06-22 | **Phase 2 Step 2 — effective kernel depth k\*** (`scripts/step2_kstar.py`, `diagnostics/effective_depth.py`). Per-depth single-target AR(k) sweep (k≤ℓ) at the Step-1 working rank r=256; scrambled null + incremental-over-lag-1 + Granger-reach contrast. **Reconstruction k\*≈1 at every depth** (median 1, max 2; bands 1/1/1): lag-1 captures ~all predictable Δ_ℓ variance (best R² 0.83–0.94), lags≥2 add median +0.018. Scrambled null → 0.000 (real structure). **Headline: reconstruction k\*≈1 vs Granger reach lag 20** — long-range structure is statistically real but reconstruction-negligible. k\*≈1 is the ±v_ℓ-tautology-dominated AR-on-Δ read (not genuine 1-layer dependence) AND consistent with a first-order Markov-in-depth SSM for Branch B. 7 new tests; **50 green**, ruff clean. Run: `runs/2026-06-22-ph2-step2-410md/`. |
