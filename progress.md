# Progress Tracker

Control panel for [*Structured Cross-Layer Propagation of Features in Transformers*](docs/README.md).
The phase docs under [docs/implementation-plan/](docs/implementation-plan/README.md) are the **pre-registration** and stay frozen — this file is the only place progress is recorded, so updating it never touches a pre-registered bar.

**Legend:** `[ ]` todo · `[~] ` in progress · `[x]` done · `[-]` skipped / N-A · `[!]` blocked

**Current focus:** _Phase 3 — Branch B SSM **complete**. **410m primary Gate-3: GO with a recorded seed-stability limitation.** The SAE-source blocker is **resolved** — the 410m-deduped audit was wrong: a complete public per-layer **residual** suite exists (MLSAE `tim-lawson/sae-pythia-410m-deduped-x64-k32-layers-{0..23}`, M=65536, used via a minimal loader + per-layer alive-compression to M\*=4433). On the 24-layer primary the SSM reconstructs Δ **above** the matched per-layer baseline at every sparsity (R² gap **+0.037→+0.066**, **3–6× the 70m's +0.008→+0.018**) and is seed-robust → the §4.2 "cross-layer propagation is load-bearing at depth" claim holds with margin. **But** decoder-direction seed-stability is **0.45 < 0.7** (structural state/feedthrough equivalent-minima degeneracy, **unmoved by 100× λ_reg**) → recorded as a limitation per the plan; features are dataset-stable (Jaccard 0.89) and the seed-stable ~45% subset feeds Phase 4. **Next: Phase 4** (baselines incl. CLT; causal existence proof on the seed-stable subset)._
**Start date:** 2026-06-20

---

## Dashboard

| Phase | Doc | Gate | Status |
|---|---|---|---|
| 1 · Diagnostic | [phase-1](docs/implementation-plan/phase-1-diagnostic.md) | **Gate 1** — is H1 supported above null? | `[x]` **PASS** |
| 2 · Characterization | [phase-2](docs/implementation-plan/phase-2-characterization.md) | **Gate 2** — does H2 (stationarity) hold? → picks Phase 3 branch | `[~]` **H2 NO → Branch B** (S1–3 done; S4–5 blocked/deferred) |
| 3 · Decomposition | [phase-3](docs/implementation-plan/phase-3-decomposition.md) | **Gate 3** — reconstruction competitive with baselines? | `[x]` **GO (410m primary, w/ recorded limitation)** — SSM beats per-layer baseline +0.037→+0.066 (3–6× the 70m) + seed-robust recon; seed-stability of decoder dirs 0.45<0.7 (structural) recorded as limitation |
| 4 · Evaluation | [phase-4](docs/implementation-plan/phase-4-evaluation.md) | — deliverable: paper + code | `[~]` next |

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

---

## Phase 2 — Characterization

> Goal: what *kind* of structure? Stationary? How deep? Generalises across relations / models?

- [x] **Step 1** — [Stationarity test](docs/implementation-plan/phase-2-characterization.md#step-1--stationarity-test) (`scripts/step1_stationarity.py`, `diagnostics/stationarity.py`) — **H2 (stationarity) FAILS, decisively & rank-robustly.** `k=5` (Granger sig to lag 20 but capped at ⌊L/3⌋−3=5 → **short-horizon only**), `r=256` (Step-3 saturation 128; spot-check promoted 128→256, pooled AR R² 0.467→0.591). Early/mid/late band kernels: within-band R² **0.74 / 0.58 / 0.67** (positive) but **every cross-band R² is negative** (early→mid −1.00, late→mid −1.89, etc.) — kernels don't transfer at all. Holds at every rank 32→256 (worst off-diag −0.32 even at r=32). Between-band ‖ΔW‖ ≈ **1.39–1.42** (≈ √2 orthogonal ceiling), principal angles **57–62°** mean. Bootstrap (n=200/band): between-band distance **80–104 SD** above within-band spread, all 3 pairs separated. No clean phase boundary (3 pair-distances ≈ equal, ratio 1.03) → **not segmented**; depth-dependence is pervasive → **Branch B (state-space / depth-indexed)**, likely *fully* depth-indexed not slowly-varying. Run: `runs/2026-06-22-ph2-step1-410md/`.
  > **Reconciles with Gate 1, not contradicts it:** Gate 1's high *pooled* AR R² (one compromise kernel + per-depth μ_ℓ) was explicitly only "consistent with" stationarity and deferred the direct test here; a generalist pooled kernel can score well while band-specific kernels fail to transfer. This is the pre-registered **"H2 fails"** outcome (proposal §4.2): H1 still holds, the dynamics are genuinely non-stationary, and CLT-style unconstrained per-(source,target) parameterisation is *validated* — "shared kernel along depth" is the wrong abstraction.
  > **Caveats carried forward:** (1) k capped → short-horizon only; long-horizon stationarity deferred to Phase 3 Step 4 MA recheck. (2) AR-on-Δ ≠ MA stationarity (proxy). (3) the ±v_ℓ shared term biases *toward* stationarity, so failing is the conservative direction. (4) 3 classes/1555 ex — per-class test is Step 4.
- [x] **Step 2** — [Effective kernel depth k\*](docs/implementation-plan/phase-2-characterization.md#step-2--effective-kernel-depth) (`scripts/step2_kstar.py`, `diagnostics/effective_depth.py`) — **reconstruction k\* ≈ 1 at every depth** (per-depth k\* median 1, max 2 (only ℓ=17); band k\* = 1/1/1). A single preceding update Δ_{ℓ-1} captures essentially all held-out predictable variance (per-depth best R² 0.83–0.94, dipping to 0.64 at ℓ17); **lags ≥ 2 add a median +0.018 R² (max +0.05)**. Scrambled null collapses the whole curve (median best R² **0.000**) → real example-specific structure. **Headline gap:** reconstruction k\*≈1 vs **Granger reach lag 20** (Step 5) — long-range depth structure is *statistically real but reconstruction-negligible*. Run: `runs/2026-06-22-ph2-step2-410md/`.
  > **Tautology caveat (per `cca-vdelta` memory):** k\*≈1 is substantially the mechanical Δ_ℓ ⊃ −v_ℓ ⊂ Δ_{ℓ-1} overlap, so it is a property of the AR-on-Δ parameterisation, **not** proof the transformer's genuine computational dependence is one layer deep. k\*≈1 *is* consistent with Branch B being a **first-order (Markov-in-depth) SSM** — good news for Phase 3 model order.
- [x] **Step 3** — [Token-position invariance](docs/implementation-plan/phase-2-characterization.md#step-3--token-position-invariance) (`scripts/collect_multipos.py`, `scripts/step3_tokenpos.py`, `diagnostics/token_position.py`, `token_positions.py`) — **NOT invariant: query-position-specific (decisive).** Re-collected residual streams at 4 token positions (first/last subject token, mid non-subject token, final prompt token; subject span via tokenizer offset mapping; final-position matches Phase-1 store exactly max|Δ|=0). AR(k\*=1, r=256) refit per position; **1523 multi-token** (4×4) + **32 single-token** (3×3) subjects. Within-position held-out R² positive (subj_first 0.52 / subj_last 0.28 / mid 0.47 / final 0.55) but **every cross-position R² negative** (mean −0.36; subj_first→final −1.18; worst retention −2.9). Kernels near-orthogonal across positions (Frobenius ≈1.25–1.46 ≈√2, mean angles 57–60°) — same "no-transfer" signature as Step-1 cross-band. Scrambled null collapses diagonals to ≈0 (−0.001) → real structure; tautology cancels in the retention ratio. **Implication:** scope narrows to *context-aggregated factual recall at the query position* (Phase 3 fits on query-position trajectories; Phase 4 states the scope). Does **not** flip Branch B (architecture derives from Step 1). Run: `runs/2026-06-23-ph2-step3-410md/`.
- [!] **Step 4 (blocked)** — [Generalisation across relation types](docs/implementation-plan/phase-2-characterization.md#step-4--generalisation-across-relation-types) — **needs dataset scaled to ≥1000/class first** (currently 3 classes / 1555 ex; see `counterfact-410m-yield`)
- [~] **Step 5** — [Cross-model replication](docs/implementation-plan/phase-2-characterization.md#step-5--cross-model-replication) — **box-feasible targets identified** (Gemma-2-2B/Llama-3.1-8B still don't fit 3 GB → cloud). Two models fit the GTX 1050 in fp32 **and** have public per-layer **residual** SAEs, so the full recheck (H1 diagnostics → H2 stationarity → Phase-3 SSM gap) is runnable on-box. **Goal:** turn the 2-point depth-scaling result (6L 70m → 24L 410m, gap +0.018→+0.066) into a **monotone trend** and add a **cross-family** replication. SAE availability verified 2026-06-25. Run order (GPT-2 first — it needs new code):
  - [~] **(a) GPT-2-small** (124M, L=12, d_model=768) — **cross-family** check (different arch/data/tokenizer than Pythia; the real generality test Gemma/Llama couldn't deliver on-box). SAEs: Joseph Bloom **`jbloom/GPT2-Small-SAEs-Reformatted`** (`res-jb`: 12 resid_pre L1 SAEs, d_sae=24576). **CODE DONE:** new `src/layer_dynamics/saes_resjb.py` (ReLU/L1, `apply_b_dec_to_input=True`, no TopK/standardize; **resid_pre ⇒ v_0 HAS an SAE, no zero-init layer**), 7 tests, ruff clean; **state-dict keys + shapes verified against the real blocks.0 repo** (`W_enc[768,24576]`, `W_dec[24576,768]`, `b_enc`, `b_dec`); `build_410m_sources.py --suite resjb`; `configs/gpt2-small.yaml` (tuned_lens omitted). **Remaining:** full-pipeline run (the build-sources step has a built-in recon-R²>0.4 gate = the real on-data smoke-test).
  - [~] **(b) Pythia-160m-deduped** (L=12, d_model=768, M=768×64=49152, k=32) — **in-family depth midpoint** (6L→**12L**→24L). SAEs: same MLSAE suite, `tim-lawson/sae-pythia-160m-deduped-x64-k32-layers-{0..11}` → **drop-in (config-only, no new code)**; same `v_0` = zero-init `u_0` caveat as 410m. `configs/pythia-160m-deduped.yaml` ready. Reliable overnight workhorse.
  - **Runner:** `scripts/run_crossmodel.sh [gpt2|p160|both]` chains the full pipeline per model (dataset→collect→step3→step5→stationarity→k*→sources→gate3 pareto→step4 sanity); robust for unattended use (run dirs resolved by newest-desc glob across midnight; per-model isolation; Phase-3 gated on the source recon sanity). Logs to `runs/crossmodel-overnight.log`.
  - **Caveat:** factual-recall yield shrinks with model size (70m ≈ none @ prob>0.3; 160m / GPT-2-small both < 410m's ~2000) → smaller datasets; fine for H1/H2 diagnostics (held at n=1555 on 410m) but tighter for per-class tests.
- [~] **Step 6** — [Gate 2 decision](docs/implementation-plan/phase-2-characterization.md#step-6--gate-2-decision) memo — **architecture LOCKED (Branch B) from Steps 1–2**; Q3 answered (query-position-specific); Q4–5 outstanding (`runs/gate2/memo.md`)

### ◆ Gate 2 — decision record
Record: stationary? **NO** (Step 1) · k\* = **≈1** (Step 2; vs Granger reach 20) · token-position-invariant? **NO — query-position-specific** (Step 3; cross-position R² all negative, mean −0.36, kernels ≈orthogonal) · circuit-general or -specific? **blocked** (Step 4, dataset <1000/class) · cross-model replicates? **deferred** (Step 5, Gemma/Llama don't fit 3 GB)

Architecture decision — **tick one** (copy into `runs/gate2/phase3_architecture.md`):
- [ ] **Branch A** — stationarity holds → block-Toeplitz convolutive kernel
- [x] **Branch B** — stationarity fails → state-space model (**fully depth-indexed**, not smoothed — drift is pervasive not gradual; **first-order**, k\*≈1)
- [ ] **Branch C** — clean phase transition → one kernel per segment (boundaries: _____)

Date: **2026-06-22** · Memo: `runs/gate2/memo.md` (architecture-locking; Q3–5 are scope/generality cross-checks that cannot flip the Step-1 branch)
- [x] `runs/gate2/figures/` (stationarity cross-band/bootstrap/angles, k\* curves, Step-3 cross-position confusions; cross-class & cross-model N/A — Steps 4–5 outstanding)
- [x] `runs/gate2/phase3_architecture.md` (one-paragraph exact decomposition spec)

---

## Phase 3 — Decomposition

> Goal: fit the generative model — sparse sources + depth kernel (or SSM). The build step.

- [x] **Step 1** — [Lock architecture from Gate 2](docs/implementation-plan/phase-3-decomposition.md#step-1--lock-the-architecture-based-on-gate-2) — locked spec is `runs/gate2/phase3_architecture.md` (Branch B: first-order, fully depth-indexed SSM; z_{ℓ+1}=A_ℓz_ℓ+B_ℓu_ℓ, Δ_ℓ≈b_ℓ+C_ℓz_ℓ+D_ℓu_ℓ; low-rank factored A/B/C/D swept r∈{16,32,64,128,(256)}; d_z~r; debug 70m, primary 410m). Implementation-plan docs are frozen pre-registration → lock recorded here, not in them.
- [~] **Step 2** — [Define sparse sources {s_ℓ'}](docs/implementation-plan/phase-3-decomposition.md#step-2--define-sparse-source-representations-lbrace-s_ellrbrace) — **SAE-input audit done (2026-06-23):** Branch B permits **per-layer width M_ℓ** (no uniform-M constraint, unlike Branch A). **Debug lane (70m-deduped): public per-layer residual SAEs exist** → `EleutherAI/sae-pythia-70m-deduped-32k` (`layers.0..5` = resid_post, d_in=512, M=32768, TopK k=16; loader: EleutherAI `sparsify`). **Primary lane (410m-deduped): RESOLVED 2026-06-25.** The EleutherAI audit was incomplete — `sae-pythia-410m-65k` is indeed non-deduped + MLP-only, but a **complete public per-layer residual suite exists from the MLSAE work** (`tim-lawson/sae-pythia-410m-deduped-x64-k32-layers-{0..23}`, Lawson et al. 2024 — already cited in the proposal; M=65536, TopK k=32, all 24 layers). Loaded via a minimal `saes_mlsae.py` (per-token standardize, tied bias, ReLU-after-TopK; `layers-N`=v_{N+1} same as the 70m suite). Dense `u` is ~10 GB → per-layer **alive-compression** to M\*=4433 (`build_410m_sources.py`, cached npz; Branch B permits per-layer M_ℓ). Matched recon R² mean 0.598. **No SAE training needed.**
  - [x] **SAE source path verified on 70m-deduped** (`src/layer_dynamics/saes.py` minimal TopK loader — no `sparsify` dep; `scripts/verify_saes.py`): matched-state reconstruction R² **mean 0.685** (v_1 0.87 / v_2 0.72 / v_3 0.64 / v_4 0.62 / v_5 0.58), TopK k=16/32768 — healthy, loader convention (decoder-bias-subtract + relu + TopK) correct. **v_0 has no embedding SAE** (proxy R²=−45.7) → source builder will **zero-init u_0** (free LASSO var), not the proxy. 70m-deduped debug dataset = top-1000 candidates by confidence (`data/pythia-70m-deduped/`, debug-only; 70m-deduped knows ~no facts at prob>0.3), residual streams collected (`[1000,7,512]`, sanity 5/5).
- [~] **Step 3** — [Loss + optimisation](docs/implementation-plan/phase-3-decomposition.md#step-3--loss-and-optimisation-branch-a--c) (alternating min; λ_L1 sweep; rank sweep; K = k\*)
  - [x] **Milestone 2: SSM model + loss** (`src/layer_dynamics/ssm.py`) — `LowRankSSM`: per-layer low-rank factored A/B (ℓ=0..L-2) + C/D/b (ℓ=0..L-1), z_0 free/zero; forward unrolls z via rank-r matvecs (D×M ops never materialised); `ssm_loss` = recon SSE + λ_L1·‖u‖₁ + λ_reg·‖operators‖_F² (exact low-rank product Frobenius); source builder `saes.build_sources` (zero-inits u_0). 7 tests (factored-vs-full forward match, loss components, grad flow, z_0 flag); **75 green**, ruff clean.
  - [x] **Milestone 3: alternating optimiser** (`src/layer_dynamics/decompose.py`, `scripts/step3_decompose.py`; `operator_norms()` added to `ssm.py` for the collapse diagnostic) — FISTA-with-backtracking input update (proximal-gradient LASSO → *exact* zeros, validated against `sklearn.Lasso` **and** closed-form least-squares) alternated with full-batch Adam over the operators (reg added once; L1 constant in that phase); held-out `infer_inputs` re-infers `u` with operators **frozen** (the honest generalisation encoder). **Debug pass on 70m-deduped** (r=32, d_z=32, λ_L1=1.0, λ_reg=1e-4, 10 outer iters, GPU ~37 s/iter): **train R²=0.876 / held-out test R²=0.858** (gap **+0.019** → operators generalise, not memorising) at **L0≈0.001** (from SAE-init 0.0004), loss 6.84e6→1.24e5, **no collapse-to-bias** (|D|=172 ≫ |b|=4.7). L0 spikes at iter-1 (FISTA grabs reconstruction from the ultra-sparse SAE init) then re-sparsifies as the LASSO bites. 10 new tests; **85 green**, ruff clean. Run: `runs/2026-06-24-ph3-decomp-70md/`.
  - [x] **Milestone 4: Step-4 sanity checks** (`src/layer_dynamics/diagnostics/decompose_sanity.py`, `scripts/step4_sanity.py`) — **all five Branch-B checks pass on 70m-deduped** (r=32, λ_L1=1.0, 10 outer). 14 new tests; **99 green**, ruff clean. Run: `runs/2026-06-24-ph3-step4-70md/`. Memory note: 5 sequential fits OOM the 3 GB GTX 1050 unless GPU is freed between fits (`_free_cuda` + `expandable_segments`) — fixed.
- [x] **Step 4** — [Sanity checks before Gate 3](docs/implementation-plan/phase-3-decomposition.md#step-4--sanity-checks-before-gate-3) — **all mandatory; no red flags**
  - [x] 1 · trivial-solution / collapse check — **no collapse** (‖ops‖=173 ≫ ‖b‖=4.7, L0≈0.001 ≠ 0)
  - [x] 2 · reconstruction eyeball (10 examples) — **median per-example R²=1.00** (0.989–0.997)
  - [x] 3 · feature activations have expected structure (top-20) — **mean relation-class purity 0.93** (clean themes: feat 8180→religion, 27754→language/French, 4110→geography)
  - [x] 4 · seed stability (3 seeds) — **match-rate@0.7 = 0.99** (mean cos 0.92; all 3 seeds → train R²≈0.876) **≫ 0.7 Gate-3 threshold**. *Metric note:* dictionary is the **fixed pretrained SAE**, so features are the same atoms across re-fits → identity-matched decoder-direction (`D_ℓ[:,j]`) cosine, the fixed-dictionary specialisation of the plan's "max cosine" (no permutation to search; a 32k-direction max-cosine search in 512-d would match spuriously).
  - [x] 5 · dataset stability (disjoint halves) — **top-100 Jaccard 0.89** (94/100 shared), shared-dir match@0.7 0.84 (mean cos 0.83)
  - [-] 6 · MA-stationarity recheck — **N/A (Branch A only)**
- [x] **Step 5** — [Gate 3 decision](docs/implementation-plan/phase-3-decomposition.md#step-5--gate-3-decision) memo (`runs/gate3/memo.md`) — **GO on the 410m primary lane, with a recorded seed-stability limitation** (reconstruction clause met with margin +0.037→+0.066; seed-stability clause 0.45<0.7 recorded as a structural limitation). 70m debug lane (`runs/2026-06-24-ph3-gate3-70md/`) was the prior method-validating Go. Primary Pareto: `scripts/gate3_pareto.py`, run `runs/2026-06-25-ph3-gate3-410md/`.

### ◆ Gate 3 — decision record
**Go criterion:** reconstruction on/above per-layer SAE Pareto at comparable sparsity **and** seed stability ≥ 0.7 (top-100 feature match).

Tick one (410m primary; the two clauses split):
- [x] **Go → Phase 4 — with a recorded seed-stability limitation** — **(410m primary lane)** reconstruction clause **MET with margin**: SSM held-out R² above the matched per-layer baseline at every L0 (gap **+0.037→+0.066**, 3–6× the 70m's +0.008→+0.018), seed-**robust** (every seed → R²=0.880, same gap; |C| grows 28→50 under sparsity). The §4.2 "cross-layer propagation is load-bearing at depth" claim holds.
- [x] **No-go — unstable → recorded as a limitation (not a hard stop)** — seed-stability clause **NOT met**: decoder-direction match@0.7 = **0.45 < 0.7**, **structural** (state/feedthrough equivalent minima; all seeds same R², different directions) and **unmoved by 100× λ_reg** (1e-4→1e-2). Plan's stronger-reg remedy exhausted; dictionary-alignment N/A (fixed SAE dictionary). Features are dataset-stable (Jaccard 0.89); Phase 4 uses the seed-stable ~45% subset.
- [ ] **No-go — under-expressive** → N/A (reconstruction beats baseline with margin)
- [ ] **No-go — both fail** → N/A (reconstruction does not fail)

Recon R²: **SSM test 0.70–0.87** vs baseline 0.65–0.82 across λ_L1∈{0.3…30}, matched-L0 gap +0.037→+0.066 · seed stability **0.45** (recorded limitation) · dataset Jaccard **0.89** · feature purity **0.83**, eyeball median R² **0.98** · Date: **2026-06-25** · Memo: `runs/gate3/memo.md`
- [x] `runs/gate3/figures/` (Pareto vs per-layer baseline `pareto.png`; reconstruction eyeball `reconstruction.png`) — under `runs/2026-06-25-ph3-gate3-410md/` and `…-step4-410md/`
- [-] `runs/gate3/checkpoints/` — fits ~16 min each to regenerate; populated by Phase-4 ablation work

> **Gate-3 outcome.** The **24-layer 410m primary** establishes the scientific claim the 70m debug lane could not: structured cross-layer propagation reconstructs Δ **materially** better than independent per-layer SAEs (gap 3–6× the shallow-model gap, seed-robust). The wrinkle that only appears at depth is a **structural seed-instability of the learned decoder directions** (state/feedthrough trade-off; unfixed by stronger reg) — recorded as a limitation per the plan, scoped out of the headline claim, and handled in Phase 4 by filtering causal candidates to the seed-stable subset. Branch B is larger than a stationary kernel by design (H2 failed) → a reconstruction-at-matched-sparsity win, **not** a parsimony win (report param counts vs CLT in Phase 4).

### ◆ Gate-3 limitation — seed-stability remediation (candidate fixes, to try one-by-one)

The recorded limitation (decoder-direction match@0.7 = **0.45 < 0.7**, unmoved by 100× λ_reg) is a
gauge/equivalent-minima degeneracy: many distinct operator sets give the **same** reconstruction
(R²=0.880 every seed), so different seeds land at different decoder directions. Reconstruction is
gauge-invariant; the raw decoder vectors are not. It has **two distinct sources**, and each fix below
targets one — so they are tried in order of principle, cheapest-decisive first.

- **Symmetry A — similarity-transform (gauge) freedom.** `(A,B,C) → (T A T⁻¹, T B, C T⁻¹)` for any
  invertible `T` leaves the input→output map identical and rotates `C`'s columns — but **`D` is
  invariant under `T`**. So Symmetry A does **not** affect the *currently measured* stability metric
  (`decoder_directions` reads `D_ℓ[:,j]`, the feedthrough — `decompose_sanity.py`); it only
  destabilises `C`, which we do not yet score.
- **Symmetry B — state↔feedthrough trade-off.** Part of Δ_ℓ is explainable by either `C_ℓ z_ℓ`
  (propagated state) or `D_ℓ u_ℓ` (local feedthrough); mass shifts between them, moving `D`. **This is
  the source of the reported 0.45** (the metric is on `D`). *It is also why the 100× λ_reg retune did
  nothing: an isotropic penalty scales **along** the orbit, so it never breaks the symmetry — the fix
  must be **transverse**.*

**Execution order (corrected 2026-06-25):** the metric is on `D`, so lead with the **Symmetry-B**
levers (Fix 3 → Fix 2); Fix 1 (Symmetry A) only matters if we **also** add a `C`-stability metric (the
cross-layer-propagation operator is the scientific object, so this is worth doing alongside Fix 1).

Tries (tick when run; record R²-impact + new match@0.7 next to each):
- [x] **Fix 3 — shrink `d_z` (capacity sweep).** *Tried d_z=8 (4× cut), `runs/2026-06-25-ph3-step4-410md-dz8/`.*
  **NULL: match@0.7 0.45 → 0.46 (unchanged) while train R² 0.892 → 0.864.** Capacity is **not** the
  lever — a smaller state still admits a non-unique state↔feedthrough split, so reducing `d_z` only
  costs reconstruction without breaking Symmetry B. (Smaller d_z would only hurt R² more → not pursued.)
  **Lesson: we must *tilt the selection* (asymmetric penalty), not shrink it → Fix 2.**
- [ ] **Fix 2 — transverse / asymmetric regulariser.** Replace the isotropic `λ_reg·‖ops‖_F²` with a
  penalty whose level sets cut across the trade-off — e.g. ridge on **`C`/state only** (selects the
  unique minimum-state-energy split, forcing `D` to take the residual deterministically) or a
  nuclear-norm allocation preference. Kills **Symmetry B**; the correct version of the failed retune.
  Cheap (small `ssm_loss` change). Result: ____
- [ ] **Fix 1 — diagonalize/canonicalize the state (modal SSM form).** Constrain `A_ℓ` to diagonal
  (S4D/S5/Mamba style), removing the continuous `T` freedom; leftover perm+scale is discrete (sort
  eigenvalues, normalize `B` rows). Kills **Symmetry A** → stabilises `C` (not `D`). *Pair with a new
  `C`-direction stability metric, else it won't move the reported number.* Cost: complex-eigenvalue
  handling in the unroll. Result: ____
- [ ] **Fix 4 — tie `C` to the fixed SAE dictionary.** Constrain decoder directions to the fixed,
  already-stable atoms (Jaccard 0.89) — or sparse combinations of them — so the readout can't drift.
  Kills A and B; most on-theme (uses the one stable object to pin the unstable one). Cost: constrains
  expressivity. Result: ____
- [ ] **Fix 5 — report gauge invariants instead (reframing, not a method fix).** Measure stability of
  gauge-*invariant* quantities — top-k decoder **subspace**, eigenvalues of `A`, or each feature's
  data-averaged Δ-contribution — rather than the raw vectors. May show the instability is cosmetic.
  Result: ____
- [ ] ~~Deterministic data-driven init alone~~ — **excluded as a standalone fix**: pins the basin so
  every seed *looks* stable but the equal-loss valley is still there (hides, doesn't remove). Only
  valid as a complement to Fix 1/2/4.

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
| 2026-06-23 | **Phase 3 scoping (Steps 1–2)** — Step 1: architecture locked = Branch B (spec in `runs/gate2/phase3_architecture.md`; frozen docs untouched). Step 2 **SAE-input audit**: Branch B allows per-layer width M_ℓ. **70m-deduped (debug) has public per-layer residual SAEs** (`EleutherAI/sae-pythia-70m-deduped-32k`: `layers.0..5`=resid_post, d_in=512, M=32768, TopK k=16; plain safetensors enc/dec → minimal loader, no `sparsify` dep). **410m-deduped (primary) has NONE** (`sae-pythia-410m-65k` is non-deduped + MLP-only; `*-deduped-*` 404) — explains the Gate-1 "deduped = SAE wash" note (deduped bought Tuned Lens, cost the 410m SAE suite). **Decision: defer 410m sourcing; build + debug Branch B on 70m-deduped with public SAEs first** (front-load the method, decide 410m train-vs-substitute with reconstruction evidence). Indexing gotcha noted: SAE `layers.N`=v_{N+1}, so sources s_0..s_{L-1}=v_0..v_{L-1} have SAEs for v_1..v_{L-1} but **not v_0** (embedding) — handle in the source builder. |
| 2026-06-23 | **Phase 2 Step 3 — token-position invariance** (`scripts/collect_multipos.py`, `scripts/step3_tokenpos.py`, `diagnostics/token_position.py`, `token_positions.py`). Re-collected residual streams at 4 token positions/example (first/last subject token via tokenizer offset-mapping span, mid non-subject token, final prompt token) → `residual_stream_multipos.h5[1555,4,25,1024]`; final-position matches Phase-1 store exactly (max|Δ|=0), top-1 5/5. Refit AR(k\*=1,r=256) per position, shared input PCA basis; **1523 multi-token (4×4) + 32 single-token (3×3)** subjects. **Result: query-position-specific, decisively** — within-position R² positive (0.28–0.55) but **every cross-position R² negative** (mean −0.36, worst retention −2.9), kernels ≈orthogonal (Frobenius ≈√2, angles 57–60°); scrambled null collapses diagonals to ≈0. Scope narrows to context-aggregated recall at the query position; Branch B unaffected. 12 new tests; **62 green**, ruff clean. Run: `runs/2026-06-23-ph2-step3-410md/`; gate2 memo Q3 + figures updated. |
| 2026-06-24 | **Phase 3 Step 3 Milestone 3 — alternating optimiser** (`src/layer_dynamics/decompose.py`, `scripts/step3_decompose.py`; `ssm.operator_norms()`). Branch B fit by alternating **full-batch Adam over operators** (inputs fixed; reg added once) with a **FISTA-with-backtracking input update** (proximal-gradient LASSO over the stacked per-example inputs, operators fixed) — proximal grad keeps the L1's *exact zeros*, which a plain Adam+L1 would not. ∇_u recon via autodiff through the SSM unroll (depth coupling handled exactly); step size by Beck–Teboulle backtracking on a shared (warm-started) Lipschitz estimate. Held-out `infer_inputs` re-infers `u` with operators frozen = honest generalisation. **Debug pass on 70m-deduped** (r=d_z=32, λ_L1=1.0, λ_reg=1e-4, 10 outer, GPU): **train R²=0.876 / test R²=0.858** (gap +0.019), **L0≈0.001** (SAE-init 0.0004), loss 6.84e6→1.24e5, **no collapse** (|D|=172 ≫ |b|=4.7). FISTA correctness unit-tested vs `sklearn.Lasso` + closed-form LS; sparsity (exact zeros) and operator-freeze tested. 10 new tests; **85 green**, ruff clean. Run: `runs/2026-06-24-ph3-decomp-70md/`. Next: Milestone 4 sanity checks (collapse/eyeball/feature-structure/seed/dataset), then Gate-3 Pareto vs per-layer SAE. |
| 2026-06-24 | **Phase 3 Step 3 Milestone 4 — Step-4 sanity checks** (`src/layer_dynamics/diagnostics/decompose_sanity.py`, `scripts/step4_sanity.py`). All five Branch-B-applicable checks (#6 is Branch A only) run on 70m-deduped over the M3 fit (r=32, λ_L1=1.0, 10 outer; 5 fits = main + 2 seeds + 2 halves): **(1)** no collapse (‖ops‖=173 ≫ ‖b‖=4.7, L0≈0.001≠0); **(2)** reconstruction eyeball median per-example R²=1.00; **(3)** top-20 feature relation-class **purity 0.93** with clean interpretable themes (religion / language-French / geography); **(4) seed stability match-rate@0.7 = 0.99** (3 seeds, mean cos 0.92) — **clears the ≥0.7 Gate-3 bar**; **(5)** dataset stability top-100 Jaccard 0.89 (shared-dir match 0.84). **Metric note:** the fixed pretrained SAE dictionary makes features identical atoms across re-fits, so seed/dataset stability uses identity-matched decoder-direction (`D_ℓ[:,j]`) cosine — the fixed-dictionary specialisation of the plan's "max cosine of decoder directions" (no permutation to search). **Compute note:** 5 sequential fits OOM the 3 GB GTX 1050 unless GPU is freed between fits (`_free_cuda` empty_cache + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`). 14 new tests; **99 green**, ruff clean. Run: `runs/2026-06-24-ph3-step4-70md/`. Next: Gate-3 Pareto vs per-layer SAE (+ 410m source decision). |
| 2026-06-25 | **Phase 3 Step 5 — Gate-3 reconstruction-vs-sparsity Pareto** (`scripts/gate3_pareto.py`; `LowRankSSM.use_state` flag added for the no-state baseline). λ_L1 sweep {0.1,0.3,1,3,10} on 70m-deduped (r=32, 10 outer), each point a full fit + held-out inference (operators frozen), GPU freed between fits. **Baseline = the matched per-layer sparse dictionary** (same model with the cross-layer state ablated: `Δ_ℓ≈b_ℓ+D_ℓu_ℓ`), so the SSM−baseline gap isolates exactly the value of cross-layer propagation (`C_ℓz_ℓ`). **Result: SSM on/above baseline at every L0** (held-out R² gap **+0.008→+0.018**, widening toward higher sparsity); knee λ≈0.3–1 → test R²≈0.86 @ L0≈0.002–0.005. Combined with Step-4 **seed stability 0.99** → **Gate-3 go criterion MET → GO (70m debug lane)**. **Honest caveat: small margin** (|C|≈14 vs |D|≈170 — state is a minor term on 6 layers, consistent with k\*≈1 + ±v_ℓ tautology) → Go validates the **method**; the scientific test is the 24-layer **410m primary**, blocked on the SAE-source decision (no public 410m residual SAE → train minimal per-layer SAEs or sparse-init substitute). 1 new test (`use_state` forward); ruff clean. Memo `runs/gate3/memo.md`; run `runs/2026-06-24-ph3-gate3-70md/`. |
| 2026-06-25 | **Seed-stability remediation plan added** (Gate-3 limitation §). Diagnosed the degeneracy as two symmetries — **A** the SSM similarity-transform gauge `(A,B,C)→(TAT⁻¹,TB,CT⁻¹)` (rotates decoder dirs), **B** the state↔feedthrough trade-off (why isotropic 100× λ_reg did nothing — must be transverse). **Key correction:** the reported match@0.7 is measured on the **feedthrough** `D_ℓ[:,j]` (`decompose_sanity.decoder_directions`), and Symmetry A leaves `D` invariant — so the diagonalize-A fix can't move the current number; the 0.45 is a **Symmetry-B** (state/feedthrough) effect. Corrected try-order: **Fix 3 first** (shrink d_z — zero-code, `--d-state` already wired), then Fix 2 (transverse/asymmetric reg, e.g. ridge-on-state-only), then Fix 1 (diagonalize A — only with a new `C`-stability metric), Fix 4 (tie C to fixed SAE dict), Fix 5 (report gauge invariants). Deterministic init excluded as standalone (hides not removes). **Next: run Fix 3 (d_z sweep).** |
| 2026-06-25 | **Cross-model pipeline built (overnight-ready).** New `saes_resjb.py` (GPT-2-small res-jb L1 loader; resid_pre ⇒ v_0 has an SAE, no zero-init; 7 synthetic tests) — state-dict keys/shapes verified live against `jbloom/GPT2-Small-SAEs-Reformatted` blocks.0 (`W_enc[768,24576]` etc.). `build_410m_sources.py` gains `--suite {mlsae,resjb}`. Two configs (`gpt2-small.yaml`, `pythia-160m-deduped.yaml`, tuned_lens omitted for robustness). `run_crossmodel.sh` chains the full H1→H2→Phase-3 pipeline per model, unattended-safe (newest-desc globbing, per-model isolation, Phase-3 gated on source recon sanity). Full suite **119 green**, ruff clean. **Next: when the GPU frees (Fix 2 done), launch `scripts/run_crossmodel.sh` overnight.** |
| 2026-06-25 | **Cross-model replication targets added (Phase 2 Step 5).** Verified two on-box-feasible models with public per-layer residual SAEs: **(a) GPT-2-small** (124M, L=12) — cross-family; SAEs = Bloom `gpt2-small-res-jb` (needs a new SAELens-convention loader); **(b) Pythia-160m-deduped** (L=12, M=49152) — in-family depth midpoint (6→12→24L); SAEs = same MLSAE suite `sae-pythia-160m-deduped-x64-k32-layers-{0..11}`, drop-in to `saes_mlsae.py` (config-only). Goal: monotone depth-scaling trend + cross-family check. Gemma/Llama still cloud-only. |
| 2026-06-25 | **Fix 3 (shrink d_z) — NULL.** d_z=8 (4× cut, `…-step4-410md-dz8`): seed match@0.7 **0.45→0.46** (unchanged), train R² **0.892→0.864**. Capacity is not the lever — a smaller state still admits a non-unique state↔feedthrough split. Confirms Symmetry B needs a *transverse/asymmetric* penalty, not less capacity. **Next: implement + run Fix 2** (asymmetric reg: penalize state path A,B,C vs feedthrough D separately, e.g. λ_D=0 to force a minimum-state-energy unique split). |
| 2026-06-25 | **Phase 3 — 410m PRIMARY lane: SAE-source blocker resolved + Gate-3 decided (GO w/ recorded limitation).** Re-audit overturned the "no 410m residual SAE" finding: a complete public per-layer **residual** suite exists from MLSAE (`tim-lawson/sae-pythia-410m-deduped-x64-k32-layers-{0..23}`, Lawson et al. 2024 — cited in the proposal; M=65536, k=32). New `src/layer_dynamics/saes_mlsae.py` (minimal loader for the mlsae convention — per-token standardize / tied bias / ReLU-after-TopK, verified vs library source; `layers-N`=v_{N+1}) + `scripts/build_410m_sources.py`. Dense `u` is ~10 GB → per-layer **alive-compression** (only 497–4433 of 65536 latents fire per layer in the narrow query-position distribution; Branch B permits per-layer M_ℓ) → M\*=4433, `u[1555,24,4433]`=662 MB cached; matched recon R² 0.598. Fit scripts take `--sources-npz` (70m path byte-identical). 8 new tests; **109 green**, ruff clean. Infra commit `b083d8a`. **Gate-3 Pareto (r=32, d_z=32, λ_L1∈{0.3…30}):** SSM held-out R² **above** the matched per-layer baseline (use_state=False) at every L0, matched-L0 gap **+0.037→+0.066** (3–6× the 70m's +0.008→+0.018), seed-**robust** (every seed → R²=0.880, identical gap; |C| grows 28→50 under sparsity) → reconstruction clause **MET with margin**; the §4.2 "cross-layer propagation is load-bearing at depth" claim holds. **Step-4 sanity:** no collapse, eyeball median R² 0.98, purity 0.83, dataset Jaccard 0.89 — **but seed-stability of decoder directions 0.45 < 0.7**. Diagnosed as **structural** (all seeds same R², different decoder directions = equivalent minima from the SSM's state/feedthrough trade-off) and **unmoved by 100× λ_reg** (1e-4→1e-2 retune, `…-step4-410md-reg2`). Plan's stronger-reg remedy exhausted; dictionary-alignment N/A (fixed SAE dictionary). **Decision (user-confirmed): GO with the seed-stability degeneracy recorded as a Branch-B limitation** — reconstruction (the central claim) is solid + seed-robust; the instability is confined to decoder-direction identifiability, not the dataset-stable feature set, so Phase 4 filters causal candidates to the seed-stable ~45% subset. Runs: `runs/2026-06-25-ph3-{gate3,step4,decomp,sources}-410md*/`. Memo `runs/gate3/memo.md`. **Next: Phase 4.** |
