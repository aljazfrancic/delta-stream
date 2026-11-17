# v1.1 post-hoc on-box checks A–D — the four checks the paper names and does not run

**Model:** Pythia-410m-deduped (L=24, D=1024), query-position residuals, N = 1,555
(1,244 train / 311 test, seed 0). One part of check C also uses Pythia-160m-deduped.
**Status of this memo:** design, statistics, tolerances and verdict rows **frozen
2026-11-17, before any of these runs exists**. Everything here is **post hoc** relative
to the pre-registered protocol (`docs/`, untouched) and to paper v1.0. Results will be
appended below under a dated heading; the sections above the results stay as written.

> **STATUS (2026-11-17, morning): criteria frozen, runs NOT started.** The tooling
> (`scripts/run_v11_checks.sh`, `scripts/v11_*.py`, three library modules, their tests)
> is written and unit-tested on synthetic data only. No rebuilt artefact and no number
> from the real store exists yet.

> **STATUS UPDATE (2026-11-17, evening — the note above is the morning's record, left as
> written).** The full run started 15:50 (commit `61fb2ea`) after a clean end-to-end smoke
> test. Two things went wrong in the first pass, and neither touches a statistic or a
> verdict row:
>
> 1. **Check B's recipe was written down wrong above.** The section on B says the sweep is
>    "identical to the recorded sweep (… 250 Adam epochs …)". The recorded sweep did not use
>    250 epochs: its own memo (`runs/2026-06-30-ph4-e5-prefix-410md/memo.md`) gives λ_reg 0.1
>    and "the same frozen-source Adam recipe as E3", i.e. 180 epochs, and the script's
>    defaults (250 epochs, λ_reg 1e-4) were not what ran. The paper's `tab:operating`
>    footnote carries the same error; this memo copied it. **The reproduction gate caught
>    it:** at the defaults the embedding-only fit gives test R² −0.2974 against the recorded
>    −0.4382, so the mean-initialised sweep was skipped. Refitting the embedding-only point
>    with 180 epochs and λ_reg 0.1 (CPU, scratch) gives −0.4384, i.e. the recorded value up
>    to CPU/GPU arithmetic. B is therefore run with 180 epochs and λ_reg 0.1 for both the
>    gate and the mean-initialised sweep — what "identical to the recorded sweep" requires.
>    Thresholds and rows are unchanged.
> 2. **The MLSAE checkpoint ran out of GPU memory** on its third fit (stability seed 2),
>    after the canonical fit and its codes were saved. `phase4_checkpoint.py` kept earlier
>    fits' tensors referenced; fixed in `a3ae4f9` (no computed value changes). The second
>    pass resumes from the saved canonical fit (`--resume`, exact by construction) and must
>    still pass the checkpoint gate.
>
> The first pass's other steps are unaffected. A second pass runs B, the MLSAE half of A
> and the MLSAE rank-64 interval of C.

## Why

The third whole-paper review of v1.1 (progress.md WP4c) left four statements resting on
checks the paper names in §8 and does not run. Each check below is tied to the sentence
it bears on.

| | Check | Sentence it bears on |
|---|---|---|
| A | State probe with codes inferred from earlier layers only, plus a feedthrough arm | §6.7 / abstract: "the state decodes the recalled answer at mid-depth better than dimension-matched PCA of its own codes" |
| B | Source-prefix sweep with the bias initialised at the per-depth mean | §6.9: the embedding-only −0.44 "is an optimisation artefact, not a floor"; "late-layer sources carry reconstruction value that propagation does not replace" |
| C | Parameter-matched gap at further seeds, over the L1 dictionary, on the untrained network, at 12 layers | §6.1 / abstract: "+0.008 parameter-matched"; intro: depth trend and half-learned split "were not repeated at matched parameters" |
| D | AR-kernel transfer between the trained and the untrained network | §3.4 (T4): "substantially architectural" is a comparison of levels |

## Rebuilt inputs and their gates

None of the inputs is on disk; each is rebuilt with its recorded recipe and must
reproduce the tracked table it originally produced **before** anything is measured on it
(`scripts/v11_gate.py`). A failed gate skips what depends on it and is reported.

| Input | Recipe | Gate (tolerance) |
|---|---|---|
| MLSAE sources | `build_410m_sources.py --suite mlsae` from the cached SAEs | alive counts, M\*, recon R² vs `runs/2026-06-25-ph3-sources-410md` (the E1 tolerances) |
| trained-L1 dictionary | the verified seed-0 retrain of E1 (`data/…/v11/sources_trained_l1_s0.npz`) | same, vs `runs/2026-06-29-ph4-e2-train-l1-410md` |
| SSM checkpoints (MLSAE, L1) | `phase4_checkpoint.py`, λ_L1 = 1, r = d_z = 32, 10 outer iterations, seed 0 | train/test R², L0, pool match rate, the 200-feature candidate pool and the stable count vs the recorded `candidate_pool.json` (1e-4) |
| recorded probe arms | the R7 code path inside the new probe script | per-layer accuracies, confidence R² and band contrasts vs the tracked R7 tables (1e-6); a miss is reported, the run is kept |
| truncated encoder | `v11_causal_codes.py` at full length | reproduces the checkpoint's `u_infer_full` (max abs difference ≤ 1e-5); **hard stop** for check A on that dictionary |
| embedding-only fit | `phase4_embed_only.py --cutoffs 0`, zero-initialised bias | train/test and per-layer R² vs the recorded sweep (1e-4) |
| rank-64 baseline (MLSAE, seed 0) | `r0_bootstrap_ci.py --rank 64` on the rebuilt checkpoint | gap and CI vs `runs/2026-07-04-r9b-gate3-ci-rank64` (1e-4) |
| untrained-network store | `phase4_t4_untrained.py --source factual` | pooled AR(1) R² at m = 0 and m = 50, both networks, vs the tracked T1 table (1e-6) |
| untrained-network L1 dictionary | `train_410m_saes.py --mode l1 --l1-coeff 0.5`, trainer defaults (2,000 epochs, lr 1e-3), seed 0 | vs `runs/2026-06-30-ph4-t4recon-train-l1-untrained-410md`; then the rank-32 knee vs `runs/2026-07-01-ph4-t4recon-gate3-l1-untrained-410md` (1e-4) |
| 12-layer store and sources | `build_dataset.py`, `collect_activations.py`, `build_410m_sources.py` with the 160m config | sources vs `runs/2026-06-26-xm-p160-sources`; then the seed-0 rank-32 knee vs `runs/2026-06-26-xm-p160-gate3` (1e-4) |

## A — the state probe without look-ahead

**What is wrong with the recorded probe.** Its codes are LASSO codes fit to each
example's whole update trajectory. The reconstruction of Δ_ℓ reaches the codes below
layer ℓ only through the state z_ℓ, so whatever they learn about Δ_ℓ is written along
the directions the state map reads. And every non-state arm is a variance-ranked PCA:
there is no per-layer arm that was trained on Δ.

**Procedure.** For each depth ℓ = 1 … 23 the recorded held-out encoder is re-solved with
the operators frozen, the same λ_L1, the same initial SAE codes, the same 100 FISTA
steps and the same Lipschitz hand-over, against Δ_0 … Δ_{ℓ-1} only
(`layer_dynamics.diagnostics.causal_codes`). Arms, all 32-dimensional, all probed with
the recorded probe's linear models, selection split and seeds:

| Arm | Codes | What it is |
|---|---|---|
| `state`, `code_pca`, `code_prefix_pca`, `resid_pca` | whole trajectory | the recorded R7 arms, verbatim |
| `feed_image` | whole trajectory | the feedthrough's rank-32 image D2_ℓᵀ u_ℓ: per-layer, reconstruction-trained, sees Δ_ℓ as the recorded state does |
| `state_causal` | Δ_{<ℓ} only | the state entering layer ℓ built from the prefix codes |
| `prefix_pca_causal` | Δ_{<ℓ} only | PCA of the same prefix codes u_0 … u_{ℓ-1} (exactly the causal state's inputs) |
| `feed_prev_causal` | Δ_{<ℓ} only | the feedthrough's image of the last past code u_{ℓ-1} |
| `sae_pca` | none | PCA of the SAE latents of v_ℓ |

**Primary statistic (per dictionary):** `state_causal − prefix_pca_causal`, answer
identity (205-way top-1), middle depth band (layers 8–15), paired bootstrap over the 311
test examples, 10,000 resamples, the band-aggregated statistic of R7.

| Per dictionary | Row |
|---|---|
| 95% CI above 0 | **survives** |
| 95% CI below 0 | **reversed** |
| otherwise | **not distinguishable** |

| Both dictionaries | Row | Consequence for the paper |
|---|---|---|
| survives on MLSAE and L1 | **survives on every dictionary** | The probe clause stays in the abstract; §6.7 reports the causal contrasts as the evidence and the recorded ones as the look-ahead-exposed version. |
| survives on one | **dictionary-dependent** | The claim is scoped to that dictionary and leaves the abstract's unconditional wording. |
| survives on neither | **does not survive** | The claim is withdrawn from the abstract, contribution 5 and the Discussion; §6.7 reports the recorded lead as look-ahead. |

**Secondary, reported without a verdict:** `state_causal − feed_prev_causal`,
`state_causal − sae_pca`, `state_causal − resid_pca`, `state − state_causal` (the size of
the look-ahead), the early and late bands, relation class. One reading rule is fixed in
advance: if `state − feed_image` is not above zero at mid-depth on a dictionary, §6.7 will
say that on that dictionary the recorded lead over PCA is what a reconstruction-trained
projection of the codes gives, not something the cross-layer state adds.

## B — the prefix sweep with a mean-initialised bias

**Procedure.** `phase4_embed_only.py --bias-init mean`: identical to the recorded sweep
(frozen MLSAE sources, rank 32, 250 Adam epochs, cutoffs 0/3/7/11/15/19/23, seed 0)
except that b starts at the train split's per-depth mean update instead of zero.

| Row | Criterion | If yes | If no |
|---|---|---|---|
| (i) embedding-only at the mean floor | \|test R²\| at ℓ_max = 0 ≤ 0.02 | §6.9's reading stands: −0.44 was the unconverged bias. The mean-initialised values are the ones §6.9 should quote for the lower half. | The −0.44 is not only the bias; §6.9's explanation is revised. |
| (ii) late-layer sources load-bearing | R²(ℓ_max = 23) − R²(ℓ_max = 19) ≥ +0.10 | "Late-layer sources carry reconstruction value" stands. | That sentence is withdrawn. |
| (iii) upper half stable | max \|ΔR²\| over cutoffs 15, 19, 23 against the recorded sweep ≤ 0.02 | §6.9's quoted upper-half numbers stand. | They are replaced by the new ones. |

Monotonicity over the seven cutoffs is reported. Whether fig13 is redrawn is the
author's decision; the recorded figure is not touched by this run.

## C — the parameter-matched gap

The gap is rank-32 SSM minus rank-64 per-layer baseline (8.36M vs 8.41M parameters) at
the λ_L1 = 1 knee under the LASSO protocol. The recorded value is one fit per arm:
+0.0077 [+0.0053, +0.0102], MLSAE, seed 0.

1. **Refits (MLSAE).** Rank-64 baseline at seeds 1 and 2 (`gate3_pareto.py --rank 64
   --lambdas 1 --variants baseline --seed s`; `--seed` sets split and initialisation, as
   in the recorded three-seed series). Gap at seed s = the recorded rank-32 SSM test R²
   at that seed (`runs/2026-06-29-ph3-gate3-410md-s{1,2}`) minus the new baseline; the
   two arms share the split but the gap is unpaired at the example level. With the
   seed-0 value: **robust to refit** if all three gaps are positive and mean − 2 sd > 0;
   otherwise **not robust to refit**. *If robust,* §6.1 quotes mean ± sd beside the
   seed-0 interval. *If not,* "small, real" is withdrawn and the abstract's
   parameter-matched clause says the margin does not survive refits.
2. **L1 dictionary.** `r0_bootstrap_ci.py --rank 64` on the rebuilt L1 checkpoint:
   paired bootstrap as for MLSAE. CI above 0 → **holds**; below 0 → **reversed**;
   otherwise **not established**. Unless it holds, the abstract's +0.008 is scoped to the
   MLSAE dictionary.
3. **Untrained network.** L1 dictionary trained on the untrained network's activations
   (the recorded recipe), rank-32 SSM and rank-64 baseline at the knee, seed 0. Share =
   untrained gap / trained L1 gap, one fit per network, no interval. It is **not
   resolvable** unless the trained gap exceeds 3 refit sd (the sd of the three MLSAE
   gaps of item 1; 0.0014, the recorded rank-matched value, if item 1 did not complete).
   Share ≥ 0.75 → **largely architectural**; ≤ 0.25 → **largely learned**; between →
   **about half**. The result replaces the intro's sentence that the half-learned split
   was not repeated at matched parameters.
4. **12 layers.** Pythia-160m-deduped over its MLSAE suite, rank-64 baseline at seeds
   0/1/2 against the recorded rank-32 SSM at the same seeds. Against the three 24-layer
   gaps: **larger at the deeper model** if the difference of means exceeds
   2·√(sd₁₂² + sd₂₄²); **larger at the shallower model** if the reverse; otherwise **no
   depth trend at matched parameters**. Two depths: at best "consistent with", never a
   trend. The result replaces the intro's sentence that the depth trend was not repeated
   at matched parameters.

## D — kernel transfer between the trained and the untrained network

**Procedure.** Both stores hold the same 1,555 prompts and share the split. On each
network the recorded pooled AR(1) kernel is fit (own input PCA basis, 256 components,
ranks 16–128, 23 depths), then scored on the other network with the per-depth
intercepts refit there — the Phase-2 transfer test with two networks in place of two
bands. Run at m = 0 and with each network's own top-50 state directions removed (T1).

**Primary statistic:** retention = R²(fit a, eval b) / R²(fit b, eval b) at m = 50,
rank 128, in both directions.

| Row | Criterion | Consequence for §3.4 |
|---|---|---|
| **transfers** | both retentions ≥ 0.5 | T4 may say the trained network's persistence-stripped kernel is largely the one the architecture gives at initialisation. |
| **does not transfer** | untrained → trained ≤ 0.1 (the Phase-2 tolerance) | T4 keeps "predictability of this size needs no training" and adds that the two kernels are different maps: "architectural" describes the level only. |
| **partial** | anything else | Reported as such, with both retentions. |

**Secondary, reported without a verdict:** the same with one scalar gain refit per
target depth (forgives a kernel that is right up to a depth-dependent scale), m = 0,
ranks 16/32/64, the relative Frobenius distance between the two effective maps, and a
scrambled-source null transfer.

## Not in scope

- **Pythia-70m as a second depth point.** The tracked record of its dataset build shows
  no kept examples under the configured threshold; its 1,000-example construction is not
  reproducible from the tracked tables with a gate we would trust.
- **Kernel transfer on GPT-2-small.** The model is not in the local cache; the run is
  offline.
- **Other operating points.** The parameter-matched comparison is at the knee only, as
  recorded.
- No threshold above will be adjusted after the fact. Anything the run suggests beyond
  these rows is reported as exploratory.

## Running it

`scripts/launch_v11_checks.sh` (detached systemd unit; `check` / `smoke` / `status` /
`stop`; resumable, `V11_DATE=<start date>` if the date has changed). Log:
`runs/v11-checks-overnight.log`. Order: sources → D → A (L1, then MLSAE) → B → C (410m)
→ C (untrained) → C (12 layers) → summary. Estimated ~10 h on the GTX 1050; the first
verdict (D) lands within the first half hour, A on the L1 dictionary within two hours.

## Results

### 2026-11-17 — all four checks complete

**Runs.** First pass 15:50–20:28 (commit `61fb2ea`); second pass 20:29–23:18 (commits
`a3ae4f9` checkpoint memory fix, `40c5af4` recorded prefix recipe), resuming every finished
step. Run directories `runs/2026-11-17-v11c-*`; summary
`runs/2026-11-17-v11c-summary/tables/checks_summary.json`; log
`runs/v11-checks-overnight.log`. **Every gate passed with a maximum difference of 0.0**:
the seed-0 L1 dictionary, the MLSAE sources, both stores' AR(1) levels, both SSM
checkpoints (train/test R², L0, the 200-feature candidate pool), the embedding-only fit
(at the recorded recipe), the seed-0 rank-64 interval (+0.007734 [+0.005314, +0.010219]),
the untrained dictionary and knee, the 12-layer sources and knee. The four recorded probe
arms reproduce the R7 tables exactly on both dictionaries; the truncated encoder reproduces
`u_infer_full` exactly at full length on both.

| Check | Verdict row |
|---|---|
| A — probe without look-ahead | **survives on every dictionary** |
| B — prefix sweep, mean-initialised bias | (i) **at the mean floor**; (ii) **late sources load-bearing**; (iii) **upper half not stable** |
| C1 — MLSAE refits | **robust to refit** |
| C2 — L1 dictionary | **holds** |
| C3 — untrained network | **largely learned** |
| C4 — 12 vs 24 layers | **larger at the deeper model** |
| D — kernel transfer | **does not transfer** |

**A.** Answer identity, middle band (layers 8–15), paired 95% CIs over the 311 test
examples:

| Contrast | MLSAE | L1 |
|---|---|---|
| recorded: state − prefix PCA (whole-trajectory codes) | +0.080 [+0.048, +0.113] | +0.173 [+0.131, +0.217] |
| **primary: causal state − PCA of the same prefix codes** | **+0.052 [+0.019, +0.085]** | **+0.133 [+0.095, +0.174]** |
| look-ahead: state − causal state | +0.042 [+0.031, +0.055] | +0.045 [+0.032, +0.058] |
| state − feedthrough image of the current code | −0.019 [−0.038, +0.001] | −0.013 [−0.029, +0.003] |
| causal state − feedthrough image of the last past code | −0.024 [−0.046, −0.003] | −0.041 [−0.057, −0.025] |
| causal state − SAE-latent PCA | +0.019 [−0.004, +0.042] | +0.058 [+0.034, +0.083] |
| causal state − raw-residual PCA | −0.076 [−0.101, −0.053] | −0.024 [−0.046, −0.003] |

The primary contrast survives on both dictionaries, at about two thirds (MLSAE) and three
quarters (L1) of the recorded lead; the look-ahead accounts for the rest. **The pre-committed
reading rule applies on both dictionaries:** the recorded state is not above the
feedthrough's own 32-dimensional image of the current code, and the causal state is *below*
the feedthrough image of the last past code. So the lead over PCA is what a
reconstruction-trained projection of the codes gives; the cross-layer state does not add to
it. Other bands, MLSAE primary: early +0.004 [−0.015, +0.023], late +0.085 [+0.055, +0.115].
Relation class, MLSAE, middle band: the causal state is *below* the prefix PCA (−0.049
[−0.060, −0.039]).

**B.** Embedding-only test R² −0.003 (recorded −0.438): (i) holds — the recorded value was
the unconverged bias. Full −0.003 / +0.068 / +0.178 / +0.395 / +0.474 / +0.645 / +0.785 at
ℓ_max = 0/3/7/11/15/19/23 (recorded −0.438 / −0.343 / −0.248 / +0.199 / +0.404 / +0.576 /
+0.770); monotone. (ii) R²(23) − R²(19) = +0.140 ≥ 0.10 (recorded +0.194): holds. (iii)
upper half moves by up to +0.071 (ℓ_max 15 and 19): not stable, so §6.9's quoted values are
replaced. **The carry-forward rows that §6.8 cites change**: with sources up to layer 3 the
per-layer R² at ℓ = 4…9 is +0.71, +0.49, 0.00, 0.00, 0.00, 0.00 (recorded +0.69, +0.41,
+0.31, +0.24, +0.23, +0.21); up to layer 7, at ℓ = 8…14: +0.71, +0.53, +0.30, +0.23, +0.05,
0.00, 0.00 (recorded +0.60, +0.39, +0.16, +0.20, +0.25, +0.25, +0.20); up to layer 11, at
ℓ = 12…16: +0.69, +0.53, +0.33, +0.22, +0.20. The long tails of the recorded rows were
the under-fit bias; a refit state carries withheld sources two layers from an early cutoff
and four to five from a middle one.

**C.** Parameter-matched knee gap (rank-32 SSM − rank-64 per-layer baseline):

| | gap | note |
|---|---|---|
| 24 layers, MLSAE, seeds 0/1/2 | +0.0077 / +0.0098 / +0.0090 (mean +0.0089 ± 0.0011) | C1 robust |
| 24 layers, L1 dictionary | +0.0055 [+0.0030, +0.0081] | C2 holds; SSM L0 0.047, baseline 0.046 |
| 24 layers, untrained network (own L1 dictionary) | −0.0254 (SSM 0.7426, baseline 0.7680; rank-matched +0.0287) | C3: share −4.6, largely learned; baseline L0 0.065 vs SSM 0.059 |
| 12 layers (Pythia-160m), seeds 0/1/2 | −0.0081 / −0.0086 / −0.0097 (mean −0.0088 ± 0.0008; rank-matched +0.012 to +0.015) | C4: larger at the deeper model (Δ +0.0176 > 2 × 0.0013) |

On the untrained network and at 12 layers the state is *worse* than a per-layer decoder of
the same size; at 24 layers on the trained network it is better, under both dictionaries
and every refit. Two depths and one family: "consistent with", not a trend.

**D.** At m = 50, rank 128: within-network R² 0.436 (trained) / 0.503 (untrained), both
exactly the tracked T1 values; untrained → trained −0.093, trained → untrained −0.136;
retentions −0.21 / −0.27; with a refit gain per depth ≈ 0.0002 / 0.0004; relative Frobenius
distance of the two maps 1.417 (√2 = unrelated); scrambled-source transfer ≈ 0. At m = 0
the cross scores are −0.09 / −0.21. The two networks' kernels are different maps: the
untrained network matches the trained one in *level* of predictability, not in kernel.

**Consequences for the paper** (proposed; the author approves the wording): §6.7, the
abstract's probe clause and contribution 5 — the lead survives the look-ahead at reduced size,
but it is a reconstruction-trained projection's lead, not the cross-layer state's; §6.1 and
the abstract — the +0.008 is robust to refits and holds over the L1 dictionary, but it is
absent on the untrained network and reversed at 12 layers; the intro's two "not repeated at
matched parameters" sentences are replaced by C3 and C4; §3.4 (T4) — "architectural"
describes the level, and the kernels do not transfer; §6.8's second caveat and §6.9's
numbers are replaced by B's; `tab:operating`'s footnote (and WP4's progress note) say 250
epochs where the sweep used 180 with λ_reg 0.1.
