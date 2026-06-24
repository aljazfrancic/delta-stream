# Phase 1 — Diagnostic

Goal: does the residual update $`\Delta_\ell = v_{\ell+1} - v_\ell`$ have any structured cross-layer dynamics at all on factual-recall inputs, above a null baseline, after controlling for context leakage?

**Gate 1.** Decide whether H1 (depth-local predictability) is supported.

> Nav: [← Implementation](README.md) · [Phase 2 →](phase-2-characterization.md) · [Proposal](../README.md) · [Glossary](../glossary.md)

---

## Step 0 — Environment

- Fresh Python env (`uv` or `conda`), Python 3.11+.
- Core dependencies: `torch`, `transformer_lens`, `datasets`, `numpy`, `scipy`, `scikit-learn`, `statsmodels` (for Granger), `h5py`, `matplotlib`, `pandas`. Pin versions in `pyproject.toml`.
- Create a repo root `layer_dynamics/` with directories `data/`, `runs/`, `scripts/`, `src/layer_dynamics/`, `figures/`.
- Every run script takes `--config config.yaml --out runs/<date>-<desc>/` and logs git commit + seed.

## Step 1 — Assemble the factual-recall dataset

**Source files.** Download the CounterFact dataset (released with [ROME](../glossary.md#rome), Meng et al. 2022) and the Pararel dataset ([Elazar et al. 2021](../glossary.md#ref-elazar-2021)). Optionally the T-REx subset used in [Geva et al. 2023](../glossary.md#ref-geva-2023).

**Filtering pipeline.**
1. Parse each triple $`(s, r, o)`$ into a natural-language prompt (e.g., for $`(s, r, o) = (\textit{Paris}, \textit{capital of}, \textit{France})`$, the prompt "Paris is the capital of" targeting "France"). Use the prompts already supplied by the source datasets where possible — CounterFact stores many triples in the reverse direction ("The capital of France is" → "Paris"), so filter or reformat to match the s-r-o convention used in §4 of the implementation-plan README before proceeding.
2. Run the prompt through [Pythia-410m](../glossary.md#pythia) via [TransformerLens](../glossary.md#transformerlens).
3. Extract the top-1 predicted next token. Keep only examples where:
   - the top-1 prediction matches the object $`o`$ (exact match on the first token), AND
   - the top-1 probability exceeds a threshold (start at 0.3; tune upward if the filtered set comfortably exceeds 5,000 examples, downward only if it does not — matches the asymmetry in [§4 of the implementation-plan README](README.md#4-target-circuits-factual-recall)).
4. Stratify by relation type. Keep at least 500 examples per relation class; if a class is too small, upweight it or drop it — record which.

**Output:** a `filtered_triples.parquet` with columns `subject`, `relation`, `object`, `prompt`, `top1_token`, `top1_prob`, `relation_class`.

**Sanity check.** Before proceeding, print a random sample of 50 rows and read them. If anything looks wrong (weird tokenisation, wrong relation label), fix it now, not later.

## Step 2 — Collect residual stream trajectories

**Script:** `scripts/collect_activations.py`.

For each example:
1. Tokenize the prompt.
2. Identify the *last prompt token* — this is the token immediately before the predicted $`o`$. All activations are extracted at this token position.
3. Run the model with TransformerLens hooks to capture the residual stream at every depth: `blocks.0.hook_resid_pre` for $`v_0`$ (the embedding output, before any block has written) plus `blocks.{l}.hook_resid_post` for $`\ell \in \lbrace 0, 1, \ldots, L-1\rbrace`$ to give $`v_1, \ldots, v_L`$. In Pythia-410m this means $`L=24`$ blocks, giving 25 residual states per example ($`v_0`$ through $`v_{24}`$).
4. Save $`v_\ell \in \mathbb{R}^{1024}`$ for every $`\ell`$, along with example metadata.

**Storage.** HDF5 file with datasets `residual_stream[N, L+1, D]`, `example_id[N]`, `relation_class[N]`. $`N=10{,}000`$, $`L+1=25`$, $`D=1024`$ → ~1 GB at float32. Save as float32 first; downcast later if compute-bound.

**Do not yet compute $`\Delta_\ell`$.** Keep $`v_\ell`$ and compute differences downstream — this leaves the door open for analyses that need $`v`$ itself and costs nothing.

**Note on token position.** All Phase 1 trajectories are collected at a single token position (the final prompt token). Because attention at every layer mixes across tokens, this choice biases the characterisation toward whatever happens at the query position specifically. Whether the AR structure found at this position generalises to other positions is tested explicitly in [Phase 2 Step 3](phase-2-characterization.md#step-3--token-position-invariance) — do not silently assume invariance.

**Sanity check.** For 5 random examples, verify that running the model's final layer-norm (`ln_final` in TransformerLens) on $`v_L`$ and then the unembedding recovers the expected top-1 token. Skipping `ln_final` and unembedding $`v_L`$ directly is the most common reason this check fails for a bug-free extraction — report that path as a known confounder before concluding the extraction is wrong.

## Step 3 — Linear depth-local predictability (H1, weak form)

**Question.** Can we predict $`\Delta_\ell`$ from $`v_\ell`$ with a low-rank linear map, better than trivial baselines?

**Procedure.** For every consecutive layer pair $`(\ell, \ell+1)`$:
1. Build matrices $`V_\ell \in \mathbb{R}^{N \times D}`$ (residual states stacked across examples) and $`Y_\ell \in \mathbb{R}^{N \times D}`$ (updates stacked across examples; $`D`$ remains the residual dimension throughout).
2. Fit [ridge regression](../glossary.md#ridge) with [low-rank](../glossary.md#low-rank) constraint (rank $`r \in \lbrace 16, 32, 64, 128\rbrace`$): $`Y_\ell \approx V_\ell W^\top`$, with $`W = A B^\top`$, $`A \in \mathbb{R}^{D \times r}`$, $`B \in \mathbb{R}^{D \times r}`$. Train on 80% of examples, evaluate $`R^2`$ on the held-out 20%.
3. Implementation: either `sklearn.linear_model.Ridge` on the full-rank problem followed by truncated SVD of $`W`$, or direct alternating minimisation. The former is easier; use it first.

**Baselines to compare against.**
- **Zero-prediction:** $`\hat\Delta_\ell = 0`$. The "does the layer do anything?" baseline. Under the standard coefficient-of-determination, its $`R^2`$ is generally negative — it loses to the mean predictor by a margin equal to $`\|\bar\Delta_\ell\|^2`$ over the total variance.
- **Mean-prediction:** $`\hat\Delta_\ell = \bar\Delta_\ell`$ (the average update at that layer across examples). $`R^2 = 0`$ by definition; any positive $`R^2`$ above this means the ridge fit captures example-specific structure beyond layer-wide bias.
- **Identity-subtracted baseline:** regress $`v_{\ell+1}`$ on $`v_\ell`$ directly (not $`\Delta_\ell`$) and report its $`R^2`$ alongside the ridge-fit $`R^2`$ on $`\Delta_\ell`$. Because $`v_{\ell+1} \approx v_\ell`$ + a small update, this regression's $`R^2`$ will sit near 1 and is dominated by the identity component; reporting it lets a reader see directly that the bulk of $`v_{\ell+1}`$'s variance is the residual connection rather than the layer's contribution. **Do not** read the *gap* between the two $`R^2`$ as "how much work the $`\Delta`$ parameterisation is doing" — the two regressions have different targets, so the gap mostly reflects the variance contribution of the identity to $`v_{\ell+1}`$ and is largely a tautology of the residual stream's additive form. The substantive predictability claim is the $`\Delta_\ell`$ ridge $`R^2`$ against the mean-prediction and Tuned-Lens baselines below; the $`v_{\ell+1}`$ regression is reported for transparency, not as a comparator.
- **Tuned-Lens-implied update (the gate baseline).** Load a [Tuned Lens](../glossary.md#tuned-lens) for Pythia-410m — public checkpoint if available, otherwise train one following Belrose et al. 2023 (a small job relative to the rest of the pipeline). Compute the lens's aligned-residual adapter output $`\hat r_\ell = T_\ell v_\ell + c_\ell`$ at every layer, where $`T_\ell`$ and $`c_\ell`$ are the TL translator and bias (using $`T`$ rather than the Belrose et al. paper's $`A`$ to avoid colliding with the state-space matrices $`A_\ell`$ in [Phase 3 Branch B](phase-3-decomposition.md#step-1--lock-the-architecture-based-on-gate-2), and $`c`$ rather than the paper's $`b`$ to avoid colliding with the per-layer decomposition bias $`b_\ell`$ in [Phase 3 Step 3](phase-3-decomposition.md#step-3--loss-and-optimisation-branch-a--c)). This lives in residual space, *before* the unembedding, so no inversion of the unembedding is needed. Define $`\hat\Delta_\ell^{\text{TL}} = \hat r_{\ell+1} - \hat r_\ell`$ and score it as a prediction of $`\Delta_\ell`$ on the same held-out split as the ridge fit. **Last-layer convention.** Standard Tuned-Lens releases supply translators $`T_\ell, c_\ell`$ for $`\ell \in \lbrace 0, \ldots, L-1\rbrace`$ — one per intermediate residual stream that needs aligning toward the final state. To define $`\hat\Delta_{L-1}^{\text{TL}} = \hat r_L - \hat r_{L-1}`$ at the top of the stack, set $`\hat r_L \equiv v_L`$ (equivalently $`T_L = I`$, $`c_L = 0`$): the final residual is by construction what TL is trained to align toward, so the "lens" at $`\ell = L`$ is the identity for free. The layer-averaged $`R^2`$ used for the gate then runs over all $`L`$ layer pairs $`\ell \in \lbrace 0, \ldots, L-1\rbrace`$ uniformly, with no last-layer exclusion. **This is the gate baseline for the AR test in [Step 5](#step-5--ark-on-delta-and-granger-causality-h1-full-form):** per [proposal §5 item 1](../README.md#5-experimental-strategy), the ridge-fit $`R^2`$ (at its best-performing rank) averaged across layer pairs must exceed the TL-implied $`R^2`$ averaged across the same pairs before Step 5 is worth running. Caveats to report alongside the comparison: (i) TL is trained for a different objective (logit-space prediction), (ii) the TL-implied diff uses $`v_{\ell+1}`$ that ridge on $`v_\ell`$ alone does not see — so TL has an information advantage; if ridge wins anyway, the depth-local structure it captures is genuinely beyond calibrated refinement.

**Report.** For each layer pair $`(\ell, \ell+1)`$, log:
- $`R^2`$ at each rank $`r`$ against each baseline.
- The rank at which $`R^2`$ saturates (the effective per-layer predictability dimension).
- A heatmap of $`R^2(\ell, r)`$ across all pairs.
- Layer-averaged ridge $`R^2`$ (at the best rank) against layer-averaged TL-implied $`R^2`$, with a clear pass/fail annotation relative to the Step 5 gate.

## Step 4 — CCA backstop

**Question.** Is there shared subspace structure between $`v_\ell`$ and $`\Delta_\ell`$ that the linear regression is missing?

**Procedure.** For each layer pair, run [canonical correlation analysis](../glossary.md#cca) between $`V_\ell`$ and $`Y_\ell`$ (same stacked-updates matrix as Step 3). Regularise the within-matrix covariance estimates (shrinkage CCA — `statsmodels.multivariate.cancorr` and `scikit-learn`'s `cross_decomposition.CCA` cover the unregularised case; for regularised / kernel / deep variants use [`cca-zoo`](https://pypi.org/project/cca-zoo/) or `pyrcca`).

**Report.** The first 10 canonical correlations per layer pair, stacked as a heatmap.

**Interpretation key.**
- High $`R^2`$ *and* high canonical correlations: linear predictability is real and captures the dominant shared structure.
- Low $`R^2`$ but high canonical correlations: ridge is underfitting; there is structure but not in the form "predict $`y`$ from $`x`$." This could be symmetric (shared latent variables affecting both) or nonlinear. Flag for Phase 2.
- Low both: genuine absence of linear dependence; H1 is in trouble at this pair.

## Step 5 — AR($`k`$) on $`\Delta`$, and Granger causality (H1, full form)

**Prerequisite.** The ridge diagnostic from [Step 3](#step-3--linear-depth-local-predictability-h1-weak-form) must beat the Tuned-Lens-implied update baseline (layer-averaged $`R^2`$ at the best-performing rank). This gate comes directly from [proposal §5 item 1](../README.md#5-experimental-strategy) and is pragmatic, not logical — §4 of the proposal notes that the ridge on $`v_\ell`$ test and the AR on $`\Delta`$ test neither entail nor rule each other out — but if the cheap diagnostic cannot separate from a calibrated per-layer lens, running the expensive AR test is not warranted. If the gate fails, skip this step and route to Gate 1 ([Step 6](#step-6--gate-1-decision)) with the cheap-diagnostic result as the headline.

**Question.** Is $`\Delta_\ell`$ better predicted from the $`k`$ preceding updates $`\Delta_{\ell-1}, \ldots, \Delta_{\ell-k}`$ than from $`v_\ell`$ alone? And how far back in depth does the dependence extend?

**Procedure.**
1. **AR($`k`$) fit.** For $`k \in \lbrace 1, 2, 3, 5, 10, 15, 20\rbrace`$ (matches the Phase 2 Step 2 sweep so that the effective-depth refinement there does not land on a lag Granger was never tested at), fit:
   $$\hat\Delta_\ell = \mu_\ell + \sum_{j=1}^k W_j \Delta_{\ell-j}$$
   with low-rank $`W_j`$ (same rank sweep as Step 3) shared across depths and **per-depth intercept $`\mu_\ell \in \mathbb{R}^D`$ fit jointly with $`W_j`$** ($`\mu`$ rather than $`c`$ to avoid colliding with the Tuned-Lens bias $`c_\ell`$ introduced in [Step 3](#step-3--linear-depth-local-predictability-h1-weak-form), and $`\mu`$ rather than $`b`$ for the same reason $`c`$ was: $`b_\ell`$ is taken by the Phase 3 decomposition bias). Evaluate $`R^2`$ on held-out examples. **Why the per-depth intercept matters.** Without $`\mu_\ell`$, the pooled fit forces a single shared kernel to fit both the AR dynamics *and* the per-depth offset $`\bar\Delta_\ell`$ (which is generally nonzero — it is the same quantity Phase 3's $`b_\ell`$ is introduced to absorb). Folding the offsets into $`W_j`$ inflates apparent depth-shared structure and, downstream, contaminates the Phase 2 Step 1 per-band stationarity comparison: per-band kernels would differ in part because $`\bar\Delta_\ell`$ differs between bands, not because the dynamics differ. Including $`\mu_\ell`$ keeps $`W_j`$ a clean transition operator, comparable across depths and across bands. **Stationarity-implicit caveat.** The $`W_j`$ here carry no $`\ell`$-index — the fit pools $`(\Delta_\ell, \Delta_{\ell-1}, \ldots, \Delta_{\ell-k})`$ tuples across all feasible depths $`\ell \ge k`$ and produces a single shared kernel. So a high pooled $`R^2`$ here is jointly consistent with H1 *and* approximate AR-stationarity, not H1 alone — and the converse failure mode matters too: if H1 holds with depth-varying kernels (the H2-fails world the proposal explicitly admits as a possibility), the pooled fit can come back disappointingly low even though the per-depth Granger F-tests in procedure item 2 below (the nested-F lag test, distinct from Phase 1's top-level Step 2) are widely significant. **Diagnostic for this combined failure.** If the pooled AR $`R^2`$ is low *and* the (depth × lag) Granger heatmap from procedure item 2 below shows widespread Bonferroni-significant cells, do not route to the Phase 1 negative-result branch. The honest read is "H1 holds non-stationarily" and Phase 2 Step 1's smoothly-varying / segmented branches (state-space or per-segment kernel) are the right downstream architecture, not a write-up of "no depth-local structure." Flag this combination explicitly in the Gate 1 memo so the Phase 2 design is not pre-committed to the stationary-kernel assumption.
2. **Nested-F lag test (Granger-style).** At every depth $`\ell`$ and for every lag $`j`$ in the sweep with $`\ell \ge j`$ (feasibility: AR($`j`$) at depth $`\ell`$ requires $`\ell \ge j`$), fit two nested AR models on examples at that depth, **each including an intercept** (the per-depth fit naturally has a single $`\mu \in \mathbb{R}^D`$ since only one $`\ell`$ is involved — same intercept symbol as the pooled AR fit above; the default in `statsmodels.tsa.api.VAR` is to include this intercept, stated explicitly so the implementation does not silently drop it):
   - reduced: lags $`\lbrace 1, \ldots, j-1\rbrace`$,
   - full: lags $`\lbrace 1, \ldots, j\rbrace`$.
   Because $`\Delta_\ell`$ is $`D`$-dimensional, the residuals are vectors and the comparison is multivariate. The default path is the F-form Granger causality test exposed by `statsmodels.tsa.api.VAR.test_causality` (which returns either an `f` or `wald` statistic on a `CausalityTestResults` object — pick `f` for finite-sample calibration and read the F statistic plus its numerator/denominator degrees of freedom off that result). For a Wilks' lambda likelihood-ratio reading of the same nested comparison, compute it manually from the reduced/full model log-likelihoods (or via `statsmodels.multivariate.manova.MANOVA.mv_test` on stacked residuals) and convert to an asymptotic p-value via the Bartlett-corrected $`\chi^2`$ approximation; Pillai's trace and Roy's largest root are alternatives that give similar p-values at the sample sizes here ($`N \ge 5{,}000`$). Pick one (the `VAR.test_causality` F-form is the lowest-friction default) and use it consistently across the sweep, recording the choice so a reader can re-derive the test from the saved statistic and degrees of freedom. The result is a p-value **per (depth $`\ell`$, lag $`j`$) pair** — "does lag $`j`$ add predictive information beyond shorter lags at depth $`\ell`$?" Strictly this is an AR-order selection test on a single series rather than two-series Granger causality; I report it as Granger-style in interpretation — it answers the same "how far back in depth does a dependence reach?" question Granger does when the two series are chosen as $`\Delta_{\ell-j}`$ and $`\Delta_\ell`$.
3. **Per-depth lag profile.** The test is inherently per-depth (the result at depth $`\ell_1`$ can differ from the result at depth $`\ell_2`$ — written as $`\ell_1, \ell_2`$ rather than $`\ell, \ell'`$ because $`\ell'`$ is reserved for the source layer in the kernel decomposition throughout this proposal), so report a 2D heatmap: (depth $`\ell`$) × (lag $`j`$) → $`-\log_{10} p`$. Plot $`-\log_{10} p`$ rather than the raw F statistic so the heatmap is directly comparable to the Bonferroni-corrected significance threshold used for $`k`$-selection in [Phase 2 Step 1](phase-2-characterization.md#step-1--stationarity-test); save the raw F statistic and the (numerator, denominator) degrees of freedom alongside the p-value in the per-cell metadata so the test can be re-derived from the saved tables.

**Null test 1 — Scrambled activations.**
1. For each layer $`\ell`$ independently, shuffle the example axis (permute $`N`$). Note: *independently per layer*, not the same permutation. This destroys within-example cross-layer structure while preserving per-layer marginal distributions exactly.
2. Re-run the AR fit and Granger test on the scrambled data.
3. Statistics significantly above the scrambled null are real; statistics at or below it are noise.

**Null test 2 — Context perturbation (the critical control).**

The bare s+r prompts from [Step 1](#step-1--assemble-the-factual-recall-dataset) (e.g. `"Paris is the capital of"` → `"France"`) usually contain no tokens outside the subject-relation span, so perturbation needs an explicit context to perturb. The procedure adds a fixed-length neutral preamble to each prompt and varies it.

1. From the Step 1 dataset, sample 1,000 examples and construct two prompt variants per example: (a) the **wrapped baseline** prepends a fixed neutral preamble (e.g. `"In school I learned that "`) so the new prompt becomes, for the Paris example, `"In school I learned that Paris is the capital of"`; (b) the **perturbed variant** keeps the same wrapped form but replaces the preamble tokens with the same number of random in-distribution function words sampled from a fixed pool. (The preamble length is deliberately left unnamed — the Tuned Lens translator $`T_\ell`$ introduced in [Step 3](#step-3--linear-depth-local-predictability-h1-weak-form) already claims the symbol $`T`$; the concrete preamble is chosen so its tokenised length is fixed across examples but its specific value is an implementation detail, not a reusable symbol.) The subject + relation-template span and the final prompt token (here `"of"`) sit at the same absolute positions in both variants. Drop any example where the model no longer recovers the correct top-1 token under the wrapped baseline (the wrapping itself can shift predictions; the test only makes sense on examples where it does not).
2. Recollect residual streams for the wrapped baseline and the perturbed variant.
3. Run the AR fit and Granger test on the perturbed variant and compare to the wrapped baseline.
4. **Interpretation.** If the Granger structure is unchanged: cross-layer dynamics at the final prompt token are a property of the subject-relation processing, depth-intrinsic. If it changes substantially: the apparent dynamics reflect context information carried by hidden variables (the preamble tokens leaking through attention into the query position), and we are characterising a cross-*context* phenomenon, not a depth-propagation phenomenon. Both outcomes are publishable; they say *different* things about H1.

## Step 6 — Gate 1 decision

At Gate 1, write a short memo (2–3 pages) answering:

1. **Is $`\Delta_\ell`$ predictable from $`v_\ell`$ above the trivial baselines (zero / mean / identity)?** (Step 3)
2. **Does the ridge fit beat the Tuned-Lens-implied update baseline (layer-averaged)?** (Step 3, TL baseline) — proposal-mandated precondition for the AR test.
3. **Is there shared subspace structure?** (Step 4)
4. **Is there AR structure over depth?** (Step 5)
5. **Do the Granger structures survive the scrambled null?** (Step 5)
6. **Do they survive the context-perturbation control?** (Step 5)
7. **What is the honest conclusion about H1?**

Route based on the answers:
- **(1, 2, 4, 5, 6) all yes, H1 cleanly supported:** proceed to [Phase 2](phase-2-characterization.md).
- **(1, 2, 4, 5) yes, (6) no (structure is context-dependent):** pivot to characterising context-reconstruction dynamics; Phase 2 is still valuable but the framing changes.
- **(1, 2) yes, pooled AR $`R^2`$ in (4) low *but* per-depth Granger heatmap in (5) widely Bonferroni-significant under the scrambled null:** this is the non-stationary-H1 case flagged in Step 5's stationarity-implicit caveat. *Do not* treat this as "(4) fails." The honest read is that H1 holds with depth-varying kernels — the pooled fit's stationarity assumption is what failed, not H1 itself. Proceed to Phase 2 with the explicit understanding that Phase 2 Step 1's decision matrix is likely to land on the smoothly-varying or segmented branch rather than block-Toeplitz.
- **(1) yes, (2) no (ridge does not beat TL):** AR test was skipped per the Step 5 prerequisite. Phase 1 is the paper — the headline result is "a cheap linear diagnostic on $`\Delta_\ell`$ does not separate from the Tuned Lens's calibrated per-layer refinement, so there is no evidence for depth-local structure beyond what a per-layer lens already captures." Spend the remaining effort corroborating this across models / circuits.
- **Any of (1, 4, 5) fails *and* the non-stationary-H1 escape route above does not apply:** Phase 1 is the paper. Spend the remaining effort writing up the negative result and running the minimum follow-ups (other models, other circuits) that make the negative result robust.

**Deliverable artefacts at Gate 1:**
- `runs/gate1/memo.md`
- `runs/gate1/figures/` — the $`R^2`$ heatmaps, CCA heatmaps, Granger profiles, null comparisons.
- `runs/gate1/tables/` — CSV of all numerical results.
