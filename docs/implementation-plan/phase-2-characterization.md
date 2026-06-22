# Phase 2 — Characterization

Given that Phase 1 found cross-layer structure in residual updates (Gate 1 passed), characterise *what kind*: is it stationary across depth? How far back does the dependence reach? Does it generalise across relation types? Across models?

**Gate 2.** Decide the decomposition architecture for Phase 3. Stationary convolutive kernel if H2 holds; state-space model if it does not.

> Nav: [← Phase 1](phase-1-diagnostic.md) · [Phase 3 →](phase-3-decomposition.md) · [Implementation](README.md) · [Proposal](../README.md) · [Glossary](../glossary.md)

---

## Step 1 — Stationarity test

**Question.** Is the AR kernel that predicts $`\Delta_\ell`$ from its recent past the *same* kernel at every depth $`\ell`$? That is, does [Toeplitz structure](../glossary.md#toeplitz) across depth hold?

**Why it matters.** If it holds, a single shared-across-depth kernel is a useful parameterisation and Phase 3 can fit a convolutive decomposition. If it fails — especially if there are clean phase transitions — a depth-indexed kernel or a [state-space model](../glossary.md#ssm) is the right abstraction and Phase 3 should use one.

**What this step tests vs. what H2 claims.** The test below runs on the AR on $`\Delta`$ form — "is the kernel predicting $`\Delta_\ell`$ from $`\Delta_{\ell-1}, \ldots, \Delta_{\ell-k}`$ the same at every depth?" — while H2 formally asserts stationarity of the *MA / generative* kernel that maps sparse sources $`s_{\ell-k}`$ into $`\Delta_\ell`$. Stationary MA implies a stationary best AR approximation, so an AR on $`\Delta`$ non-stationarity finding rules H2 out. The converse is weaker: AR on $`\Delta`$ stationarity is necessary but not strictly sufficient for MA stationarity without an invertibility assumption. Report AR on $`\Delta`$ stationarity as a strong proxy, not a logical equivalent, and recheck MA stationarity directly via the kernel fit in Phase 3 Step 4's sanity checks once the decomposition has been trained.

**Procedure.**
1. Partition the updates $`\Delta_\ell`$ (indexed $`\ell = 0, \ldots, L-1`$ per the $`\Delta_\ell = v_{\ell+1} - v_\ell`$ convention fixed in the proposal) into three non-overlapping depth bands: early ($`0 \le \ell \lt \lfloor L/3 \rfloor`$), middle ($`\lfloor L/3 \rfloor \le \ell \lt \lfloor 2L/3 \rfloor`$), late ($`\lfloor 2L/3 \rfloor \le \ell \lt L`$). On Pythia-410m with $`L=24`$, that is $`\Delta_\ell`$ for $`\ell`$ in 0–7, 8–15, 16–23.
2. Within each band, fit an AR($`k`$) kernel **of the same form as [Phase 1 Step 5](phase-1-diagnostic.md#step-5--ark-on-delta-and-granger-causality-h1-full-form): per-depth intercepts $`\mu_\ell`$ plus a band-shared kernel $`\lbrace W_j\rbrace`$** ($`\mu`$ rather than $`c`$ — Phase 1 Step 3's Tuned-Lens bias already claims $`c_\ell`$). The across-band stationarity comparison in step 3 below evaluates only the kernel matrices — the per-depth $`\mu_\ell`$ are nuisance parameters that absorb the per-depth $`\bar\Delta_\ell`$, so any band-to-band difference in $`\bar\Delta_\ell`$ is absorbed by $`\mu_\ell`$ rather than $`W_j`$. Without this, two bands could produce different $`W_j`$ purely because of mean-offset differences, and the stationarity test would falsely reject. Pick $`k`$ from the Phase 1 Step 5 Granger profile in two stages:
   - **Bonferroni denominator.** Use the count of (depth, lag) pairs where the F-test was feasible. This count is strictly less than $`L \times |\text{lags}|`$, because AR($`j`$) at depth $`\ell`$ requires $`\ell \ge j`$ — i.e. the strictly upper-triangular region above the diagonal $`\ell = j`$ of the (depth × lag) heatmap (depth on rows, lag on columns, both indices increasing in standard direction) is empty. Concretely, summing across the lag sweep: $`\sum_{j \in \text{lags}} (L - j)`$, with the convention that any $`j \gt L-1`$ contributes zero. Do **not** use $`L`$ as the denominator (which would amount to silently collapsing the lag dimension and counting only depths).
   - **$`k`$ selection.** Take the maximum across $`\ell`$: for each candidate lag $`j`$ in the sweep, ask whether *any* depth $`\ell`$ had a Bonferroni-significant F-test at lag $`j`$ (under the denominator above), and let $`k`$ be the largest such $`j`$. **Feasibility cap on $`k`$ for this step.** AR($`k`$) at depth $`\ell`$ requires $`\ell \ge k`$, so the early band $`[0, \lfloor L/3 \rfloor)`$ has only $`\lfloor L/3 \rfloor - k`$ depths where the fit is feasible. Require at least 3 valid depths in the early band — the across-band comparison needs per-band samples, not just existence at a single depth — which caps $`k`$ at $`\lfloor L/3 \rfloor - 3`$. On Pythia-410m ($`L=24`$) this is $`k \le 5`$; on Gemma-2-2B ($`L=26`$) also $`k \le 5`$; on Llama-3.1-8B ($`L=32`$) the cap is $`k \le 7`$ but the Phase 1 sweep $`\lbrace 1, 2, 3, 5, 10, 15, 20\rbrace`$ has no value between 5 and 10, so the largest in-sweep value satisfying the cap is still $`k = 5`$ — the same effective value used on Pythia and Gemma, not 7. If the Granger-selected lag exceeds the cap — including the "no clear cutoff" case where Granger remains significant at the largest lag tested — use the largest value in the Phase 1 sweep that satisfies the cap ($`k = 5`$ on all three target models) and note explicitly in the Gate 2 memo that Step 1 tests only short-horizon stationarity; long-horizon stationarity (lags beyond the cap) is deferred to the Phase 3 Step 4 MA-recheck on the fitted generative model, which does not need a per-band depth quota. Phase 2 Step 2 below will refine the effective depth $`k^\ast`$; rerun this Step 1 with $`k = \min(k^\ast, \lfloor L/3 \rfloor - 3)`$ only if that value differs materially from the provisional setting. Lock $`r`$ at the saturation rank identified in Phase 1 Step 3 (the smallest rank whose ridge $`R^2`$ is within ~5% of the asymptote at the largest rank swept); if the Phase 1 saturation rank is layer-dependent, take the median across layer pairs. **This $`r`$ choice is an engineering heuristic, not a derivation.** Phase 1 Step 3 fits ridge on $`v_\ell \to \Delta_\ell`$, while Step 1 here fits AR on $`\Delta_{\ell-1:\ell-k} \to \Delta_\ell`$ — different signal subspaces in general, and [proposal §4 H1](../README.md#4-hypothesis) explicitly notes the two tests neither entail nor rule each other out. The Phase 1 saturation rank is reused only because running a separate AR-rank sweep before the across-band comparison would double Phase 2's compute budget. **Spot-check at the locked $`r`$:** before computing the across-band Frobenius / principal-angle / cross-band $`R^2`$ diagnostics, fit the pooled AR($`k`$) kernel of [Phase 1 Step 5](phase-1-diagnostic.md#step-5--ark-on-delta-and-granger-causality-h1-full-form) at the locked $`r`$ and at $`2r`$, and compare held-out $`R^2`$. If the $`2r`$ fit is materially better (>5% absolute $`R^2`$ gain), rerun Step 1 with $`r`$ doubled and report the larger value as the working rank; otherwise the locked $`r`$ is fine.
3. Compare the three kernels. Four diagnostic checks:
   - **Frobenius distance** between kernel matrices, normalised by Frobenius norm. The AR($`k`$) kernel is a stack $`\lbrace W_j\rbrace_{j=1}^{k}`$; compute the distance both per-lag $`j`$ and on the concatenated stack, and report both.
   - **Subspace overlap:** SVD each kernel, compare the top $`r`$ singular vectors via principal angles. Small angles → same subspace acts differently; large angles → different subspaces.
   - **Cross-band prediction:** take the kernel fit on early layers and use it to predict middle-layer updates. Report the $`R^2`$ degradation relative to the within-band fit.
   - **Bootstrap test:** bootstrap examples within each band and compute the sampling distribution of the within-band kernel. Is the between-band distance larger than the within-band sampling variability?

**Decision matrix.**
- All three bands statistically indistinguishable (small Frobenius distance, small principal angles, cross-band $`R^2`$ nearly equal to within-band, between-band distance within bootstrap noise): **stationarity holds.** Phase 3 uses a single block-Toeplitz-constrained kernel.
- Systematic drift (kernels change smoothly with depth): **stationarity fails smoothly.** Phase 3 uses a depth-indexed kernel, or equivalently a state-space model $`z_{\ell+1} = f_\ell(z_\ell)`$ with $`f_\ell`$ varying slowly in $`\ell`$ ($`z`$ rather than $`s`$ — see [Phase 3 Branch B](phase-3-decomposition.md#step-1--lock-the-architecture-based-on-gate-2) for why).
- Clean phase transition (two or three kernels each homogeneous internally, different between): **segmented stationarity.** Phase 3 uses one kernel per segment. Record the segment boundaries for the paper — "when does the transformer change computational regime with depth?" is an independently interesting question.

## Step 2 — Effective kernel depth

**Question.** How many layers back do earlier updates meaningfully contribute to the prediction?

**Procedure.**
1. Fix the decomposition from Step 1 (one kernel, depth-indexed, or per-segment — whichever Step 1 chose, matching the three outputs of Step 1's decision matrix).
2. For kernel lengths $`k \in \lbrace 1, 2, 3, 5, 10, 15, 20\rbrace`$, refit and measure held-out reconstruction $`R^2`$.
3. Define the effective depth $`k^\ast`$ as the smallest $`k`$ whose $`R^2`$ is within some tolerance (say, 95%) of the asymptote at the maximum $`k`$.
4. Report $`k^\ast`$ as a function of depth (if stationarity failed) and as a function of relation class (Step 4).

**Note:** $`k^\ast`$ is itself a headline empirical number. "Residual updates at layer $`\ell`$ depend on updates from roughly the $`k^\ast`$ preceding layers" is a statement the field does not currently have a number for. Even if nothing else in this project works, reporting $`k^\ast`$ carefully across models is useful.

## Step 3 — Token-position invariance

**Question.** The $`\Delta_\ell`$ trajectories collected in Phase 1 were sampled at a single token position — the final prompt token. Does the AR structure found there also hold at other token positions, or is it specific to that one position?

**Why it matters.** Attention at every layer pulls in information from other tokens, so the final-prompt-token trajectory folds in contributions from the entire context. If the AR kernel is specific to the final position — e.g. because context aggregation is concentrated there — then what the kernel really characterises is *context-aggregated factual recall at the query position*, not a general layer-to-layer update dynamics. The token-position-invariance check decides which of these two interpretations the data supports, and it is a direct test of the attention-mixes-across-tokens caveat flagged in [proposal §4.3](../README.md#43-where-the-emg-intuition-sits--and-where-it-stops).

**Procedure.**
1. For the same factual-recall examples used in Phase 1, recollect residual-stream trajectories at additional token positions, not only the final one. Store under the same HDF5 layout with an extra `token_pos` index.
2. Choose three deterministic probe positions per example in addition to the final prompt token: (a) the first subject token, (b) the last subject token, (c) a mid-prompt non-subject token. Record the selection rule so it is reproducible. For single-token-subject examples (a) and (b) coincide; partition the dataset into multi-token-subject and single-token-subject subsets and report their sizes — the 4×4 confusion below runs on the multi-token subset (where (a) and (b) are distinct), and a parallel 3×3 confusion (with (a) = (b) merged) runs on the single-token subset.
3. For each probe position independently, refit the AR($`k^\ast`$) kernel from Step 2 on trajectories at that position only, using the same low-rank $`r`$ and the same held-out split.
4. Compare the four kernels (final-prompt-token from Phase 1 plus the three above) with the Step 1 diagnostics: Frobenius distance, subspace overlap via principal angles, cross-position prediction $`R^2`$.
5. On the multi-token-subject subset, report a 4×4 cross-position confusion matrix — kernel trained at position $`p`$, evaluated at position $`q`$. The diagonal is the within-position fit; off-diagonal drop is the position dependence. On the single-token-subject subset, report the parallel 3×3 confusion with (a) and (b) merged and flag any qualitative difference between the two subsets in the Gate 2 memo.

**Interpretation.**
- Kernels near-identical across positions, small off-diagonal $`R^2`$ drop: AR structure is a property of the depth dynamics, not of a specific position. The convolutive picture is defensible without qualification.
- Final-prompt-token kernel differs systematically, with large off-diagonal drop: the Phase 1 dynamics are specific to the query position — where, plausibly, most context aggregation happens. The framing narrows to "context-aggregated factual recall at the query position," still a real phenomenon but a more honest scope.
- Kernels look random across positions: the original AR signal was thin; rerun Phase 1 with more examples per position before proceeding.

**Cost estimate.** Adding ~3 probe positions per example roughly quadruples activation storage (~4 GB on Pythia-410m) and re-collection compute, but does not change the downstream analysis pipeline.

## Step 4 — Generalisation across relation types

**Question.** Are the dynamics a property of the transformer (universal across all factual-recall examples) or of the specific circuit being recruited (different for dates vs. geography vs. people)?

**Procedure.**
1. Split the dataset by `relation_class` into stratified subsets: at least 3 classes with ≥ 1,000 examples each.
2. Refit the kernel within each class (using the architecture decided in Step 1).
3. Cross-apply: take the geography-fit kernel and predict date-relation updates; report $`R^2`$ degradation. All pairs.
4. Plot a confusion-style matrix of cross-class reconstruction quality.

**Interpretation.**
- Kernels near-identical across classes: cross-layer dynamics are a *general* property of the residual stream, not circuit-specific. Strong result — it argues the kernel reflects architectural priors rather than learned task structure.
- Kernels systematically differ: different circuits propagate differently. This is expected and points toward "circuit-class-specific convolutive structure," which is weaker than a universal kernel but still stronger than "no structure."
- Kernels look random across classes: the cross-layer structure from Phase 1 may not be robust to data changes; treat carefully and rerun diagnostics with more examples per class.

## Step 5 — Cross-model replication

**Question.** Does the characterisation from Steps 1–3 hold on a different model family?

**Procedure.**
1. Repeat Steps 1, 2 and 4 on **[Gemma-2-2B](../glossary.md#gemma-scope)** with matching factual-recall examples (stationarity, effective depth, cross-class generalisation). **Prerequisite:** Phase 1 Steps 3 and 5 must also run on Gemma activations first — Phase 2 Step 1's feasibility cap depends on $`L`$ (satisfied trivially on Gemma) but its $`k`$ selection reads from a Gemma-specific Granger profile, so the Pythia Granger profile cannot be reused. Collect Gemma residual-stream trajectories, fit the ridge diagnostic and TL baseline, run the AR($`k`$) + Granger sweep, and clear the Step-5 TL gate on Gemma (waive-and-report if no Tuned Lens is available for Gemma-2-2B) before the Phase 2 kernels are fit. Gemma-2-2B has a larger residual dimension (2304) and more layers (26), so absolute kernel sizes differ, but the *characterisation* is directly comparable. The token-position check (Step 3) is optional cross-model — include it only if compute permits, since it roughly quadruples activation collection.
2. If compute permits, repeat the linear and AR tests only (Steps 3 and 5 of Phase 1) on Llama-3.1-8B to get a three-model comparison. Step 5 has a Tuned-Lens prerequisite on every model where it is run (ridge from Step 3 must beat the TL-implied update); on Llama-3.1-8B either use a public Tuned Lens if one is available for this exact checkpoint, or train one as part of this cross-check, or waive the TL gate for the Llama run and report that waiver explicitly in the Gate 2 memo — do not silently drop the gate.

**Reporting.** For each (model, characterisation property) pair, a single-row summary: stationary Y/N, $`k^\ast`$, strongest cross-class transfer pair, weakest. These rows become a headline table in the paper.

## Step 6 — Gate 2 decision

Write a Gate 2 memo answering:

1. **Does stationarity hold?** (Step 1) → architecture choice for Phase 3.
2. **What is the effective depth $`k^\ast`$?** (Step 2)
3. **Is the AR structure token-position-invariant, or specific to the query position?** (Step 3)
4. **Is the structure circuit-general or circuit-specific?** (Step 4)
5. **Does any of this replicate cross-model?** (Step 5)

Based on Step 1:
- **Stationarity holds → Phase 3 uses a block-Toeplitz-constrained convolutive decomposition** (fit $`\Delta_\ell \approx \sum_{k=0}^{K} W_k s_{\ell-k}`$ with shared $`W_k`$ and per-layer sparse sources).
- **Stationarity fails smoothly → Phase 3 uses a state-space model** ($`z_{\ell+1} = A_\ell z_\ell + B_\ell u_\ell`$ with $`A_\ell`$ smoothly varying or low-rank parameterised; $`z`$ rather than $`s`$ because the Branch A sources $`\lbrace s_{\ell'}\rbrace`$ occupy that symbol already — see [Phase 3 Branch B](phase-3-decomposition.md#step-1--lock-the-architecture-based-on-gate-2)).
- **Segmented stationarity → Phase 3 uses one kernel per segment** with the segment boundaries taken from Step 1.

**Deliverable artefacts at Gate 2:**
- `runs/gate2/memo.md`
- `runs/gate2/figures/` — stationarity comparison, $`k^\ast`$ curves, cross-class confusion, cross-model table.
- Decision recorded: `runs/gate2/phase3_architecture.md` — one paragraph specifying the exact decomposition form that Phase 3 will implement.
