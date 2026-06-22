# Structured Cross-Layer Propagation of Features in Transformers

**A research proposal — mechanistic interpretability / model internals**

**Expected output:** a public paper / preprint, plus reproducible code

> **Navigation**
> - This document — the research statement.
> - [Implementation plan](implementation-plan/README.md) — phased, step-by-step experimental protocol with go/no-go gates.
> - [Glossary](glossary.md) — every technical term and every cited paper is listed with a plain-language entry. No prior knowledge is assumed. Terms in the text below link directly to glossary entries.

---

## 1. One-paragraph summary

Current mechanistic interpretability tools decompose transformer activations layer by layer. They find a lot, but they make an implicit assumption worth questioning: that a [feature](glossary.md#feature) looks roughly the same wherever it appears. I propose to test a stronger hypothesis — that the *layer-to-layer updates* in the [residual stream](glossary.md#residual-stream) (Elhage et al., 2021) have structured, predictable dynamics across depth, and that this structure is well modeled as a sparse, convolutive process along depth. The project is an empirical investigation on factual-recall circuits in open-weight models ([Pythia](glossary.md#pythia), [Gemma](glossary.md#gemma-scope), small [Llama](glossary.md#llama) — specifically Llama-3.1-8B, if compute permits), with explicit go/no-go gates, causal ablation-based evaluation, and a pre-registered "the hypothesis is wrong" exit path that is itself a publishable result.

---

## 2. Motivation: why this specific hypothesis

The hypothesis below is motivated by an analogy to **convolutive blind source separation**, the problem studied in high-density [surface EMG](glossary.md#surface-emg) decomposition. There, you record a mixture of signals at many electrodes, each source contributes a characteristic spatio-temporal pattern, the mixing is convolutive rather than instantaneous, and you must recover the individual sources and their firing patterns without knowing the mixing kernels in advance. The canonical technique for this setting is [Convolution Kernel Compensation (CKC)](glossary.md#ckc) (Holobar & Zazula, 2007).

This analogy is the *source* of the hypothesis, not a plan to literally import an EMG algorithm into an LLM. The formal model tested here is framed in native interpretability and signal-processing terms (sparse coding with a depth-indexed dictionary, autoregressive models on residual updates, [Granger causality](glossary.md#granger-causality), low-rank kernel estimation). The convolutive-source-separation work is the source of the intuition; it is not the formal model.

---

## 3. The problem: decomposing a mixture whose parts overlap

[Mechanistic interpretability](glossary.md#mech-interp) begins with a simple empirical fact: a single neuron in a large language model rarely corresponds to a single human-legible concept. Instead, concepts are represented as directions in activation space, and many more concepts are represented than there are neurons. This is the [superposition hypothesis](glossary.md#superposition) (Elhage et al., 2022). The same research programme gave us the [residual-stream framing](glossary.md#math-framework) that every cross-layer method — including this one — relies on: transformer components communicate by reading and writing to a shared additive channel, and depth-wise feature persistence is a consequence of that channel being a running sum (Elhage et al., 2021). The dominant way to recover these directions is a [sparse autoencoder (SAE)](glossary.md#sae) — you train a wide, L1-regularised autoencoder on activations from a single layer and interpret its latents as features (Cunningham et al., 2023; Bricken et al., 2023; Templeton et al., 2024).

Per-layer SAEs work, but they answer only a per-layer question: *"what features live here?"* They do not answer *"how does this feature travel through depth?"* — and that second question is the interesting one for circuit analysis, causal editing, and any safety application that needs to predict what happens downstream of an intervention.

The field has begun moving on this. Three recent directions are directly relevant:

- **[Sparse crosscoders](glossary.md#crosscoder)** (Lindsey et al., 2024) fit a single dictionary of features across *all* layers at once. The same latent can fire in many layers; the decoder spreads it into the residual stream at each layer with a learned weight. Crosscoders make the persistence of features across depth visible and measurable.
- **[Multi-layer SAEs (MLSAE)](glossary.md#mlsae)** (Lawson et al., 2024) pursue the same goal with shared-dictionary training. Their finding is the one I want to flag: *individual MLSAE latents are often active on only one layer for a given token, but which layer varies across tokens and prompts.* This is in tension with a naive picture where a feature simply "lives at depth ℓ" — it complicates both the per-layer and the cross-layer story.
- **[Cross-Layer Transcoders (CLT)](glossary.md#clt)** (Ameisen et al., 2025 — the "Circuit Tracing" work that is the substrate for attribution graphs on a production model). Each CLT feature reads from the residual stream at a single source layer $`\ell'`$ and *writes* a contribution into the output of every MLP layer at or after the source, $`\ell \in \lbrace \ell', \ell'+1, \ldots, L-1\rbrace`$ (using the 0-indexed convention $`\ell \in \lbrace 0, 1, \ldots, L-1\rbrace`$ fixed in the [implementation plan](implementation-plan/README.md) for a model with $`L`$ blocks) via a **separate, fully-free decoder vector $`d^{\ell' \to \ell}`$ per target layer**. Stacked over the dictionary of features rooted at each source $`\ell'`$, this amounts to a family of decoder matrices $`W_{\text{dec}}^{\ell' \to \ell}`$ — one per (source, target) pair, with *no* prior tying the matrix at one depth pair to the matrix at another. Counting across the stack, there are $`O(L^2)`$ such matrices (one per ordered pair with $`\ell' \le \ell`$); per individual feature there are $`O(L)`$ decoder vectors (one per target layer at or after the source). CLT is therefore the maximally expressive member of this family — it can in principle fit any depth-asymmetric, feature-to-feature interaction — at the cost of a large parameter budget and no structural interpretation of how parameters vary with depth.

Two further threads matter for framing:

- **[Jacobian Sparse Autoencoders (JSAE)](glossary.md#jsae)** (Farnik et al., 2025) sparsify the Jacobian of MLP computations in the SAE feature basis — targeting per-component *causal* layer-to-layer sensitivities rather than correlational structure. JSAE does not assume or test any shared structure across depth.
- **[RouteSAE](glossary.md#routesae)** (Shi et al., 2025) attaches a lightweight router to a shared SAE that dynamically picks which layer's residual-stream content to decode, per token. It tackles the observation — highlighted by Lawson et al.'s MLSAE finding — that a single feature can appear at different depths in different contexts. It is not a propagation-kernel model.

All of these methods reveal that depth is not a nuisance to average over. My hypothesis is more specific than any of them: it proposes a particular *parameter-sharing structure* across depth that none of them assume or test.

---

## 4. Hypothesis

**As information propagates through transformer layers, the contribution each layer adds to the residual stream has structured, predictable dynamics as a function of depth.** Concretely, I propose two nested claims, weak-to-strong:

**(H1) Depth-local predictability.** For factual-recall inputs, the layer-to-layer update
$`\Delta_\ell = v_{\ell+1} - v_\ell`$
is predictable from the preceding few updates $`\Delta_{\ell-1}, \Delta_{\ell-2}, \ldots`$ by a low-rank linear map, above a null baseline on scrambled activations. That is, residual updates are an [autoregressive](glossary.md#autoregressive) process along depth. A cheaper related diagnostic — whether $`\Delta_\ell`$ is linearly predictable from $`v_\ell`$ alone — is run first in Phase 1 as a sanity check. It is not a formal precursor: ridge on $`v_\ell`$ applies a single map to the accumulated sum $`v_\ell = v_0 + \sum_{j \lt \ell} \Delta_j`$, whereas the AR model has separate weights on each lagged $`\Delta`$, so the two tests neither logically entail nor rule each other out. The diagnostic is informative (and in particular must out-perform the Tuned-Lens-derived baseline before it is worth running the AR test at all — see §5 item 1), but H1 proper is the AR claim.

**(H2) Approximate stationarity across depth.** Furthermore, the prediction kernel is approximately shared across depth — the same depth-filter predicts early-to-mid updates and mid-to-late updates — so that the generative model
$$\Delta_\ell \approx \sum_{k=0}^{K} W_k \, s_{\ell-k}$$
with $`\lbrace s_{\ell'}\rbrace_{\ell'=0}^{L-1}`$ sparse source representations, one per source layer (convention: $`s_{\ell-k} = 0`$ for $`\ell-k \lt 0`$), and $`\lbrace W_k\rbrace`$ a low-rank, [block-Toeplitz-constrained](glossary.md#toeplitz) depth kernel (each $`W_k`$ a $`D \times M`$ matrix; the "block" qualifier because the shared-along-depth entries are matrices, not scalars), is a useful approximation of the first-order depth dynamics. Fitting additionally includes an unpenalised per-layer bias $`b_\ell`$ that absorbs the layer-wise mean $`\bar\Delta_\ell`$ (see [Phase 3 Step 3](implementation-plan/phase-3-decomposition.md#step-3--loss-and-optimisation-branch-a--c)); the bias is an engineering detail that lives outside the kernel and does not affect the stationarity claim on $`\lbrace W_k\rbrace`$, which is a claim about transition structure and not about layer-wise offsets.

H1 is a *much* weaker claim than H2. H1 says "layers don't just add independent noise to the residual stream; the updates have cross-layer structure." H2 says "the structure is simple enough to be modelled by a single kernel across depth." **I treat stationarity as a hypothesis to be tested, not a premise.** If stationarity fails, H1 may still hold, and we learn something: the transformer's depth dynamics are non-stationary, a [state-space model](glossary.md#ssm) is the right abstraction, and CLT-style methods that make no stationarity assumption are already capturing the relevant structure.

### 4.1. Why model $`\Delta_\ell`$ and not $`v_\ell`$ directly

The residual stream is additive: $`v_{\ell+1} = v_\ell + \Delta_\ell`$. If you regress $`v_{\ell+1}`$ on $`v_\ell`$ you trivially recover approximately the identity, and the non-trivial content — what each layer *adds* — is buried under the residual-connection baseline. Modelling $`\Delta_\ell`$ directly strips that baseline out and lets any structure in the updates speak clearly. This is the same reason differencing is standard in classical time-series analysis: you subtract the trend to see the dynamics.

### 4.2. Relationship to Cross-Layer Transcoders — the novelty claim

CLT (Ameisen et al., 2025) is the closest existing method to what I am proposing, and the difference is cleanest to state in parameter-counting terms.

CLT places a free decoder matrix $`W_{\text{dec}}^{\ell' \to \ell}`$ at every (source, target) pair with $`\ell' \le \ell`$. That is $`O(L^2)`$ independent decoder matrices across the stack, each of shape $`D \times M_{\ell'}`$ (residual dimension × dictionary width at source $`\ell'`$), with *no prior* connecting the matrices at different depth pairs. Under H2, these $`O(L^2)`$ matrices collapse to a single kernel indexed by the depth *difference* $`k = \ell - \ell'`$, shared across every position $`\ell`$ in the stack, and vanishing for $`k \gt K`$ (the effective kernel depth):

$$W_{\text{dec}}^{\ell' \to \ell} = W_{k}, \qquad k = \ell - \ell', \qquad W_k = 0 \text{ for } k \gt K.$$

This is a lower-triangular [block-Toeplitz constraint](glossary.md#toeplitz) along depth — "block" because the repeated entries are matrices ($`W_k \in \mathbb{R}^{D \times M}`$), not scalars, and "lower-triangular" because the structure is causal: layer $`\ell`$ depends only on contributions written at $`\ell' \le \ell`$. It reduces the total count of distinct transition matrices from $`O(L^2)`$ to $`O(K)`$ with $`K \ll L`$, and it imposes a strong prior: *the contribution a source layer writes into a target layer depends only on how many layers ago the source fired, not on where in the stack source and target happen to sit.*

The hypothesis is therefore best read as a CLT-falsifier in a very specific sense. It asks whether CLT's free-parameter budget is load-bearing or redundant.

- **H2 holds, my decomposition reconstructs comparably to CLT.** The $`O(L^2)`$ CLT matrices were mostly compressible along a depth-difference axis, and the transition dynamics are approximately a stationary convolution through depth. My method becomes a parsimonious reparameterization of a subspace CLT can already discover, with much fewer parameters and a single interpretable kernel shape along depth.
- **H2 fails, my decomposition reconstructs badly.** CLT's free parameters are doing real work; the dynamics are genuinely non-stationary; and "shared kernel along depth" is the wrong abstraction. This is an *independent validation* of CLT's unconstrained parameterization and a useful constraint on theories of how information flows through transformer depth.

Either way the field learns something. In the first case, we gain a much tighter vocabulary for writing down cross-layer circuits. In the second, we gain a principled reason to stop looking for depth-shared structure. Note that the comparison is not fully apples-to-apples: CLT writes into MLP-block outputs while the decomposition here targets residual updates $`\Delta_\ell`$ that fold in both attention and MLP contributions. I will revisit this in the evaluation (§6) by running the CLT comparison against residual-stream-write variants.

### 4.3. Where the EMG intuition sits — and where it stops

The intuition — that a feature might have a *characteristic propagation pattern through depth*, the way a motor unit's action potential has a characteristic propagation pattern through muscle tissue — is where the project comes from. But the formal model must be justified on its own terms in the transformer. Concretely, three ways the analogy is imperfect, each of which the experimental design has to respect:

- **Nonlinearity.** In EMG, the convolutive mixing model is grounded in the approximately linear time-invariant physics of the tissue volume conductor. In a transformer, each layer is nonlinear (MLP, attention, normalisation). **Block-Toeplitz structure is an approximation, not a derivation.** The empirical question is whether it is a useful approximation — whether imposing it reveals structure that unconstrained methods miss.
- **Attention mixes across tokens, not just across depth.** EMG signals at any single channel have a well-defined temporal axis; the transformer's "depth axis" at a single token is not purely local, because attention at every layer pulls in information from other token positions. The residual update $`\Delta_\ell`$ at the final prompt token therefore depends on the residual states of all other tokens at layer $`\ell`$, not just on the target token's own depth history. This means the cross-layer dynamics I characterise may be as much a signature of attention-mediated context reconstruction as they are of depth-intrinsic propagation. Two checks aimed directly at this caveat: Phase 1's context-perturbation null test (§5, point 2) asks whether the AR structure survives a swap of surrounding tokens, and Phase 2's token-position-invariance test ([implementation plan, Phase 2 Step 3](implementation-plan/phase-2-characterization.md#step-3--token-position-invariance)) asks whether the same kernel fits trajectories collected at non-final token positions. The combination decides whether the characterisation is depth-intrinsic, context-reconstructive, or specific to the query position.
- **Sources as latent features, not spikes.** In EMG, sources are spike trains with clear physical meaning. In a transformer, "sources" are latent features. I place sparse sources $`s_{\ell'}`$ at every source layer $`\ell' \in \lbrace 0, 1, \ldots, L-1\rbrace`$ (initialised from per-layer SAE latents; one source per residual-stream state that can *write into* a subsequent update), mirroring CLT's source-at-every-layer parameterisation and leaving the stationarity of the decoder kernel $`\lbrace W_k\rbrace`$ as a separately testable claim. Prior factual-recall work (Geva et al., 2023) shows that subject-enrichment and relation-binding features are *built up* in early-to-middle layers, not present at layer 0 — so a single-source-at-embedding variant (all $`s_{\ell'} = 0`$ for $`\ell' \gt 0`$) would almost certainly under-fit late-layer updates; that variant is reported as an ablation lower bound, not as the main model. The real go/no-go for the convolutive form is H2 (stationarity). The proxy test at Gate 2 fits AR on $`\Delta`$ kernels on early, middle, and late depth bands and asks whether they agree; if the per-band kernels diverge systematically, a state-space model with depth-varying dynamics is the right abstraction, not a shared convolution. Gate 2 cannot rule stationarity *in* on its own (the AR-to-MA direction is not logically tight — see [Phase 2 Step 1](implementation-plan/phase-2-characterization.md#step-1--stationarity-test)); the generative test — per-layer sources plus a shared depth kernel reconstructing $`\Delta_\ell`$ well — is run in Phase 3 Step 4 once the decomposition has been trained.

The EMG analogy generates the hypothesis. The hypothesis must survive on its own.

---

## 5. Experimental strategy

The full step-by-step protocol is in [implementation-plan/README.md](implementation-plan/README.md). Here I summarize the logic.

**Choice of inputs: factual-recall circuits.** I do not test the hypothesis on generic text. I test it on **subject-relation-object triples** where the model reliably knows the answer (e.g. *Paris — capital of — France*, *Einstein — born in — Germany*). This gives three properties that generic-text probing does not: (i) clean ground truth, (ii) *localised circuits already identified by prior work* ([ROME](glossary.md#rome), Meng et al., 2022; [knowledge neurons](glossary.md#knowledge-neurons), Dai et al., 2022; Geva et al., 2023), and (iii) a setting where causal interventions have an established, measurable meaning. Without a circuit with known causal semantics, "we found a feature" risks becoming a narrative exercise. One caveat I take seriously: Hase et al. (2023) showed that "where a fact is causally traced" and "where editing a fact works" are not the same location, so I treat the ROME/Geva layer ranges as strong priors for where to *look* rather than as settled ground truth.

**Four empirical layers, in order.**

1. **Linear depth-local predictability** (cheap diagnostic, not a formal precursor to H1). Fit $`\Delta_\ell \approx W v_\ell`$ with ridge regression under a [low-rank](glossary.md#low-rank) constraint, one model per consecutive layer pair. Compare $`R^2`$ against the zero-baseline $`\Delta_\ell = 0`$ *and* against what [Tuned Lens](glossary.md#tuned-lens) (Belrose et al., 2023) already predicts — the lens already shows that layers progressively refine the output distribution, so a simple per-layer refinement signal is not in itself evidence of anything novel. The diagnostic signal must out-predict Tuned Lens's implied per-layer update before it is worth running the AR test at all. Note that Tuned Lens is trained to produce per-layer logit predictions, not residual-update predictions. The "implied per-layer update" is defined as $`\hat r_{\ell+1} - \hat r_\ell`$, where $`\hat r_\ell = T_\ell v_\ell + c_\ell`$ is the lens's aligned-residual adapter output (the translator $`T_\ell`$ and bias $`c_\ell`$ are the Tuned Lens's learned per-layer affine parameters; $`T`$ rather than $`A`$ to avoid collision with the state-space matrices $`A_\ell`$ in [Phase 3 Branch B](implementation-plan/phase-3-decomposition.md#step-1--lock-the-architecture-based-on-gate-2), and $`c`$ rather than $`b`$ to avoid collision with the per-layer decomposition bias $`b_\ell`$ in [Phase 3 Step 3](implementation-plan/phase-3-decomposition.md#step-3--loss-and-optimisation-branch-a--c)). At the top of the stack, where standard TL releases supply no $`T_L, c_L`$, take $`\hat r_L \equiv v_L`$ (equivalently $`T_L = I`$, $`c_L = 0`$) — TL is trained to align intermediate states toward the final residual, so the lens at $`\ell = L`$ is the identity for free; the implementation plan ([Phase 1 Step 3](implementation-plan/phase-1-diagnostic.md#step-3--linear-depth-local-predictability-h1-weak-form)) fixes this convention so the layer-averaged gate runs over all $`L`$ pairs uniformly. This lives in residual space, *before* the unembedding — so no inversion of the unembedding is needed. This is an indirect baseline for two reasons — TL is trained for a different objective, and the diff uses $`v_{\ell+1}`$ that the ridge fit on $`v_\ell`$ alone does not — and should be reported as such. Add [canonical correlation analysis (CCA)](glossary.md#cca) as a nonlinear-sensitive backstop: if CCA finds shared subspaces where ridge does not, there is nonlinear structure ridge is missing. Use shrinkage CCA given the residual dimension (overfitting is otherwise expected). This test shares no formal implication relation with H1 in either direction (see §4 H1) — it is run first because it is cheap and because a positive result motivates the AR test; H1 proper is the AR on $`\Delta`$ claim tested in item 2.
2. **Autoregressive depth structure and Granger causality** (H1, full form). Fit $`\Delta_\ell \approx \mu_\ell + \sum_{j=1}^k W_j \Delta_{\ell-j}`$ with increasing $`k`$ — per-depth intercepts $`\mu_\ell`$ alongside a depth-shared kernel $`\lbrace W_j\rbrace`$, so the kernel captures the AR dynamics independent of per-depth mean offsets in $`\bar\Delta_\ell`$. The intercept is named $`\mu`$, not $`c`$, because $`c_\ell`$ is the Tuned-Lens bias introduced in item 1 above; $`\mu`$ rather than $`b`$ for the same reason $`c`$ avoided $`b`$ — $`b_\ell`$ is the Phase 3 decomposition bias. The implementation plan ([Phase 1 Step 5](implementation-plan/phase-1-diagnostic.md#step-5--ark-on-delta-and-granger-causality-h1-full-form)) explains why omitting $`\mu_\ell`$ would contaminate the downstream Phase 2 stationarity diagnostic. Use [Granger causality](glossary.md#granger-causality) to determine how far back in depth earlier layers meaningfully contribute. Two null controls: (a) **scrambled activations** — shuffle examples independently at each depth to destroy within-example depth structure, and (b) **context perturbation** — swap surrounding tokens (not the target token) and test whether the Granger structure changes. (b) is the critical control: if it *does* change, the apparent cross-layer structure is really cross-context information leaking through hidden variables, and we learn the dynamics are *context-reconstructive*, not *depth-intrinsic*.
3. **Stationarity** (H2). Fit separate kernels on early, middle, and late depth bands (non-overlapping thirds of the layer stack; the exact partition and feasibility cap on the AR order are in [Phase 2 Step 1](implementation-plan/phase-2-characterization.md#step-1--stationarity-test)). Statistically compare them. If similar, block-Toeplitz structure is justified. If they differ systematically, state-space models are the right abstraction. If there is a clean phase transition at a specific depth, a segmented kernel is the right abstraction. *This is a go/no-go gate for the stationary-kernel decomposition in the decomposition step below.*
4. **Effective kernel depth.** For kernel lengths $`k \in \lbrace 1, 2, 3, 5, 10, 15, 20\rbrace`$ (the exact sweep used in [Phase 2 Step 2](implementation-plan/phase-2-characterization.md#step-2--effective-kernel-depth)), observe when reconstruction error stops falling. The sweep is capped below the model's layer count — Pythia-410m has 24 blocks, Gemma-2-2B has 26, so $`k \le 20`$ leaves headroom and $`k = 50`$ would be infeasible on any of the target models. The "effective depth" — how many layers back carry relevant information — is itself an interesting empirical number. I will compare it across relation types (geography, dates, people) and across models (Pythia vs. Gemma vs. Llama) to see whether it is a property of transformers in general or of specific circuits.

**Decomposition step.** If H1 survives, and conditional on whether H2 survives, I fit either:
- a stationary depth kernel $`\Delta_\ell \approx \sum_{k=0}^{K} W_k s_{\ell-k}`$ with per-source-layer sparse sources $`\lbrace s_{\ell'}\rbrace_{\ell'=0}^{L-1}`$ and $`W_k`$ low-rank (if H2 holds), or
- a state-space model $`z_{\ell+1} = f(z_\ell) + \epsilon_\ell`$ with similar sparsity and low-rank structure (if H2 fails). $`z`$ rather than $`s`$ here because $`\lbrace s_{\ell'}\rbrace`$ is already taken by the per-source-layer sparse sources in the stationary-kernel branch above; the SSM latent state is a different object.

Both are standard. Neither is CKC; the EMG analogy is motivation, not code reuse.

---

## 6. Evaluation

**Baselines.** For every evaluation I compare the features recovered by my method against:
- **Per-layer SAE** baselines using publicly available weights ([Gemma Scope](glossary.md#gemma-scope), [Pythia SAE checkpoints](glossary.md#pythia-sae)) wherever they cover the layers I need. Where coverage is missing, train minimal per-layer SAEs on the same activations rather than re-deriving the full published training recipe — see [Phase 4 Step 1](implementation-plan/phase-4-evaluation.md#step-1--baselines-precisely-specified).
- **Crosscoder / MLSAE** baselines where public weights exist; otherwise I replicate the published training recipe on the same activations I use for my method.
- **[Cross-Layer Transcoder (CLT)](glossary.md#clt)** (Ameisen et al., 2025) — the *primary* conceptual baseline, since [§4.2](#42-relationship-to-cross-layer-transcoders--the-novelty-claim) positions the hypothesis as a shared-kernel reparameterization of CLT's $`O(L^2)`$ decoder matrices and the H2-holds outcome is literally "my decomposition reconstructs comparably to CLT." If public CLT weights for a target model are available at evaluation time, use them; otherwise replicate the Ameisen et al. 2025 recipe at a scoped-down size on the same activations. Because native CLT writes into MLP outputs while the decomposition here targets residual updates $`\Delta_\ell`$ (which also fold in attention), run both a native MLP-output-write form and a residual-stream-write variant so the comparison is apples-to-apples — this is the "residual-stream-write variants" comparison promised in §4.2. If compute prevents training even a scoped-down CLT within the available compute budget, record that explicitly as a limitation and frame Crosscoder / MLSAE as the closest feasible proxy; do not silently drop the comparison.
- **[Tuned Lens](glossary.md#tuned-lens)** (Belrose et al., 2023) as a cross-depth baseline that is specifically designed to forecast the final prediction from intermediate residual streams.

**Headline metrics.** Reconstruction quality (MSE on $`\Delta_\ell`$), sparsity–reconstruction Pareto curves, and stability of recovered features across random seeds and across datasets. These are standard and not sufficient.

**The critical evaluation is causal.** Metrics-only comparisons produce methods-paper noise and I want to do better than that. The headline test is:

1. Find a feature recovered by my method that *is not* found by the per-layer SAE, crosscoder, or MLSAE baselines. Matching uses two criteria — cosine similarity over decoder directions *and* Pearson correlation of per-example activation patterns — with novelty defined as max match below threshold on *both* criteria; [Phase 4 Step 3](implementation-plan/phase-4-evaluation.md#step-3--the-causal-existence-proof-the-critical-result) fixes the thresholds. CLT (Baseline D above) is deliberately excluded from the novelty-matching set: under H2 my decomposition is by construction a shared-kernel reparameterization of CLT (see §4.2), so CLT overlap is the *expected* outcome, not a novelty failure, and requiring novelty against CLT would set an impossible bar by design. The CLT comparison is about reconstruction parsimony — comparable quality at a smaller parameter budget — not feature novelty.
2. Ablate it (or, symmetrically, activate it).
3. Check whether the model's behaviour changes as predicted: e.g. does ablating a candidate factual-recall feature cause the model to fail on exactly the subject-relation queries it should, and not on unrelated queries?

If yes: I have an existence proof that the method recovers something genuinely new and causally real. If no: the method is finding artefacts and I must diagnose why.

**What "the hypothesis is wrong" looks like, and why that is publishable.** If H1 fails — if layer-to-layer updates have no cross-layer structure above null — that is a *strong constraint* on our theory of how transformers process information through depth. It argues that CLT- and JSAE-style methods, which model layer-to-layer dynamics without assuming structure, are getting the relevant signal, and that any attempt at a global kernel is chasing noise. This is a result the field would want to know, it is a cheap result to produce (Phase 1 reaches the go/no-go gate), and it is a legitimate paper outcome. The go/no-go structure is not decorative — it is how I keep the project scientifically honest.

---

## 7. Why this matters for safety

Models routinely generate plausible-sounding explanations for their outputs that do not correspond to what actually happened internally. This **confabulation gap** between stated reasoning and actual computation is a direct alignment risk: a model whose explanations diverge from its actual computation cannot be reliably steered by intervening on its stated reasoning. Mechanistic interpretability exists in part to close this gap by giving us direct, causal access to the internal variables that matter.

Better decomposition tools help. If my method recovers features that per-layer methods miss, and those features have testable causal effects on behaviour, then downstream circuit analysis and attribution graphs (Ameisen et al., 2025, and related work) get a richer vocabulary to work with. If my method fails, the field learns that depth dynamics in transformers are too irregular for any single cross-layer abstraction to capture, and CLT-style per-pair approaches are the frontier.

I am not claiming this project solves alignment. I am claiming it is a well-scoped, empirically decidable question whose answer either way is useful to the people working on alignment, and that the signal-processing background described in §2 is a specific angle from which to ask it.

---

## 8. Feasibility

- The mathematical tools involved — ridge-regularised estimation, AR modelling, alternating optimisation of source and kernel estimates, low-rank approximation, [CCA](glossary.md#cca), and [Granger causality](glossary.md#granger-causality) — are all standard and available off the shelf.
- The public tooling ([TransformerLens](glossary.md#transformerlens), Gemma Scope, Pythia SAE checkpoints, Tuned Lens implementations) removes most of the boilerplate. The work is analysing activations rather than training SAEs from scratch, which shortens the setup considerably.
- The go/no-go gate at the end of Phase 1 is cheap. If H1 does not hold on Pythia-410m, that becomes clear early, and the project can pivot to writing up the negative result rather than continuing down a dead end.

---

## 9. References

Citations are keyed to the [glossary](glossary.md), where each paper has a short plain-language entry, a one-line contribution summary, and a link. Primary references:

1. Elhage, N., Nanda, N., Olsson, C., Henighan, T., Joseph, N., Mann, B., Askell, A., Bai, Y., Chen, A., Olah, C., et al. (2021). *A Mathematical Framework for Transformer Circuits*. Transformer Circuits Thread. [transformer-circuits.pub/2021/framework/index.html](https://transformer-circuits.pub/2021/framework/index.html). — [[glossary](glossary.md#ref-elhage-2021)]
2. Elhage, N., et al. (2022). *Toy Models of Superposition*. Transformer Circuits Thread. — [[glossary](glossary.md#ref-elhage-2022)]
3. Bricken, T., et al. (2023). *Towards Monosemanticity: Decomposing Language Models with Dictionary Learning*. Transformer Circuits Thread. — [[glossary](glossary.md#ref-bricken-2023)]
4. Cunningham, H., Ewart, A., Riggs, L., Huben, R., Sharkey, L. (2023). *Sparse Autoencoders Find Highly Interpretable Features in Language Models*. — [[glossary](glossary.md#ref-cunningham-2023)]
5. Templeton, A., et al. (2024). *Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet*. Transformer Circuits Thread. — [[glossary](glossary.md#ref-templeton-2024)]
6. Lawson, T., Farnik, L., Houghton, C., Aitchison, L. (2024). *Residual Stream Analysis with Multi-Layer SAEs*. — [[glossary](glossary.md#ref-lawson-2024)]
7. Lindsey, J., Templeton, A., Marcus, J., Conerly, T., Batson, J., Olah, C. (2024). *Sparse Crosscoders for Cross-Layer Features and Model Diffing*. Transformer Circuits Thread. — [[glossary](glossary.md#ref-lindsey-2024)]
8. Ameisen, E., Lindsey, J., Pearce, A., Gurnee, W., Turner, N. L., Chen, B., et al. (2025). *Circuit Tracing: Revealing Computational Graphs in Language Models*. Transformer Circuits Thread. [transformer-circuits.pub/2025/attribution-graphs/methods.html](https://transformer-circuits.pub/2025/attribution-graphs/methods.html). — [[glossary](glossary.md#ref-ameisen-2025)]
9. Meng, K., Bau, D., Andonian, A., Belinkov, Y. (2022). *Locating and Editing Factual Associations in GPT* (ROME). NeurIPS. — [[glossary](glossary.md#ref-meng-2022)]
10. Geva, M., Bastings, J., Filippova, K., Globerson, A. (2023). *Dissecting Recall of Factual Associations in Auto-Regressive Language Models*. EMNLP. — [[glossary](glossary.md#ref-geva-2023)]
11. Dai, D., Dong, L., Hao, Y., Sui, Z., Chang, B., Wei, F. (2022). *Knowledge Neurons in Pretrained Transformers*. ACL. — [[glossary](glossary.md#ref-dai-2022)]
12. Hase, P., Bansal, M., Kim, B., Ghandeharioun, A. (2023). *Does Localization Inform Editing? Surprising Differences in Causality-Based Localization vs. Knowledge Editing in Language Models*. NeurIPS. — [[glossary](glossary.md#ref-hase-2023)]
13. Belrose, N., et al. (2023). *Eliciting Latent Predictions from Transformers with the Tuned Lens*. — [[glossary](glossary.md#ref-belrose-2023)]
14. Nanda, N., Bloom, J. (2022). *TransformerLens*. — [[glossary](glossary.md#ref-transformerlens)]
15. Biderman, S., et al. (2023). *Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling*. ICML. — [[glossary](glossary.md#ref-biderman-2023)]
16. Lieberum, T., et al. (2024). *Gemma Scope: Open Sparse Autoencoders Everywhere All at Once on Gemma 2*. — [[glossary](glossary.md#ref-lieberum-2024)]
17. Farnik, L., Lawson, T., Houghton, C., Aitchison, L. (2025). *Jacobian Sparse Autoencoders: Sparsify Computations, Not Just Activations*. [arXiv:2502.18147](https://arxiv.org/abs/2502.18147), ICML 2025. — [[glossary](glossary.md#ref-farnik-2025)]
18. Shi, W., Li, S., Liang, T., Wan, M., Ma, G., Wang, X., He, X. (2025). *Route Sparse Autoencoder to Interpret Large Language Models*. [arXiv:2503.08200](https://arxiv.org/abs/2503.08200), EMNLP 2025 (main). — [[glossary](glossary.md#ref-shi-2025)]
19. Holobar, A., Zazula, D. (2007). *Multichannel Blind Source Separation Using Convolution Kernel Compensation*. IEEE Transactions on Signal Processing 55(9). — [[glossary](glossary.md#ref-holobar-2007)]

---

*Continue to: [Implementation plan](implementation-plan/README.md) · [Glossary](glossary.md)*
