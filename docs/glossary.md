# Glossary

A plain-language reference for every non-trivial term used in the [research proposal](README.md) and the [implementation plan](implementation-plan/README.md). Organised by topic. No prior knowledge is assumed — every entry is self-contained.

**How to read:** when the proposal links to `glossary.md#something`, that anchor is defined in this file. Paper citations have IDs of the form `ref-authoryear`; concept anchors are lowercase keywords.

**Contents**
- [Part A — Core interpretability concepts](#part-a--core-interpretability-concepts)
- [Part B — Transformer anatomy, in one page](#part-b--transformer-anatomy-in-one-page)
- [Part C — Statistical and mathematical tools](#part-c--statistical-and-mathematical-tools)
- [Part D — Motivation: EMG, motor units, and CKC](#part-d--motivation-emg-motor-units-and-ckc)
- [Part E — Interpretability methods and tools](#part-e--interpretability-methods-and-tools)
- [Part F — Cited papers](#part-f--cited-papers)
- [Part G — Methods considered and not cited](#further-cross-layer-methods)

---

## Part A — Core interpretability concepts

<a id="mech-interp"></a>
### Mechanistic interpretability

The research programme of trying to reverse-engineer, in terms of human-legible concepts and algorithms, what a neural network is actually computing inside. It is distinct from **behavioural evaluation** (which asks what the model does) and from **explainability methods** that produce post-hoc justifications (which often describe plausible-sounding reasons that may not match the actual computation). A useful analogy: a behavioural eval is like measuring a CPU's output; mechanistic interpretability is like tracing individual logic gates.

Why it matters for safety: if a model's stated reasoning diverges from its actual computation, steering its stated reasoning does not steer the model. Mechanistic interpretability aims to give direct, causal access to the variables that actually drive behaviour.

<a id="feature"></a>
### Feature (interpretability sense)

A "feature" is a direction in the model's activation space that corresponds to a legible concept — e.g. "the token is a French city," "the sentence is a question about dates," "the user is being rude." Features are not individual neurons; they are typically *combinations* of neurons, because models pack many more concepts than they have neurons (see [superposition](#superposition)). A [sparse autoencoder](#sae) is the standard tool for finding them.

<a id="superposition"></a>
### Superposition (the superposition hypothesis)

Observation: a single neuron in a large model almost never corresponds to a single legible concept. Instead, neurons participate in many features at once, and many features are encoded in overlapping combinations of neurons. The "hypothesis" part: this happens *because* the model has more features it wants to represent than it has neurons, and packing them into overlapping directions — with sparse activation patterns so they rarely collide — is an efficient solution. First articulated as a formal toy model in Elhage et al. 2022 ([ref](#ref-elhage-2022)).

Why it matters: if features were one-per-neuron, interpretation would be trivial ("what does neuron #4217 do?"). Superposition is the reason that question doesn't work and the reason we need decomposition methods at all.

<a id="confabulation"></a>
### Confabulation (in LLMs)

When a model produces a plausible-sounding explanation for its output that does not correspond to what actually happened in its forward pass. The model didn't "use the steps it describes" — it generated text that *looks like* a correct explanation. This is directly downstream of not having a transparent computation, and it is one of the reasons alignment needs interpretability: you cannot trust self-reports about reasoning without some external check.

---

## Part B — Transformer anatomy, in one page

<a id="transformer"></a>
### Transformer

The neural network architecture used by essentially all modern large language models. A transformer is a stack of identical "layers." Each layer reads a sequence of vectors (one per input token), transforms them, and passes them to the next layer. The key ingredients are **attention** (each token looks at every other token and pulls in information) and **MLP blocks** (per-token nonlinear transformations). Normalization layers and skip connections glue the pieces together.

<a id="math-framework"></a>
### Mathematical framework for transformer circuits (Elhage et al. 2021)

The first systematic treatment of how information flows through a transformer in terms of linear-algebraic reads and writes on a shared additive channel. It articulated the **residual-stream framing** that every cross-layer interpretability method since has inherited: the attention and MLP blocks don't overwrite the token state, they *read from* and *write to* a running sum, so different components communicate by sharing subspaces of that sum. It also decomposed attention into an output-value (OV) circuit and a query-key (QK) circuit, making the "what information is copied" and "where from" pieces analysable separately. Elhage et al. 2021 ([ref](#ref-elhage-2021)) is the conceptual substrate this proposal stands on: H1 and H2 are claims about how things written into the residual stream at one depth behave with respect to things written later.

<a id="residual-stream"></a>
### Residual stream

The central object of modern interpretability, formalised in [Elhage et al. 2021](#math-framework). Inside each transformer layer, the layer does not *replace* the token's representation — it *adds* to it. The running sum, as you go up the stack, is called the residual stream. Formally, if $`v_\ell`$ is the state of the residual stream after layer $`\ell`$, then schematically

$$v_{\ell+1} = v_\ell + \text{Attn}_\ell(v_\ell) + \text{MLP}_\ell(v_\ell)$$

(LayerNorm omitted for exposition: each target model in this proposal places normalisations inside each block, but the exact pattern differs. **Pythia** (GPT-NeoX-derived, like GPT-J and PaLM) uses Pre-LN with standard LayerNorm and computes attention and MLP **in parallel** from the same block input: $`v_{\ell+1} = v_\ell + \text{Attn}_\ell(\text{LN}_1(v_\ell)) + \text{MLP}_\ell(\text{LN}_2(v_\ell))`$. **Llama-3.1** uses Pre-LN with RMSNorm and runs attention and MLP **sequentially**: $`v_{\ell+1} = v_\ell + \text{Attn}_\ell(\text{RMSNorm}_1(v_\ell)) + \text{MLP}_\ell(\text{RMSNorm}_2(v_\ell + \text{Attn}_\ell(\text{RMSNorm}_1(v_\ell))))`$. **Gemma-2** uses RMSNorm in a sandwich (pre+post) pattern around each sublayer — both the attention and MLP outputs are themselves RMSNormed before being added back to the residual: $`h = v_\ell + \text{RMSNorm}_{\text{post,A}}(\text{Attn}_\ell(\text{RMSNorm}_{\text{pre,A}}(v_\ell)))`$ and $`v_{\ell+1} = h + \text{RMSNorm}_{\text{post,M}}(\text{MLP}_\ell(\text{RMSNorm}_{\text{pre,M}}(h)))`$. The simplified additive form at the top of this entry is for readability of the additive-channel framing; for actual $`\Delta_\ell`$ extraction in the implementation plan, read the values straight from [TransformerLens](#transformerlens) hooks at `blocks.{l}.hook_resid_post` rather than re-deriving from any of these formulas — the hook captures the post-block residual state regardless of the per-block pattern.)

The update added by layer $`\ell`$ is

$$\Delta_\ell = v_{\ell+1} - v_\ell.$$

Because the residual stream is additive, features tend to persist through depth (a feature written in an early layer is still present in later layers unless another layer writes something that cancels it). This is why cross-layer interpretability methods are natural: features live in the residual stream, not strictly in one layer.

**Why $`\Delta_\ell`$ is the object of interest in this proposal.** Regressing $`v_{\ell+1}`$ on $`v_\ell`$ trivially recovers the identity, because of the residual connection. Modelling $`\Delta_\ell`$ directly strips that baseline out and isolates what each layer adds.

**Caveat on "depth-local."** At a fixed token position, $`\Delta_\ell`$ is only depth-local in a soft sense: attention at every layer pulls in information from *other token positions* at the same depth. So the cross-layer dynamics of $`\Delta_\ell`$ at a single token mix in horizontal (cross-token) as well as vertical (cross-depth) structure. This is why Phase 1 of the experimental plan includes an explicit context-perturbation null test.

<a id="attention"></a>
### Attention head

The part of a transformer layer that lets tokens read from other tokens. Each "head" implements a small pattern-matching operation: roughly, "find the tokens most similar to me in some learned projection and copy some of their information back." A layer has many heads in parallel.

<a id="mlp"></a>
### MLP block

A small two-layer feedforward network applied independently to each token's residual-stream state. In most modern transformers, MLP blocks are where most of the parameters — and, arguably, most of the factual knowledge — live.

---

## Part C — Statistical and mathematical tools

<a id="ridge"></a>
### Ridge regression

Linear regression with an extra penalty on the size of the learned coefficients. Given inputs $`X`$ and targets $`y`$, ordinary least squares finds $`W`$ minimising $`\|y - Xw\|^2`$. Ridge minimises $`\|y - Xw\|^2 + \lambda \|w\|^2`$. The extra term stabilises the solution when the input features are correlated or when you have more features than examples — both common in high-dimensional activation data. It is the default tool for "is $`y`$ linearly predictable from $`x`$?" when $`x`$ is high-dimensional.

<a id="low-rank"></a>
### Low-rank matrix / low-rank approximation

A matrix $`W \in \mathbb{R}^{D \times D}`$ has "rank $`r`$" if it can be written as $`W = A B^\top`$ with $`A, B \in \mathbb{R}^{D \times r}`$ for some $`r \ll D`$. Intuitively: the matrix only mixes a low-dimensional subspace of its input into a low-dimensional subspace of its output. For interpretability and for statistical estimation in high dimensions, low-rank constraints are crucial — a full $`D \times D`$ matrix has $`D^2`$ parameters, which is hopelessly over-parameterised when $`D = 4096`$. A rank $`r`$ parameterisation has only $`2Dr`$, and it often captures almost all of the structure.

<a id="cca"></a>
### Canonical correlation analysis (CCA)

Given two sets of variables $`X`$ and $`Y`$ (each a multivariate vector), CCA finds the linear combinations of $`X`$ and the linear combinations of $`Y`$ that are *maximally correlated* with each other. It returns a sequence of "canonical correlations": the first is the highest correlation achievable, the second is the highest that is uncorrelated with the first, and so on.

Why it matters in this proposal: linear regression asks "can I predict $`y`$ from $`x`$?" CCA asks the more symmetric question, "do $`x`$ and $`y`$ share subspaces?" If ridge regression says "I can't predict $`\Delta_\ell`$ from $`v_\ell`$" but CCA finds high canonical correlations, there is nonlinear or non-functional structure (e.g. both quantities encode the same thing along some direction, but not in the form $`y = f(x)`$ ) that the regression was missing.

<a id="granger-causality"></a>
### Granger causality

A statistical test for whether one time series contains information useful for forecasting another, beyond what is already in the target's own past. Given two series $`X_t`$ and $`Y_t`$:
- Fit a model predicting $`Y_t`$ from $`Y_{t-1}, Y_{t-2}, \ldots`$
- Fit a second model adding $`X_{t-1}, X_{t-2}, \ldots`$ as extra predictors.
- If the second model forecasts significantly better, $`X`$ "Granger-causes" $`Y`$.

"Granger cause" is a weaker notion than ordinary causation — it means "informative about the future of" — but in controlled settings with null baselines it is a clean way to quantify how far back a dependence reaches. In this proposal, I replace *time* with *depth*: does $`\Delta_{\ell-k}`$ carry information about $`\Delta_\ell`$ beyond what $`\Delta_{\ell-1}, \ldots, \Delta_{\ell-k+1}`$ already provide? The answer tells me how many layers of depth context the residual-update dynamics actually use.

<a id="autoregressive"></a>
### Autoregressive model

A model that predicts the current value of a series from its own past values:
$$y_t = \sum_{j=1}^k W_j y_{t-j} + \text{noise}$$
"AR($`k`$)" is shorthand for order $`k`$ autoregressive. In this proposal the index $`t`$ is depth, not time, and $`y`$ is $`\Delta_\ell`$.

<a id="toeplitz"></a>
### Toeplitz matrix / Toeplitz structure

A Toeplitz matrix is one where each descending diagonal is constant — the $`(i,j)`$ entry depends only on $`i - j`$. When you write the convolution of a signal with a fixed kernel as a matrix multiplication, the kernel matrix is Toeplitz. More generally, "Toeplitz-constrained" means "the same operator is applied at every position."

**Block-Toeplitz generalisation.** When the repeated entries are matrices rather than scalars, the structure is called *block-Toeplitz*: the $`(i, j)`$ *block* depends only on $`i - j`$. In this proposal the stacked depth-kernel is block-Toeplitz — each block $`W_k \in \mathbb{R}^{D \times M}`$ is a matrix, and the constraint is that $`W_{\text{dec}}^{\ell' \to \ell}`$ depends only on $`k = \ell - \ell'`$. I use "Toeplitz-constrained" and "block-Toeplitz-constrained" interchangeably; the matrix-valued entries are implicit whenever the constraint is applied to a residual-stream kernel.

**What it means across depth.** A block-Toeplitz constraint across depth means "the kernel predicting $`\Delta_\ell`$ from $`\Delta_{\ell-1}, \ldots, \Delta_{\ell-k}`$ (the AR form) — or, equivalently, the kernel mapping source $`s_{\ell-k}`$ into the contribution to $`\Delta_\ell`$ (the MA / generative form) — is the same kernel at every depth $`\ell`$." That is exactly the stationarity assumption, which is a hypothesis, not a premise. The AR and MA forms are related but not strictly equivalent: stationary MA implies a stationary best AR approximation, but AR on $`\Delta`$ stationarity alone does not imply MA stationarity without an invertibility assumption.

<a id="stationarity"></a>
### Stationarity

A process is stationary if its statistical properties do not depend on where you are in the series. In a time series, that means the dynamics look the same in January as in July. In this proposal, where the axis is depth rather than time, stationarity means "the layer-to-layer update dynamics look the same in early layers as in late layers." This is almost certainly *approximately* true and almost certainly *not exactly* true — early layers of transformers tend to do more syntactic / low-level processing and late layers more semantic / task-specific processing — and the empirical question is whether the approximation is useful.

<a id="ssm"></a>
### State-space model (SSM)

A model of the form
$$z_{\ell+1} = f(z_\ell, u_\ell), \quad y_\ell = g(z_\ell)$$
where $`z_\ell`$ is a latent state that evolves from step to step according to $`f`$, and $`y_\ell`$ is an observation computed from it. In interpretability terms, a state-space model for residual updates would say: there is a latent state $`z_\ell`$ that *evolves* with depth, and the residual update is a function of the current state. This is strictly more general than a stationary convolution, because the dynamics $`f`$ can (and typically does) depend on $`\ell`$. If the stationarity test in this proposal fails, a state-space model is the natural fallback (see [Phase 3 Branch B](implementation-plan/phase-3-decomposition.md#step-1--lock-the-architecture-based-on-gate-2)).

**Symbol note.** The canonical SSM literature writes the state as $`s_\ell`$; this proposal writes it as $`z_\ell`$ to avoid collision with the Branch A sources $`\lbrace s_{\ell'}\rbrace`$ (per-layer SAE latents), which are a different object — sparse codes in a dictionary space per source layer, not an evolving latent state.

<a id="alternating-minimization"></a>
### Alternating minimization

An optimisation recipe for problems of the form "minimise $`L(a, b)`$ jointly over $`a`$ and $`b`$" where holding one variable fixed makes the other easy. Pattern: fix $`a`$, optimise $`b`$; fix $`b`$, optimise $`a`$; repeat. Common in dictionary learning (fix the dictionary, solve sparse codes; fix the codes, update the dictionary; repeat) and in matrix factorisation. I use it in the decomposition step: alternate between solving for the per-layer sparse sources $`\lbrace s_{\ell'}\rbrace`$ (with the kernel fixed) and updating the depth kernel $`\lbrace W_k\rbrace`$ (with the sources fixed).

<a id="l1"></a>
### L1 penalty (sparsity)

Adding $`\lambda \sum_i |x_i|`$ to the loss drives individual coordinates of $`x`$ exactly to zero, not just to small values. This is the mathematical mechanism behind "sparse" representations and is at the heart of every sparse autoencoder training loop.

<a id="ablation"></a>
### Ablation (causal testing in mechinterp)

"Turn off" a component — a neuron, a feature, an attention head — during the forward pass and see what changes in the output. Variants include zero-ablation (replace with zero), mean-ablation (replace with average activation over a distribution), and resample-ablation (replace with the activation from a different input). A feature that "represents $`X`$" should, when ablated, break the model's ability to do tasks that depend on $`X`$. Ablation is the minimum standard for claiming you found something causal — without it you've found correlations.

---

## Part D — Motivation: EMG, motor units, and CKC

*This section provides background for the motivation portion of the proposal. The formal model in this proposal is NOT the EMG model; this material is here only so that a reader can follow the intuition used in [README §4.3](README.md#43-where-the-emg-intuition-sits--and-where-it-stops).*

<a id="motor-unit"></a>
### Motor unit

The smallest controllable unit of skeletal muscle: a single motor neuron and the set of muscle fibres it innervates. When the neuron fires, *all* its fibres contract together, producing a brief electrical waveform — a "motor unit action potential." Each motor unit has its own characteristic waveform determined by fibre geometry. Voluntary muscle force is controlled by firing many motor units at different rates, with the firings overlapping in time.

<a id="surface-emg"></a>
### Surface electromyography (surface EMG)

Recording the electrical activity of muscle non-invasively, with electrodes on the skin. In **high-density** surface EMG (HD-sEMG) you use a grid of many electrodes (64, 128, or more) instead of one pair, so that a motor unit action potential is sampled at many spatial points and you can use spatial information to separate sources.

<a id="blind-source-separation"></a>
### Blind source separation

The general problem: observed signals are mixtures of unknown sources, and you must recover the sources without knowing in advance what they look like or how they were mixed. The classical example is the cocktail-party problem — several people talking at once into several microphones, can you recover the individual voices? In EMG, the sources are motor unit firing trains and the mixing operator is the tissue between each muscle fibre and each electrode.

<a id="ckc"></a>
### Convolution Kernel Compensation (CKC)

A specific algorithm for blind source separation of high-density surface EMG, introduced by Holobar and Zazula (2007, [ref](#ref-holobar-2007)). The key observation: the mixing from motor unit spike trains to recorded signals is a *convolution* (a sum over lagged copies of the spike train weighted by the motor-unit action-potential shape), not a simple linear mix. CKC reformulates this as a compensated linear problem by projecting onto the inverse of the covariance structure of the observations, and then uses the structure of the compensated signal to identify individual motor unit firings.

**Why any of this is in the proposal:** it is not, in the sense that the formal model does not use it. It appears only because convolutive source separation with a depth-like structure (here: a *spatio-temporal* kernel between source and electrode) is the analogy that motivates asking whether transformer residual updates might have an analogous structure along depth. The formal model in the proposal does not use CKC. The EMG material here is reference material for the motivation, nothing more.

---

## Part E — Interpretability methods and tools

<a id="sae"></a>
### Sparse autoencoder (SAE)

A neural-network method for finding [features](#feature) in activation space. You take activations from some layer of a model (e.g. the residual stream at layer $`\ell`$ of Pythia-410m), pass them through a *wide* encoder (often many times wider than the activation dimension), apply a sparsity penalty (typically L1 on the latents), and ask the decoder to reconstruct the original activation. The trained latents tend to be much more monosemantic than individual neurons — each latent corresponds, much of the time, to a single human-legible concept. Standard references: Cunningham et al. 2023 ([ref](#ref-cunningham-2023)), Bricken et al. 2023 ([ref](#ref-bricken-2023)), Templeton et al. 2024 ([ref](#ref-templeton-2024)).

**Limitation for this proposal:** a per-layer SAE is trained on one layer in isolation, which means it cannot tell you whether the same feature exists at a different depth or how it transforms as it propagates.

<a id="crosscoder"></a>
### Crosscoder (sparse crosscoder)

An extension of SAEs to multiple layers *at once*. A crosscoder reads activations from many layers simultaneously and fits one shared dictionary of features; each feature has one encoder but many decoders (one per layer), so a single feature can "write" into the residual stream at multiple depths with different weights. Introduced by Lindsey et al. 2024 ([ref](#ref-lindsey-2024)). Crosscoders make cross-layer persistence measurable and have been used for model-diffing applications.

<a id="mlsae"></a>
### Multi-Layer Sparse Autoencoder (MLSAE)

Another approach to cross-layer features, trained with a shared dictionary. Lawson et al. 2024 ([ref](#ref-lawson-2024)) — whose finding is directly relevant to this proposal — observed that individual MLSAE latents are often active on only one layer for a given token, but *which* layer varies across tokens and prompts. This complicates the naive picture where a feature has a single home depth.

<a id="clt"></a>
### Cross-Layer Transcoder (CLT)

Introduced as the basis for the "Circuit Tracing" attribution-graph pipeline (Ameisen et al. 2025, [ref](#ref-ameisen-2025)). A CLT feature reads from the residual stream at a single source layer $`\ell'`$ (via a linear encoder) and writes a contribution to the output of every MLP layer at or after the source, $`\ell \in \lbrace \ell', \ldots, L-1\rbrace`$ (0-indexed: for a model with $`L`$ blocks, MLP indices run $`0`$ through $`L-1`$, matching the convention used in the [implementation plan](implementation-plan/README.md)) via a **separate decoder vector $`d^{\ell' \to \ell}`$** per target layer. Stacked over all features rooted at source $`\ell'`$, these give a decoder matrix $`W_{\text{dec}}^{\ell' \to \ell} \in \mathbb{R}^{D \times M_{\ell'}}`$ — one such matrix per (source, target) pair. The parameters of these matrices are fully free: there is no prior tying the matrix at one depth pair to the matrix at another.

**Parameter counts to keep straight.** Across the whole stack there are $`O(L^2)`$ such decoder matrices (one per ordered pair with $`\ell' \le \ell`$). *Per individual feature* there are $`O(L)`$ decoder vectors (one per target layer at or after the source). CLT's unconstrained parameterisation gives it maximum expressiveness for depth-asymmetric interactions, at the cost of not imposing any structural interpretation of how transition parameters vary with depth. CLT features are the substrate for the attribution graphs that trace how information flows through MLPs layer by layer.

**Relationship to the present proposal.** The hypothesis in this proposal can be viewed as a strong parameter-sharing prior on CLT's decoder matrices: $`W_{\text{dec}}^{\ell' \to \ell} = W_{k}`$ with $`k = \ell - \ell'`$, shared across every position in the stack, and vanishing for $`k`$ larger than some effective depth $`K`$. That prior reduces the count of *distinct transition matrices* from $`O(L^2)`$ to $`O(K)`$, and per-feature reduces $`O(L)`$ free decoder vectors to $`O(K)`$ kernel taps that are shared across source positions. It is valid only if the layer-to-layer update dynamics are approximately stationary along depth, which is exactly what Phase 2 tests. Note also a scope difference: CLT writes into MLP-block outputs, while this proposal targets residual updates $`\Delta_\ell`$ that fold in attention as well.

<a id="jsae"></a>
### Jacobian Sparse Autoencoder (JSAE)

Farnik, Lawson, Houghton & Aitchison, 2025 ([arXiv:2502.18147](https://arxiv.org/abs/2502.18147), ICML 2025) — [full ref](#ref-farnik-2025). A JSAE attaches two SAEs to an MLP block (one on input, one on output) and then sparsifies the **Jacobian** of the MLP computation — the linearised input-to-output sensitivities — in the SAE feature basis. Where CLT captures layer-to-layer *contribution*, JSAE captures per-component *causal sensitivity* between features across a single MLP block. The authors show that pre-trained transformers have substantially sparser Jacobians than randomly-initialised ones, suggesting that computational sparsity — not just representational sparsity — is something transformers learn. Relevant for framing this proposal because (i) it is the closest existing work on *causal* rather than *correlational* layer-to-layer structure, and (ii) one of its authors, Tim Lawson, is also the lead author on [MLSAE](#mlsae).

<a id="routesae"></a>
### RouteSAE (Route Sparse Autoencoder)

Shi, Li, Liang, Wan, Ma, Wang & He, 2025 ([arXiv:2503.08200](https://arxiv.org/abs/2503.08200), EMNLP 2025 main) — [full ref](#ref-shi-2025). Attaches a **lightweight router** to a single shared SAE; the router reads the residual stream at each layer and computes normalised weights that determine which layer's activations the SAE decodes on a per-token basis. The motivation is the observation — most clearly documented by Lawson et al.'s [MLSAE](#mlsae) — that the same feature can activate at different depths depending on the input. RouteSAE lets a single feature dictionary capture this without requiring separate SAEs per layer. Under a matched sparsity budget the authors report both more features recovered and higher interpretability scores than per-layer baselines on Llama-3.2-1B-Instruct. **Relevance to this proposal:** RouteSAE solves a different cross-layer problem than the one H1/H2 address — it is about *where along depth to read from*, not about *how updates propagate along depth* — so the methods are complementary rather than competing.

<a id="tuned-lens"></a>
### Tuned Lens

A calibrated per-layer linear probe for transformers, introduced by Belrose et al. 2023 ([ref](#ref-belrose-2023)). It learns a small affine adapter per layer that maps the residual stream at that layer onto something that decodes, via the same unembedding matrix, into the final output distribution. The result: you can ask, at every layer, "what would the model predict right now?" The Tuned Lens is strictly better calibrated than the older "Logit Lens" (which just applies the unembedding directly). In this proposal I use it as a cross-depth baseline for "how much does each layer change the model's expected output?"

<a id="transformerlens"></a>
### TransformerLens

A Python library (Nanda & Bloom 2022, [ref](#ref-transformerlens)) that wraps open-weight transformers and exposes every internal activation (residual stream, attention patterns, MLP intermediates) through a uniform hook API. It is the standard tool for mechanistic interpretability work on open models and removes most of the engineering cost of "grab the activations from layer $`\ell`$." I will use it to collect the $`\Delta_\ell`$ trajectories the proposal relies on.

<a id="pythia"></a>
### Pythia

A suite of open-weight transformer language models ranging from 70M to 12B parameters, all trained on the same data in the same order (Biderman et al. 2023, [ref](#ref-biderman-2023)). Pythia is the standard platform for interpretability research that needs reproducible, openly-documented models at multiple scales.

<a id="pythia-sae"></a>
### Pythia SAE checkpoints

Publicly released pre-trained SAEs for Pythia residual streams, distributed via the interpretability community (EleutherAI and partners). I use them as per-layer SAE baselines without retraining from scratch.

<a id="gemma-scope"></a>
### Gemma Scope

A comprehensive release by DeepMind (Lieberum et al. 2024, [ref](#ref-lieberum-2024)) of sparse autoencoders trained at every layer and several sites within the Gemma 2 family. Gemma Scope is the largest publicly available collection of ready-to-use SAEs and is the obvious per-layer SAE baseline on Gemma.

<a id="llama"></a>
### Llama

Open-weight transformer language models from Meta. Used in this proposal as a secondary backbone for generalisation tests — "does anything I find on Pythia also hold on Llama?" — because it is a different architecture family trained on different data.

<a id="rome"></a>
### ROME (Rank-One Model Editing)

A method for editing factual associations in transformer language models by directly patching a specific MLP layer (Meng et al. 2022, [ref](#ref-meng-2022)). ROME's lasting contribution for the present proposal is not the edit itself — it's the **causal localisation**: they showed that factual associations like "Paris is the capital of France" are stored in specific middle MLP layers and can be pinpointed with causal mediation analysis. This is the reason factual recall is a good target circuit: prior work has *already* localised it.

<a id="knowledge-neurons"></a>
### Knowledge neurons

Individual MLP neurons that causally contribute to the recall of specific factual associations, identified via integrated gradients attribution (Dai et al. 2022, [ref](#ref-dai-2022)). Part of the same factual-recall literature as ROME; together these papers make factual recall the best-instrumented circuit in interpretability.

---

## Part F — Cited papers

Each entry: one-line contribution, why this proposal cites it, and a pointer to the canonical source.

<a id="ref-elhage-2021"></a>
**Elhage, Nanda, Olsson, Henighan, Joseph, Mann, Askell, Bai, Chen, Olah et al. 2021 — A Mathematical Framework for Transformer Circuits.** Transformer Circuits Thread. [transformer-circuits.pub/2021/framework/index.html](https://transformer-circuits.pub/2021/framework/index.html). *Contribution:* introduced the residual-stream framing of transformers (attention and MLP blocks read from and write to a shared additive channel rather than overwriting the token state), decomposed attention into OV and QK circuits, and established the linear-algebraic vocabulary that every subsequent mechanistic-interpretability paper — including this proposal — uses. *Used here to:* anchor the residual-stream framing underlying §3 of the proposal and the definition of $`\Delta_\ell`$ the hypothesis quantifies.

<a id="ref-elhage-2022"></a>
**Elhage et al. 2022 — Toy Models of Superposition.** Transformer Circuits Thread. *Contribution:* a formal toy model showing that networks with more features than neurons pack features into overlapping directions when those features are sparse, providing the theoretical basis for the "superposition" framing. *Used here to:* motivate the existence of features as overlapping combinations of neurons and the need for decomposition methods.

<a id="ref-bricken-2023"></a>
**Bricken et al. 2023 — Towards Monosemanticity: Decomposing Language Models with Dictionary Learning.** Anthropic. *Contribution:* demonstrated that L1-sparse autoencoders trained on transformer activations recover many monosemantic, interpretable features, giving SAEs their modern prominence. *Used here to:* establish the current per-layer-SAE baseline.

<a id="ref-cunningham-2023"></a>
**Cunningham, Ewart, Riggs, Huben, Sharkey 2023 — Sparse Autoencoders Find Highly Interpretable Features in Language Models.** *Contribution:* the other foundational paper for SAE-based feature discovery, developed in parallel to Bricken et al. *Used here to:* same reason.

<a id="ref-templeton-2024"></a>
**Templeton et al. 2024 — Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet.** Anthropic. *Contribution:* scaled SAE-based feature discovery to a frontier model and demonstrated broad coverage of abstract concepts (including safety-relevant ones). *Used here to:* justify the relevance of SAE-family methods as baselines on large models, and as evidence that per-layer decomposition is already mature enough to be the thing to compare against.

<a id="ref-lawson-2024"></a>
**Lawson, Farnik, Houghton, Aitchison 2024 — Residual Stream Analysis with Multi-Layer SAEs.** *Contribution:* trained a single shared-dictionary SAE reading from multiple layers and found that individual latents are often active on only one layer per token but at different layers across tokens. *Used here to:* motivate the hypothesis — this finding is in tension with a naive "feature lives at depth $`\ell`$" account and is precisely the kind of depth-structure the proposal wants to characterise.

<a id="ref-lindsey-2024"></a>
**Lindsey, Templeton, Marcus, Conerly, Batson, Olah 2024 — Sparse Crosscoders for Cross-Layer Features and Model Diffing.** Anthropic. *Contribution:* introduced sparse crosscoders, which fit one feature dictionary with per-layer decoders across all layers simultaneously. *Used here to:* establish the crosscoder baseline and frame the "cross-layer features exist" prior.

<a id="ref-ameisen-2025"></a>
**Ameisen, Lindsey, Pearce, Gurnee, Turner, Chen et al. 2025 — Circuit Tracing: Revealing Computational Graphs in Language Models.** Anthropic / Transformer Circuits Thread. [transformer-circuits.pub/2025/attribution-graphs/methods.html](https://transformer-circuits.pub/2025/attribution-graphs/methods.html). Companion paper: *On the Biology of a Large Language Model*. *Contribution:* Cross-Layer Transcoders (CLT), in which each feature reads from one residual-stream layer and writes contributions to every subsequent MLP layer via fully-free per-(source, target) decoder matrices, plus an attribution-graph methodology built on top of CLT features and applied to an 18-layer model. *Used here to:* the primary baseline and conceptual anchor; the hypothesis of this proposal is a shared-kernel reparameterisation of CLT's transition matrices and the methods are directly comparable.

<a id="ref-meng-2022"></a>
**Meng, Bau, Andonian, Belinkov 2022 — Locating and Editing Factual Associations in GPT (ROME).** NeurIPS. *Contribution:* localised factual associations to specific middle MLP layers via causal mediation analysis and demonstrated rank-one edits that change facts without retraining. *Used here to:* justify factual recall as the circuit class for the experiments — the circuits are already known and causally testable.

<a id="ref-geva-2023"></a>
**Geva, Bastings, Filippova, Globerson 2023 — Dissecting Recall of Factual Associations in Auto-Regressive Language Models.** EMNLP. *Contribution:* decomposed the factual-recall process into stages (subject enrichment, relation propagation, attribute extraction) and identified the layers and components responsible for each. *Used here to:* inform the choice of factual-recall targets and the layer ranges where effects are expected.

<a id="ref-dai-2022"></a>
**Dai, Dong, Hao, Sui, Chang, Wei 2022 — Knowledge Neurons in Pretrained Transformers.** ACL. *Contribution:* identified individual MLP neurons whose activations causally contribute to specific factual recalls via integrated gradients. *Used here to:* independent evidence that factual recall has a localised, causally testable substrate.

<a id="ref-hase-2023"></a>
**Hase, Bansal, Kim, Ghandeharioun 2023 — Does Localization Inform Editing? Surprising Differences in Causality-Based Localization vs. Knowledge Editing in Language Models.** NeurIPS. *Contribution:* showed empirically that the layer identified by causal-tracing methods (à la ROME) is not in general the layer where editing a fact succeeds — "where a fact is stored" (by tracing) and "where a fact can be changed" (by editing) are decoupled. *Used here to:* a caveat on the ROME/knowledge-neurons picture — factual-recall circuits are the best-instrumented target class available, but the prior-work layer assignments are strong priors, not ground truth.

<a id="ref-elazar-2021"></a>
**Elazar, Kassner, Ravfogel, Ravichander, Hovy, Schütze, Goldberg 2021 — Measuring and Improving Consistency in Pretrained Language Models.** Transactions of the Association for Computational Linguistics. *Contribution:* introduced the **Pararel** dataset — paraphrased cloze templates over subject-relation-object triples — and showed that pretrained language models are inconsistent in their factual predictions across paraphrases of the same relation. *Used here to:* secondary dataset source for the factual-recall triples assembled in [Phase 1 Step 1](implementation-plan/phase-1-diagnostic.md#step-1--assemble-the-factual-recall-dataset), alongside CounterFact. Cited only in the implementation plan.

<a id="ref-belrose-2023"></a>
**Belrose et al. 2023 — Eliciting Latent Predictions from Transformers with the Tuned Lens.** *Contribution:* a calibrated per-layer lens that decodes the residual stream at each layer into the output vocabulary, fixing the miscalibration problems of the older Logit Lens. *Used here to:* provide a cross-depth baseline for "how much does each layer change the model's prediction?"

<a id="ref-transformerlens"></a>
**Nanda, Bloom 2022 — TransformerLens.** *Contribution:* a Python library exposing every internal activation of open-weight transformers through a uniform hook API. *Used here to:* the entire activation-collection pipeline in Phase 1 of the implementation plan is built on TransformerLens.

<a id="ref-biderman-2023"></a>
**Biderman et al. 2023 — Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling.** ICML. *Contribution:* released a family of open transformers (70M–12B) trained on the same data with full checkpoints. *Used here to:* primary experimental platform because of reproducibility, public SAEs, and multiple scales.

<a id="ref-lieberum-2024"></a>
**Lieberum et al. 2024 — Gemma Scope: Open Sparse Autoencoders Everywhere All at Once on Gemma 2.** DeepMind. *Contribution:* a comprehensive release of SAEs at every layer and site of Gemma 2. *Used here to:* secondary experimental platform and per-layer SAE baseline on a non-Pythia architecture.

<a id="ref-farnik-2025"></a>
**Farnik, Lawson, Houghton, Aitchison 2025 — Jacobian Sparse Autoencoders: Sparsify Computations, Not Just Activations.** [arXiv:2502.18147](https://arxiv.org/abs/2502.18147), ICML 2025. *Contribution:* trains paired SAEs on the input and output of an MLP block and adds a sparsity penalty on the *Jacobian* of the MLP in the feature basis, producing features connected by sparse computation graphs rather than merely sparse activations. *Used here to:* closest prior work on *causal* per-block layer-to-layer structure; contrasted with CLT (contribution, not sensitivity) and with the shared-kernel hypothesis of this proposal.

<a id="ref-shi-2025"></a>
**Shi, Li, Liang, Wan, Ma, Wang, He 2025 — Route Sparse Autoencoder to Interpret Large Language Models.** [arXiv:2503.08200](https://arxiv.org/abs/2503.08200), EMNLP 2025 main. *Contribution:* a lightweight router attached to a shared SAE that dynamically weights activations from multiple layers on a per-token basis, addressing the same-feature-different-depth phenomenon highlighted by MLSAE. *Used here to:* an example of an orthogonal cross-layer approach (layer-selection vs. depth-kernel) cited to clarify the positioning of the present hypothesis.

<a id="ref-holobar-2007"></a>
**Holobar, Zazula 2007 — Multichannel Blind Source Separation Using Convolution Kernel Compensation.** IEEE Transactions on Signal Processing 55(9). *Contribution:* the CKC algorithm for HD-sEMG decomposition. *Used here to:* cited once, in the motivation paragraph, as the source of the intuition — convolutive blind source separation along a depth-like axis — that motivated the present hypothesis. It is *not* the formal model of this proposal.

---

<a id="further-cross-layer-methods"></a>
## Part G — Methods considered and not cited

One remark, to document a decision rather than a gap.

**Matryoshka SAEs (Bussmann et al., 2025, [arXiv:2503.17547](https://arxiv.org/abs/2503.17547)).** Train multiple nested dictionaries of increasing width on the activations of a single layer, so that smaller dictionaries must independently reconstruct the input and therefore learn coarser features. This is a *sparsity-width* hierarchy within a layer, not a depth hierarchy across layers, and does not compete with the present proposal's hypothesis. I considered citing it as an example of "hierarchical" SAE work and decided it would confuse rather than clarify the cross-layer framing.

---

*Back to: [research proposal](README.md) · Continue to: [implementation plan](implementation-plan/README.md)*
