# Implementation Plan

The experimental protocol for the research proposal [*Structured Cross-Layer Propagation of Features in Transformers*](../README.md).

This file is the overview. Each phase below is its own document with step-by-step instructions, commands, data structures, and decision criteria.

> **Navigation**
> - [← Research proposal](../README.md)
> - [Glossary](../glossary.md) — every technical term and citation has an entry
> - **Phase 1 — Diagnostic:** [phase-1-diagnostic.md](phase-1-diagnostic.md)
> - **Phase 2 — Characterization:** [phase-2-characterization.md](phase-2-characterization.md)
> - **Phase 3 — Decomposition:** [phase-3-decomposition.md](phase-3-decomposition.md)
> - **Phase 4 — Evaluation and write-up:** [phase-4-evaluation.md](phase-4-evaluation.md)

---

## 1. What this plan is and isn't

This is an **instruction manual to my future self**. It reads as commands ("do X"), not as argumentation. The justification for every choice lives in the [research proposal](../README.md); this plan tells me *how* to run the experiments, *in what order*, and *what decision to make at each gate*.

It is also **pre-registration**. The go/no-go gates are written down here before any data is collected so that I do not retroactively lower the bar when a result is disappointing. The failure-mode branches (what I do if H1 fails, what I do if H2 fails) are pre-specified.

It is **not**: a list of paper sections, a timeline I will slavishly follow, or a commitment to never reorder steps. The real value of writing the plan is that it forces decisions that would otherwise be made under pressure mid-experiment.

---

## 2. Phases at a glance

| Phase | Goal | Output | Go/no-go gate |
|---|---|---|---|
| **1. Diagnostic** | Does $`\Delta_\ell`$ have cross-layer structure at all? | R², Granger profiles, null-test results | **Gate 1:** Is H1 (depth-local predictability) supported above null? |
| **2. Characterization** | What *kind* of structure? Stationary? How deep? Generalisable across relations / models? | Kernel comparison plots, effective-depth estimates, cross-model results | **Gate 2:** Does H2 (stationarity) hold? → determines decomposition architecture |
| **3. Decomposition** | Fit a generative model: sparse sources + depth kernel (or SSM if stationarity failed) | Trained decomposition on Pythia-410m, feature dictionary | **Gate 3:** Is reconstruction competitive with SAE/crosscoder baselines? |
| **4. Evaluation** | Causal existence proof, baseline comparisons, paper write-up | Ablation results, Pareto curves, draft paper | Deliverable |

Each phase ends with an explicit decision point. **If a phase fails its gate, the project does not silently continue** — it pivots to writing up the negative result.

---

## 3. Target models

Primary: **[Pythia-410m](../glossary.md#pythia)** — small enough to iterate quickly, residual stream dimension 1024, 24 layers, fully reproducible training, public [SAE checkpoints](../glossary.md#pythia-sae) available.

Secondary (Phase 2 generalisation + Phase 4 evaluation): **[Gemma-2-2B](../glossary.md#gemma-scope)** — supported by the most comprehensive public SAE suite ([Gemma Scope](../glossary.md#gemma-scope)), larger residual dimension, different architecture family.

Tertiary (Phase 2 generalisation, optional): **Llama-3.1-8B** or similar, only if compute permits, only for the "does this hold on another architecture family?" cross-check.

The plan is structured so the primary experiments fit on a single A100-40GB or equivalent; the secondary sweeps would benefit from more.

---

## 4. Target circuits: factual recall

**Why factual recall.** The research proposal ([§5](../README.md#5-experimental-strategy)) explains the scientific reason. The operational reason: factual-recall circuits are the best-instrumented circuits in the mechinterp literature ([ROME](../glossary.md#rome), [knowledge neurons](../glossary.md#knowledge-neurons), [Geva et al. 2023](../glossary.md#ref-geva-2023)). I know which layers matter, which tokens matter, and which interventions have ground-truth effects. This removes most of the "is this a real feature or a narrative?" risk.

**Dataset.** 5,000–10,000 curated subject-relation-object triples, filtered for:
- The primary model reliably outputs the correct object. Default threshold: top-1 probability > 0.3 on Pythia-410m, tuned upward if the filtered set comfortably exceeds 5,000 examples and downward only if it does not. The exact cutoff and the resulting sample size are recorded in [Phase 1, Step 1](phase-1-diagnostic.md#step-1--assemble-the-factual-recall-dataset).
- Multiple relation types are represented: geography, dates, people, professions, etc. I want diversity across *relations* so that Phase 2 can ask "does the kernel generalise across relation types?"
- No test-set contamination with any model's training set that I can avoid (this matters less for an interpretability study than for a capability one, but I will note any contamination warnings in the paper).

**Source candidates** for the dataset: the CounterFact dataset from the ROME paper and the Pararel dataset ([Elazar et al. 2021](../glossary.md#ref-elazar-2021)) as the primary sources, optionally the T-REx subset used in [Geva et al. 2023](../glossary.md#ref-geva-2023) if the union is short of the 5,000-example target. I will union and re-filter these rather than build one from scratch.

**Token of interest per example.** For each triple $`(s, r, o)`$, the model is prompted with the natural-language template supplied by the source dataset. The s-r-o convention is fixed here: for $`(s, r, o) = (\textit{Paris}, \textit{capital of}, \textit{France})`$, the prompt is "Paris is the capital of" and the target continuation is "France" — the subject is the input, the object is what the model must predict. [Phase 1 Step 1](phase-1-diagnostic.md#step-1--assemble-the-factual-recall-dataset) specifies the reformatting rules for datasets (notably CounterFact) that store triples in the reverse direction. The activation I collect is the residual stream state at the *final prompt token* (the last token before the model would generate $`o`$) at *every layer*. This gives me one $`L`$-long trajectory per example, where $`L`$ is the number of layers.

---

## 5. What I am collecting and saving

For every example in the dataset, from the primary model, I save:

- **Residual stream vectors** $`v_\ell \in \mathbb{R}^D`$ at every layer $`\ell \in \lbrace 0, 1, \ldots, L\rbrace`$ (layer 0 = embedding output, layer $`L`$ = final pre-unembed).
- Derived on the fly in later phases: the layer-to-layer updates $`\Delta_\ell = v_{\ell+1} - v_\ell`$. I save $`v_\ell`$ and compute $`\Delta_\ell`$ downstream rather than saving both, to save disk.
- **Metadata per example:** subject, relation, object, top-1 prediction and probability, any filter flags.

**Storage estimate.** $`N = 10{,}000`$ examples $`\times`$ $`L+1 = 25`$ residual-stream states (Pythia-410m has $`L=24`$ blocks, giving 25 states $`v_0, \ldots, v_{24}`$) $`\times`$ $`D = 1024`$ dim $`\times`$ 4 bytes (float32) $`\approx`$ 1 GB. Trivial.

Format: a single HDF5 or `safetensors` file per model, with per-example and per-layer indexing. I do not store attention patterns or MLP intermediates — this project is about the residual stream.

---

## 6. Work plan by phase

A phase-by-phase breakdown of the work and the deliverable at each gate.

### Phase 1 — Diagnostic ([phase-1](phase-1-diagnostic.md))

Environment setup, dataset assembly, TransformerLens pipeline, activation collection for Pythia-410m. Linear predictability test (ridge), CCA, reporting per-layer-pair $`R^2`$ and canonical correlation profile. AR($`k`$) fits, Granger causality profile, scrambled-activations null test, context-perturbation null test. **This phase ends at Gate 1.**

**Deliverable at Gate 1.** A short memo answering: *Does H1 hold on factual-recall examples in Pythia-410m, above null baselines, with context-perturbation controls?* Three possible outcomes:
1. **Yes, cleanly:** proceed to Phase 2.
2. **Partial / context-dependent:** the dynamics exist but context perturbation changes them — the structure is cross-context, not depth-intrinsic. Pivot to characterising *this*, which is itself publishable.
3. **No:** Phase 1 is the paper. Write up the negative result with the null tests as the main contribution and pivot to a related question (e.g. how CLT features relate to Granger structure).

### Phase 2 — Characterization ([phase-2](phase-2-characterization.md))

Stationarity test: fit kernels separately on early, mid, late layer ranges; statistical comparison. **Decision: Toeplitz-constrained decomposition in Phase 3, or state-space model.** Effective kernel depth $`k^\ast`$ — how far back in depth does the AR structure reach before returns diminish? Generalisation across relation types: same experiments on geography vs. date vs. person-birthplace subsets. Replicate the linear and AR tests on Gemma-2-2B — do the kernels transfer, and is the effective depth similar? **This phase ends at Gate 2.**

**Deliverable at Gate 2.** Characterisation of the dynamics: stationary or not, effective depth, cross-relation transfer, cross-model transfer. Decide the Phase 3 decomposition architecture (convolutive if H2 holds; SSM otherwise).

### Phase 3 — Decomposition ([phase-3](phase-3-decomposition.md))

Prototype the decomposition on Pythia-70m first (to debug fast), then scale to Pythia-410m: alternating minimization loop, sparsity sweeps, reconstruction sanity checks. Then stability and seed-sensitivity — train the decomposition under multiple random seeds and data subsets and measure feature recovery agreement. **This phase ends at Gate 3.**

**Deliverable at Gate 3.** A trained decomposition on Pythia-410m with reconstruction quality and sparsity numbers that can be compared against per-layer SAE and crosscoder baselines. Go/no-go: is reconstruction within reasonable bounds of the per-layer SAE Pareto frontier?

### Phase 4 — Evaluation and write-up ([phase-4](phase-4-evaluation.md))

Baseline comparisons, Pareto curves, stability metrics. The causal existence proof: find a feature my method recovers that baselines miss, ablate it, verify behaviour change on factual-recall queries. Paper draft, figures, anonymised code release, internal review, polish.

---

## 7. Cross-cutting engineering decisions

### 7.1. Reproducibility

Every experiment is a Python module with a CLI. Every run logs its config, git commit, and random seed to a single directory named by date + short description. No notebooks as source of truth — notebooks are for exploration and their results must be reproducible from a script. Figures are regeneratable from saved intermediates.

### 7.2. Null tests are mandatory, not decorative

The temptation in exploratory interpretability work is to find something interesting-looking and move on. This project has two structural null tests baked into Phase 1 (scrambled activations, context perturbation) that must be run *before* any claim is made, and the plan specifies their exact form. If I am tempted to skip them, I should reread this sentence.

### 7.3. Compute budget

The diagnostic phase needs essentially no GPU time beyond collecting activations — ridge, CCA, AR models, Granger tests are CPU jobs and scale fine on activation tensors this size. The decomposition and evaluation phases need real GPU for SAE baselines (if any need to be retrained) and for ablation sweeps. Budget target: a single A100, with scale-up availability for the evaluation-phase baseline comparisons.

### 7.4. When to ask for help

The gate memos exist to force surfacing results early: I will share each memo with collaborators / reviewers at the time of the gate rather than only after the final result. The *worst* thing I can do is fail silently and then scramble for help at the end — the go/no-go structure exists partly to prevent this.

---

## 8. Deliverables summary

- **Gate 1 memo:** H1 supported / not / context-dependent.
- **Gate 2 memo:** stationarity decision, effective depth, cross-relation/cross-model results.
- **Gate 3 memo:** decomposition trained, baseline-comparable reconstruction.
- **Final paper draft:** negative or positive result, causally verified, compared to baselines.
- **Public code release** (with the paper): reproducible pipeline from raw activations to final figures.

---

*Next: [Phase 1 — Diagnostic](phase-1-diagnostic.md)*
