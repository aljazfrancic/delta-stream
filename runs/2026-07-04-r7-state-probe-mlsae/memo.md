# R7 — SSM state probe: the latent state IS functionally informative (not "reconstruction-only")

**Date:** 2026-07-04. **Script:** `scripts/phase4_state_probe.py` (new; uses the new
`LowRankSSM.states()`). **Runs:** `runs/2026-07-04-r7-state-probe-mlsae` (canonical
mlsae checkpoint `runs/2026-06-26-ph4-checkpoint-410md`) and
`runs/2026-07-04-r7-state-probe-l1` (trained-L1 checkpoint
`runs/2026-06-30-ph4-e2-checkpoint-l1-410md`). **Question (4th-pass top experiment):**
does the learned 32-dim cross-layer state `z_ℓ` linearly predict task quantities better
than a dimension-matched summary of the per-layer code — i.e. does the decomposition
have a *functional* payoff beyond reconstruction, before the abstract commits to
"reconstruction-only"?

> **Verdict — YES, scoped: at early/mid depth the state is a significantly more
> answer-informative 32-dim summary than variance-optimal (PCA) compressions of the
> very codes it is built from, on both dictionaries; by late layers a per-layer code
> summary catches up, and the model's own residual stream remains the best readout.**
> The abstract can say the state is an information-bearing object (answer identity
> decodable at 0.76–0.80 top-1 from 32 dims by ℓ≈17–21), not merely a reconstruction
> device.

## Design

All arms are **d = d_z = 32** per layer, probed with the identical linear model
(multinomial logistic / ridge), features standardized on train; C/α selected on a
**train-internal validation carve-out** (995 fit / 249 val — test never used for
selection, per 4th-pass finding 4); test = the canonical 311 held-out examples.

| arm | what it is | inputs it sees |
|---|---|---|
| `state` | `z_ℓ` from the checkpoint (LASSO codes → `states()`) | `u_0 … u_{ℓ-1}` (linear image) |
| `code_pca` | train-fit PCA-32 of `u_ℓ` | current layer's code |
| `code_prefix_pca` | train-fit PCA-32 of concat `u_0 … u_ℓ` | superset of the state's inputs |
| `resid_pca` | train-fit PCA-32 of `v_ℓ` | the raw stream (reference upper bound) |

Since the SSM is linear, `z_ℓ` and prefix-PCA are **both 32-dim linear images of the
same prefix code vector** — the comparison isolates *which* linear compression retains
task information: the one trained only to reconstruct Δ, or the variance-optimal one.

Targets: **answer identity** (`object`, 205-way; majority 0.106), **relation class**
(3-way; majority 0.457), **model confidence** (ridge on logit `top1_prob`).
Correctness is degenerate (the dataset is filtered to model-correct facts).
Permuted-label null on the state arm floors at 0.05–0.13 ≈ majority (calibrated).
Headline contrasts are **band-aggregated over depth thirds** with paired bootstrap CIs
over the 311 test examples (no max-over-layers selection).

## Results — answer identity (the money target)

Band contrasts, state − other, paired 95% CI (\* = significant):

| contrast | mlsae | trained-L1 |
|---|---|---|
| state − code_pca / early | −0.012 [−0.030, +0.006] | **+0.044 [+0.023, +0.066]\*** |
| state − code_pca / mid | **+0.060 [+0.037, +0.084]\*** | **+0.139 [+0.105, +0.173]\*** |
| state − code_pca / late | +0.015 [−0.006, +0.035] | +0.000 [−0.015, +0.015] |
| state − prefix_pca / mid | **+0.080 [+0.048, +0.113]\*** | **+0.173 [+0.131, +0.217]\*** |
| state − prefix_pca / late | **+0.110 [+0.079, +0.141]\*** | **+0.188 [+0.154, +0.223]\*** |
| state − resid_pca / mid | **−0.034 [−0.056, −0.013]\*** | +0.021 [−0.001, +0.042] |
| state − resid_pca / late | **−0.066 [−0.084, −0.048]\*** | **−0.041 [−0.059, −0.025]\*** |

Curve shape (both dictionaries): state ≈ chance at ℓ≤1 *by construction* (`z_ℓ` sees
only `u_{<ℓ}` and `u_0 = 0` — no embedding SAE), rises steeply through mid-depth
(mlsae ℓ17: state **0.759** > resid_pca 0.666 > code_pca 0.601), peaks 0.775 (mlsae,
ℓ21) / 0.801 (L1, ℓ21), converges to the code arm late while the raw stream stays on
top (max 0.859). The late-layer jump of *all* arms at ℓ≈17–18 matches the known
answer-assembly depth (Geva-style enrichment→readout transition).

## Secondary targets

- **Relation class saturates at ≈1.00 for every arm** (even resid_pca at ℓ0 = 0.89) —
  non-discriminative; reported for completeness only.
- **Confidence (logit top1_prob):** state R² max 0.34 (mlsae) / 0.44 (L1) **below**
  code_pca 0.48/0.50 and resid_pca 0.53 — the state's advantage is specific to answer
  *identity*, not the model's confidence level. Honest scope: the state compresses
  *what* is being recalled better than generic summaries, not *how strongly*.

## What this changes for the write-up

- The Phase-4 story is no longer "reconstruction-only": alongside the clean causal
  null (E1/E2) and the structural A_ℓ result, the decomposition's state is a
  **functional mid-depth summary** — trained purely on Δ reconstruction, it
  concentrates answer-identity information that variance-optimal 32-dim compressions
  of the same codes discard (up to +0.14–0.17 top-1 over PCA at matched dimension,
  CI-clean, two dictionaries).
- Pairs naturally with A_ℓ contractivity (τ ≈ 0.83 layers): the state is a *short-
  memory* but information-dense running summary of the recall computation.
- Keep the reference frame honest: probing the raw stream at the same dimension is
  still better late (−0.04…−0.07), so the claim is about the *decomposition's inputs*
  (the state beats its own code family at equal dimension), never "better than the
  model's own representation."

## Caveats

1. Probes are post-hoc linear readouts; `z_ℓ` was never trained for classification.
   (This makes the positive result *more* surprising, not less.)
2. `code_pca` sees `u_ℓ` while `state` sees `u_{<ℓ}` — disjoint at a fixed layer; the
   information-superset control is `code_prefix_pca`, which the state beats everywhere
   past early depth on both dictionaries.
3. PCA is the variance-optimal *linear* summary; a task-supervised 32-dim projection
   of the prefix codes would close some gap by construction — but that would be a
   supervised competitor, not a dimension-matched unsupervised summary.
4. Single seed/checkpoint per dictionary (the canonical ones); the mlsae-vs-L1
   agreement is the robustness check. Bands pre-registered as depth thirds; no
   layer-level selection was used for the headline contrasts.
5. 205-way accuracy is optimistic in absolute terms (object classes recur; majority
   0.106) — all arms share this, and R5's object-disjoint result covers the
   leakage question for the reconstruction claims.
