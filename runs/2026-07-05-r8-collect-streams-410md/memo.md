# R8 — attn-out vs MLP-out Δ split: where does the cross-layer structure live?

**Date:** 2026-07-05. **Design:** split each residual update into its two additive
sublayer writes, Δ_ℓ = attn_out_ℓ + mlp_out_ℓ (exact in TransformerLens; Pythia is
`parallel_attn_mlp=True`), store each as a **pseudo-stream** (w_0 = v_0,
w_{ℓ+1} = w_ℓ + write_ℓ) so `deltas()` returns the per-layer writes and the existing
`--h5` H1/H2/k* pipelines run unchanged. **Runs:** collection
`runs/2026-07-05-r8-collect-streams-410md`; analyses `runs/2026-07-05-r8-{step5,
stationarity,kstar}-{attn,mlp}-410md`. **Question (2026-07-02 review, optional
mechanism polish):** the residual-stream AR structure could be an artefact of the
±v_ℓ persistence tautology. The split streams carry **no ±v_ℓ term by construction**,
so they are a tautology-free read of where the cross-layer structure actually sits —
and they separate the attention write from the MLP write.

> **Verdict.** Both sublayer streams independently reproduce the residual-stream
> story **tautology-free**: H1 holds (AR real−null attn **0.542** / mlp **0.653**,
> Granger **112/112** vs **0/112** both, nulls ≈0), and H2 fails identically
> (`pervasive_drift`, every cross-band R² negative, rank promoted 256→512 both). The
> new result is the **k\* contrast**: the **MLP write is strictly first-order at every
> depth (k\*=1)** while the **attention write carries longer-range depth memory
> (k\* median 2, rising to 10 at ℓ=22)**. The MLP stream also dominates energy
> (median **78%** of ‖Δ‖²; attention **22%**, min **5%** late). So the residual
> stream's headline k\*≈1 is the MLP write's first-order-ness showing through the
> energy-dominant channel; the smaller attention write is where the deep Granger
> reach lives.

## Collection integrity (three independent checks)

- **Additivity** attn_out + mlp_out vs the same-pass residual Δ: rel-Frobenius
  **7.13e-08** (exact up to fp32).
- **Residuals vs the Phase-1 store**: max deviation **0.00e+00** (bit-identical — the
  split used the same activations).
- **Pseudo-stream round-trip** (cumsum → `deltas()` → write): max abs **~1e-6** both
  streams.

## H1 (tautology-free) — both streams

| stream | AR best R² (k, rank) | scrambled null | real−null | Granger sig | H1-nonstat flag |
|---|---|---|---|---|---|
| attn_out | 0.542 (k20, r128) | ≈0 | **0.542** | **112/112** | False |
| mlp_out  | 0.653 (k20, r128) | ≈0 | **0.653** | **112/112** | False |

Neither stream contains −v_ℓ, so this is the clean answer to the tautology worry the
review raised: the AR-on-Δ signal is **not** an artefact of the persistence term —
it survives in both sublayer writes that lack that term entirely. (Caveat: the split
writes are both functions of the shared upstream state v_ℓ, so this shows *structure
in what the sublayers write*, not sublayer→sublayer causation.)

## H2 — both streams `pervasive_drift` (→ Branch B)

MLP cross-band held-out R² (within-band on the diagonal, positive; every off-diagonal
negative — kernels do not transfer across depth bands):

| fit \ eval | early | middle | late |
|---|---|---|---|
| early | **0.914** | −1.03 | −0.63 |
| middle | −0.63 | **0.800** | −0.66 |
| late | −1.85 | −3.09 | **0.786** |

Same signature as the residual stream and the attention stream: no clean segment
boundary (triangle ratio 1.008), between-band ‖ΔW‖ ≈ √2, working rank promoted
256→512 by the spot-check. Verdict `pervasive_drift` on all three of {residual, attn,
mlp} → the depth-indexed (not stationary, not segmented) Branch B is the right form
for each.

## k\* — the contrast that carries the mechanism

| stream | per-depth k\* median | max k\* (depth) |
|---|---|---|
| residual (Phase 2) | 1 | 2 (ℓ17) |
| **mlp_out** | **1** | **1** (flat) |
| **attn_out** | **2** | **10** (ℓ22) |

Reading: the energy-dominant MLP write is **memoryless in depth** — a single
preceding write reconstructs it everywhere. The attention write is the smaller
channel but its effective order grows with depth, reaching 10 late in the stack.
This localises the residual stream's "statistically-deep-but-reconstruction-shallow"
tension (Granger reach 20 vs recon k\*≈1): the deep-reaching part is the attention
write, and it is reconstruction-negligible only because attention contributes a
minority of the update energy. Pairs with the A_ℓ-contractivity result (fig 12) — a
contractive state gives the first-order MLP-dominated residual — and with the
MLP-feedthrough-dominance of the fitted SSM (|D| ≫ |C|).

## Scope

Optional mechanism analysis; does not change any gate. Same 3-class / 1555-ex factual
store, query position. The energy split (78/22 MLP/attn) is Pythia-specific
(`parallel_attn_mlp=True`); serial-block families would need a re-collection.
