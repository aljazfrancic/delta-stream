# Phase 4 — E5: source-prefix sweep (the embedding-only lower bound, generalised)

**Date:** 2026-06-30. **Model:** pythia-410m-deduped. **Run:** `runs/2026-06-30-ph4-e5-prefix-410md`
(`scripts/phase4_embed_only.py --cutoffs 0 3 7 11 15 19 23`). **Sources:** frozen mlsae
`u[1555,24,4433]`, rank 32, λ_reg 0.1, same frozen-source Adam recipe as E3.

[phase-3 Step 2] pre-registers an **embedding-only** source variant (sources only at `s_0`, all
`u_ℓ=0` for ℓ>0) "reported as a lower bound", expected (per [§4.3], Geva et al.) to under-fit
late-layer updates because subject/relation features are *built up* across depth, not present at the
embedding. Since v_0 has **no SAE** in any available dictionary (u_0 is zero-init everywhere), the
literal ℓ0-only variant is the example-independent floor, so E5 reports the **full sweep**: keep
sources for ℓ≤ℓ_max, zero the rest, and fit the SSM (free to propagate the kept sources forward).
The ℓ_max=0 point is the registered lower bound; the curve measures how load-bearing each band of
per-layer sources is.

> **Verdict — per-layer sources are load-bearing at EVERY depth, especially late; propagation cannot
> substitute for them.** Held-out Δ-reconstruction R² rises monotonically as more layers' sources are
> added — **−0.44 (ℓ0) → +0.20 (ℓ11) → +0.58 (ℓ19) → +0.77 (ℓ23)** — only becoming positive past the
> halfway layer and still climbing steeply over the last four layers (ℓ19→23 adds **+0.19**). So late-
> layer sources carry real, **non-propagable** reconstruction value: you cannot reconstruct late Δ_ℓ
> from early/mid sources propagated through the state chain. This confirms the §4.3 "features built up
> across depth" prediction and justifies the central design choice — sparse sources at *every* layer,
> not just the embedding.**

## Result — source-prefix buildup (frozen mlsae sources, rank 32)

| ℓ_max | # source layers | test R² |
|---|---|---|
| 0 (embedding-only lower bound) | 1 | **−0.438** |
| 3 | 4 | −0.343 |
| 7 | 8 | −0.248 |
| 11 | 12 | +0.199 |
| 15 | 16 | +0.404 |
| 19 | 20 | +0.576 |
| 23 (full) | 24 | **+0.770** |

`tables/embed_only.json`, `figures/source_prefix_sweep.png`.

## Interpretation — the ablation ladder, with E3

Combining E5 with the E3 `use_state` ablation (same frozen sources, rank, recipe) gives the full
decomposition of *where the reconstruction comes from*:

| variant | what it keeps | test R² |
|---|---|---|
| embedding-only (ℓ_max=0) | only u_0 (=0) + propagation | −0.44 |
| per-layer, no propagation (`use_state=False`, E3) | all sources, **no** cross-layer | +0.73 |
| **full SSM** | all sources + propagation | +0.77 |

- **Per-layer sources carry almost everything.** Removing them (ℓ_max=0) collapses reconstruction to
  the example-independent floor (−0.44); keeping them but removing cross-layer propagation
  (`use_state=False`) only drops +0.77→+0.73. So the per-layer feedthrough sources `D_ℓ u_ℓ` do the
  bulk of the work, and cross-layer propagation `C_ℓ z_ℓ` adds a small (+0.04–0.05) increment —
  consistent with k*≈1 (first-order) and the feedthrough-dominance finding (S1: 74% of contribution
  mass at the home layer).
- **Late sources are not redundant.** The buildup does not saturate early: ℓ_max=19 (20 layers of
  sources) still leaves +0.19 of held-out R² on the table that only the last four layers' sources
  recover. Propagating earlier sources forward cannot reconstruct late-layer Δ — the late-layer
  features are genuinely new, matching Geva et al.'s built-up-recall picture.
- **The embedding-only floor is degenerate by construction (honest framing).** v_0 has no SAE in any
  dictionary (mlsae *and* trained-L1 both zero-init u_0), so at ℓ_max=0 the SSM has no input: the
  state trajectory `z_ℓ=(∏A)z_0` is example-independent, giving the per-depth-mean-like floor (R²<0
  where that fixed trajectory is wrong, worst at the last layer). The pre-registered lower bound is
  therefore the floor here; the informative content is the buildup curve above it.

## Bottom line

E5 turns the pre-registered embedding-only lower bound into a buildup curve that directly supports
the paper's source design: sparse sources at every layer are load-bearing, late layers especially,
and cross-layer propagation is a small additive term on top of dominant per-layer feedthrough. With
E3 it bounds the decomposition from both sides — removing propagation costs little (+0.73 vs +0.77),
removing per-layer sources is catastrophic (floor) — a clean, complete ablation story.
