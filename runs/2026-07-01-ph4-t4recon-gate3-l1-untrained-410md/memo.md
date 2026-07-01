# Untrained reconstruction control (T4 follow-up) — the reconstruction win is ~half learned

**Question.** T4 showed the AR-on-Δ *predictability* is architectural (random-init ≈ trained).
Does that deflation reach the paper's HEADLINE — the Branch-B SSM *reconstruction* win
(cross-layer state beating the per-layer `use_state=False` baseline)? Fit the identical
Gate-3 Pareto on **random-init** Pythia-410m activations and compare the SSM−baseline gap
to trained.

**Setup (matched).** Both runs: L=24, D=1024, d_z=r=32, split 1244/311 seed 0, M≈1177,
λ∈{0.3,1,3,10}. Sources = L1 SAE trained on each model's *own* activations (untrained SAE
recon R² mean +0.917, min +0.895 — dictionary faithful, comparison valid).
Trained ref: `runs/2026-06-29-ph4-e2-gate3-l1-410md`; untrained: this run
(`scripts/train_410m_saes.py --h5 …-untrained/… ` then `gate3_pareto.py --h5 …`).

**Result — SSM−baseline gap (held-out R²), matched λ:**

| λ | trained gap | untrained gap | untrained/trained |
|---|---|---|---|
| 0.3 | +0.0506 | +0.0327 | 65% |
| 1   | +0.0483 | +0.0287 | 59% |
| 3   | +0.0437 | +0.0228 | 52% |
| 10  | +0.0413 | +0.0140 | 34% |
| **mean** | **+0.046** | **+0.025** | **53%** |

**Verdict: PARTIAL — the reconstruction win is ~half architectural, ~half learned.**

- Untrained activations DO benefit from cross-layer state (gap +0.025 mean, positive,
  monotone in λ, clearly ≠ 0) → the reconstruction win is **not purely learned**; a
  substantial architectural component carries over from T4.
- BUT untrained is only **53% of trained** → training roughly **doubles** the cross-layer
  benefit. Unlike T4 (untrained matched/exceeded trained → *fully* architectural), the
  reconstruction method captures a **genuine learned signal ≈ half its advantage**
  (learned increment = trained − untrained = **+0.021** mean).
- **The learned fraction GROWS with sparsity:** at λ=10 (sparse, the interpretability-
  relevant operating point) untrained is only 34% → learned **dominates (66%)**. At the
  dense end (λ=0.3) it is 65% architectural. The cross-layer structure that survives
  sparsification is the *learned* part.

**Dissociation (the write-up point).**
- AR-on-Δ depth predictability (H1 diagnostic) → **architectural** (T4).
- Branch-B SSM reconstruction (the method contribution) → **~half learned**, and
  *majority*-learned at the sparse operating point.
So the deflation does **not** swallow the headline: the decomposition extracts something
genuinely learned even though the raw depth structure it exploits is architectural.
Report the trained gap +0.046 honestly as "of which ~half (+0.025) is present at random
init; learned increment +0.021, rising to a majority under sparsity."

**Absolute levels.** Untrained ssm R² 0.68–0.75 vs trained 0.81–0.87 (untrained activations
are intrinsically less reconstructable; the *gap* controls for this via the per-layer
baseline, which is the pre-registered quantity). Figure `figures/pareto.svg`. Single init,
seed 0; qualitative conclusion draw-robust (consistent with T4's robustness).
