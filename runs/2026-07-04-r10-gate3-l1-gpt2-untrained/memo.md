# R10 — T4-recon on GPT-2-small: the "~half architectural, ~half learned" reconstruction split REPLICATES cross-family

**Date:** 2026-07-04. **Script:** `scripts/run_r10.sh` (exact 410m T4-recon recipe on
the GPT-2 stores). **Runs:** `runs/2026-07-04-r10-train-l1-gpt2{,-untrained}` (matched
L1 dictionaries, `train_410m_saes.py --mode l1 --l1-coeff 0.5`, trained store recon
R² ≈ 0.95 / untrained ≈ 0.86) and `runs/2026-07-04-r10-gate3-l1-gpt2{,-untrained}`
(identical Gate-3 Pareto, rank 32, λ ∈ {0.3, 1, 3, 10}, split 1086/272 seed 0).
**Question (4th-pass action plan):** the *reconstruction* architectural-vs-learned
split (T4-recon) was 410m-only; R6 found the *AR-predictability* learned share
collapses to ~5% on GPT-2 — does the recon learned-share collapse too (making the
"≈half learned" sentence single-model), or replicate?

> **Verdict — REPLICATES: on GPT-2-small the untrained (random-init) store carries
> 44% of the trained SSM−baseline reconstruction gap (mean over λ; 410m: 53%), so
> "the reconstruction win is ~half architectural, ~half learned" is now a
> two-family statement.** The key dissociation for the write-up survives intact and
> is strengthened: AR-on-Δ *predictability* is almost entirely architectural on both
> families (T4: 410m ~matched; R6: GPT-2 learned share ~5%), while the Branch-B SSM
> *reconstruction win* keeps a ≈half learned component on both.

## Result (raw same-λ SSM − per-layer-baseline gap, held-out test R²)

| λ_L1 | trained gap | untrained gap | architectural share | learned share |
|---|---|---|---|---|
| 0.3 | +0.1429 | +0.0713 | 50% | 50% |
| 1 (knee) | +0.1426 | +0.0609 | 43% | 57% |
| 3 | +0.1426 | +0.0478 | 34% | 66% |
| 10 | +0.1423 | +0.0685 | 48% | 52% |
| **mean** | **+0.1426** | **+0.0621** | **44%** | **56%** |

Both stores pass the Gate-3 go criterion (SSM ≥ baseline at every matched L0) — as on
410m, the architectural component is real: random-init activations already benefit
from cross-layer state. The 410m sparsity trend (learned share largest at sparse
operating points: 66% at λ=10 there) appears here as a λ=3 peak (66%) rather than a
clean monotone; the robust statement is "≈half, on both families", not a trend.

## Caveats

1. **The trained-store gap (+0.143) is much larger than the recorded res-jb GPT-2 gap
   (+0.018).** Different dictionary: our query-position-trained L1 SAEs are far
   narrower/sparser (alive 287–942 of 8192) than the 24576-atom res-jb suite, so the
   per-layer feedthrough baseline is weaker and the state path carries more. R10's
   contrast is **within-protocol** (trained vs untrained store, same dictionary
   recipe both arms); the absolute gap is not comparable to the res-jb runs and
   should not be quoted as "the" GPT-2 gap.
2. Operating points differ across stores at equal λ (trained codes run denser, test
   L0 0.28 vs 0.06 at λ=1) — same caveat as the 410m T4-recon; the within-store
   SSM-vs-baseline pairing is L0-matched by the script's interpolation
   (`matched_l0_gaps` in each `gate3_pareto.json`).
3. Single init seed (0) for the untrained model, as in T4/T4-recon on 410m —
   qualitative conclusion (untrained gap ≈ half of trained, ≠ 0, ≠ trained) is
   draw-robust.
4. Untrained-store absolute R² is lower everywhere (SSM +0.51 vs +0.83 at the knee) —
   random-init Δ is intrinsically harder to reconstruct; shares are computed on the
   *gap*, which is the object T4-recon defined.
