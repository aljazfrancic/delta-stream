# A_ℓ operator characterization — the learned depth-propagation operator (Phase 4)

**Date:** 2026-06-30. **Model:** pythia-410m-deduped (L=24, D=1024). **Checkpoints:**
clean **L1-sourced** SSM (`runs/2026-06-30-ph4-e2-checkpoint-l1-410md`, d_z=r=32, seed 0)
as primary; **mlsae-sourced** seeds 0/1/2 (`…ph4-checkpoint-410md`, `…ph4-gauge-410md`)
for stability. Pure operator analysis — no model forward; runs in seconds on CPU.
Code: `src/layer_dynamics/diagnostics/operator_spectrum.py`, `scripts/phase4_aL_operator.py`
(+4 tests). Run: `runs/2026-06-30-ph4-aL-operator-410md/`.

> **Why.** Phase-4's causal existence proof is NULL (recall is a distributed,
> mass-driven code — E1/E2), so the decomposition's demonstrated value is
> reconstruction. This is the **positive structural readout** that balances it: a
> description of the cross-layer state-transition operator `A_ℓ` (z_ℓ→z_{ℓ+1}) the SSM
> learned. Its **eigenvalues are gauge-invariant** (Symmetry A: A→TAT⁻¹ fixes the
> spectrum), so the spectrum is a property of the *model*, not of the fit — exactly
> what S1 found seed-stable.

## Three headline results

**1. The learned state is contractive at every depth.** Spectral radius `ρ(A_ℓ)`:
median **0.43**, max **0.88**, geometric mean **0.30** — and **0/23 layers have ρ≥1**.
Mean `|λ|` per layer ≈ 0.08 (most of the 32 modes are strongly decaying; only a few
near the spectral radius carry state). So the depth dynamics never amplify: information
injected into the latent state *forgets* with depth. The implied spectral memory is
`τ = −1/ln(ρ_geomean) ≈ **0.83 layers**.

**2. The operator memory horizon is the mechanism behind k*≈1.** The state-path kernel
`‖C_ℓ (∏A) B_j‖_F` (a source `u_j` reaching Δ_ℓ through the state, gap g=ℓ−j),
normalised to gap 1, decays:

| gap g | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| ratio | 1.00 | **0.485** | 0.211 | 0.089 | 0.036 | 0.015 |

≈ halving per extra layer. This is the SSM-native explanation of the **reconstruction
k\*≈1** (Phase-2 Step 2: lags ≥2 add only +0.018 R²): a contractive `A` means a source
two layers back contributes ~half, three layers back ~a fifth — so a first-order
(Markov-in-depth) read captures essentially all the state's reconstruction value.
Consistent with E5 (per-layer feedthrough carries most; propagation adds +0.04–0.05).

**3. The spectrum is seed-stable and dictionary-robust.** Cross-seed sorted-modulus
cosine across mlsae seeds 0/1/2 = **0.999** (pairs 0.999/0.998/0.999) — the eigenvalue
spectrum is essentially identical across re-fits (the gauge-invariant object; cleaner
than S1's raw-`D` 0.36 / impulse 0.29, which are *not* gauge-invariant). Cross-**dictionary**
L1↔mlsae sorted-modulus cosine = **0.993**: the contractive spectrum is the same whether
the sources are trained-L1 or pretrained-mlsae TopK. (Per-layer ρ-*profile* correlation
L1↔mlsae is only **0.54** — the *distribution* of moduli is shared, but the layer-by-layer
ρ assignment differs between dictionaries; ρ peaks in the middle layers, ~0 at the very
ends, for both.)

## What to claim, and how

- **Positive, robust, gauge-invariant:** "The learned depth-propagation operator is
  contractive at every layer (ρ<1, median 0.43); its ~1-layer memory horizon is the
  mechanism behind the reconstruction-effective k\*≈1, and its eigenvalue spectrum is
  seed-stable (0.999) and source-dictionary-robust (0.993)."
- This is a **model property, not a fit artefact** (gauge-invariance + cross-seed +
  cross-dictionary), so it is safe to put in the paper as the structural counterpart to
  the (negative) causal result. It also coheres the story: H2-fails (depth-varying) +
  contractive + short memory ⇒ a first-order, fully depth-indexed SSM is exactly the
  right abstraction (Branch B), and the free per-pair CLT decoders are redundant (E3).
- **Honest scope:** the per-layer ρ profile is only moderately shared across dictionaries
  (0.54) — report the *spectrum-level* invariance (0.99), not a claim that each layer's
  ρ is pinned. Figures: `figures/{eig_plane,rho_by_depth,modulus_hist,state_decay}.png`.
