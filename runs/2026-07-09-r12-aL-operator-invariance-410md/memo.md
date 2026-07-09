# R12 — ρ(A_ℓ) is not gauge-invariant (referee-pass correction C1)

**Date:** 2026-07-09. **Script:** `scripts/phase4_aL_operator.py` (corrected).
**Run:** `runs/2026-07-09-r12-aL-operator-invariance-410md`. **Supersedes the gauge
claim in** `runs/2026-06-30-ph4-aL-operator-410md` **and in S1**
(`runs/2026-06-29-ph4-gauge-410md`).

`diagnostics/operator_spectrum.py` asserted: *"its eigenvalues are gauge-invariant —
Symmetry A (`A → T A T⁻¹`) leaves the spectrum fixed"*. That holds for a **time-invariant**
SSM. Ours is **depth-indexed**: `A_ℓ, B_ℓ, C_ℓ` are free per layer, so the symmetry group
is a per-layer `T_ℓ ∈ GL(d_z)`

    A_ℓ → T_{ℓ+1} A_ℓ T_ℓ⁻¹,  B_ℓ → T_{ℓ+1} B_ℓ,  C_ℓ → C_ℓ T_ℓ⁻¹,  z_0 → T_0 z_0

which leaves every `Δ̂_ℓ` unchanged. Only a **global** `T_ℓ ≡ T` acts on `A_ℓ` by
similarity. Even the scalar subgroup `T_ℓ = c_ℓ I` rescales `ρ(A_ℓ)` by `c_{ℓ+1}/c_ℓ`.

> **Verdict.** Contractivity is a property of the **fitted min-Frobenius representative**
> that `λ_reg` selects, not of the model. The gauge-invariant object is the realised
> state path `C_ℓ (A_{ℓ-1}…A_{j+1}) B_j`, and it carries the whole mechanism claim.

## The audit (on the real L1 checkpoint, `ssm_seed0.pt`)

Apply a random output-preserving scalar gauge (`c_ℓ = exp(N(0,1))`):

| quantity | before | after | invariant? |
|---|---|---|---|
| model output `Δ̂` (max abs change) | — | **2.4e-7** | ✅ unchanged |
| `ρ(A_ℓ)` median | 0.430 | **0.259** | ❌ |
| `ρ(A_ℓ)` max | 0.881 | **4.909** | ❌ |
| non-contractive transitions (`ρ≥1`) | **0/23** | **5/23** | ❌ |
| state-path `‖C(∏A)B‖_F` ratio to gap 1 | [1, .485, .211, .089] | identical | ✅ (Δ ≤ 1.3e-8) |

A monotone gauge also drags `ρ_geomean` 0.300 → 0.231 (so `τ = −1/ln ρ_geo` moves
0.83 → 0.68 layers): the published memory constant was gauge-dependent too.

## What replaces it — the invariant readout

State-path kernel norm by depth gap, normalised to gap 1, and its e-folding `τ_inv`:

| fit | gap1 | gap2 | gap3 | gap4 | τ_inv (layers) |
|---|---|---|---|---|---|
| L1 (primary) | 1.000 | **0.485** | 0.211 | 0.089 | **1.15** |
| mlsae seed 0 | 1.000 | 0.441 | 0.200 | 0.081 | 1.21 |
| mlsae seed 1 | 1.000 | 0.418 | 0.169 | 0.071 | 1.17 |
| mlsae seed 2 | 1.000 | 0.450 | 0.206 | 0.085 | 1.20 |

**The state path halves per layer of depth gap** — stable across 3 seeds × 2 dictionaries,
and invariant by construction. This is the mechanism consistent with `k*≈1`, and the only
version of it we now quote. (Epistemics unchanged: it is estimated on the same data `k*`
was measured on, so it explains within the model class; the model-free evidence for
`k*≈1` remains the Step-2 per-depth AR sweeps.)

## Also retired: the "cosine 0.999" stability statistic

`spectrum_cosine` compares two **sorted non-negative** vectors, so it is bounded far from
0 by construction. Measured here:

- cross-seed sorted-modulus cosine: **0.999**
- **two independent random `|N(0,1)|` spectra of the same length: 0.999**

It cannot distinguish agreement from the shared monotone shape. The honest statistic is
`spectrum_rel_diff` (median relative difference of the sorted moduli):

- cross-seed **0.086**; cross-dictionary (L1 vs mlsae) **0.111**; per-layer ρ-profile
  correlation **0.543**.

So the modulus distribution reproduces to ~10% and the layer assignment much less so —
still a real invariance-of-the-fit claim, just an order of magnitude weaker than "0.999"
implied. (S1's *system-level* conclusion survives: reconstruction R² 0.880 every seed,
decoder subspace ~0.6, feature-set Jaccard 0.89, and now the invariant state-path decay.)

## Code changes

- `diagnostics/operator_spectrum.py`: corrected docstring; `spectrum_by_depth` relabelled
  as "in the fitted min-Frobenius gauge"; new `state_path_tau`, `apply_scalar_gauge`,
  `spectrum_rel_diff`; `spectrum_cosine` docstring warns about its own vacuity.
- `phase4_aL_operator.py`: leads with the invariant readout, runs the gauge audit, emits
  `gauge_audit` + `state_path_tau_layers_invariant` + `cross_seed_modulus_rel_diff_mean`
  + `cross_seed_modulus_cosine_random_null` into `tables/aL_stats.json`.
- `figures.aL_spectrum`: left panel titled gauge-dependent, right panel gauge-invariant.
- `tests/test_operator_spectrum.py`: +4 tests (gauge preserves output; ρ moves but the
  state path does not; the cosine is vacuous on random spectra; `state_path_tau` recovers
  a known e-folding). **12 green in that file, 208 repo-wide.**
