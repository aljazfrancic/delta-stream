# E8 — Residual-write CLT vs SSM on Qwen2-0.5B (cross-model parsimony control)

**Purpose.** E3 (the §4.2 headline) showed on Pythia-410m that the depth-indexed
state-space SSM *ties* a free `O(L²)` residual-write CLT held-out at ~6× fewer params.
That was a single-model number. E8 re-runs the **identical** comparison
(`scripts/phase4_baseline_clt.py`, 180 epochs, batch 256, rank 32, seed 0,
reg-grid 1e-4…1e-1) on **Qwen2-0.5B** (a third family — RoPE/GQA/SwiGLU) using its
cached **chanind L1** sources (`data/Qwen/Qwen2-0.5B/sources_qwen.npz`, the 12-layer
SAE-coverage prefix, N=2540). Run: `runs/2026-07-01-ph4-e8-clt-qwen/`.

> **Small script change:** `phase4_baseline_clt.py` now truncates `delta` to the
> sources' layer count when the SAE covers only a prefix (mirrors the Gate-3 pipeline's
> "partial SAE coverage → truncating to first 12 layers"). Needed because the Qwen h5 is
> `(2540, 25, 896)` → 24 Δ-layers, but the sources are 12. Guarded (`delta.shape[1] > L`)
> so the 410m/full-coverage path is unchanged.

## Result (best λ_reg per model, held-out test R²)

| model | params | test R² | overfit | vs per_layer |
|---|---|---|---|---|
| per_layer (no cross-layer) | 0.93M | +0.7681 | +0.0253 | — |
| **ssm** (structured cross-layer) | 1.84M | **+0.8050** | +0.0333 | **+0.037** |
| clt_full (free `O(L²)` decoders) | 5.97M | +0.8259 | +0.0564 | +0.058 |

**SSM − CLT held-out gap: −0.0209** (CLT at **3.24×** the SSM's params).

## Read — the parsimony story is DEPTH-GRADED, not a universal tie

| | Pythia-410m (24L, E3) | Qwen2-0.5B (12L, E8) |
|---|---|---|
| SSM − CLT held-out gap | **−0.0035** (tie) | **−0.0209** (CLT edges it) |
| CLT param multiple vs SSM | 6.27× | 3.24× |

- **What generalises cleanly:** the SSM beats the no-cross-layer `per_layer` baseline by
  **+0.037** on a third family → structured cross-layer propagation is load-bearing in
  Qwen too (consistent with the depth-scaling gap +0.043@12L and the H1/H2 replication).
- **What is depth-dependent:** the SSM's structural constraint is nearly free at 24L
  (costs 0.0035, free per-pair decoders redundant, 6.3× param saving) but costs ~0.021 at
  12L, where the free CLT decoders still buy real reconstruction and the `O(L²)` param
  blowup is milder (3.24×). So the SSM **closes the gap to an unconstrained CLT as depth
  grows** — coherent with the paper's "cross-layer structure is more compressible at depth"
  thesis, but it means the clean "ties a free CLT" claim is a **24L statement, not universal**.

## Write-up guidance

Report E3 as the primary (24L tie at 6.3× fewer params) and E8 as the cross-family
depth control: *"On a third family at half the depth the free CLT still edges the SSM by
~0.02 R² at 3.2× the params; the near-tie is a depth phenomenon — the shared depth-indexed
form becomes adequate exactly where the free `O(L²)` parameterisation becomes redundant and
expensive."* Do **not** state the tie as model-general.

**Caveats.** Single seed (0), matching E3. Qwen source = L1 (chanind) vs 410m TopK (mlsae) —
different dictionary, so this is a same-method / different-model+dictionary point, not a
controlled dictionary swap; the direction (SSM ≫ per_layer; CLT modestly > SSM) is the
robust content. 12L is Qwen's SAE-coverage prefix of a 24L model, not a full-depth 12L model.
