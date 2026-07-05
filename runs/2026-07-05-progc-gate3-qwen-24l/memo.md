# Programme C — full-depth Qwen: the fig06 point was suppressed by SAE coverage, not model depth

**Date:** 2026-07-05. **Design:** the fig06 depth-scaling figure plots Qwen2-0.5B at
x=12 because the only public L1 suite (chanind) covers blocks 0..11 of the 24-layer
model — so the 4th-pass review flagged that the figure cannot tell "gap tracks
covered depth" from "gap tracks model depth". This resolves it with a
**dictionary-controlled coverage contrast**: train our own L1 SAEs (E2/R10 protocol,
R4-clean `--exclude-test`) on the **same cached Qwen activations** at (a) all 24
layers and (b) the first 12 layers only, then run the identical Gate-3 Pareto on each.
Same model, same rows, same dictionary recipe, same seed — the *only* difference is
SAE coverage. **Runs:** `runs/2026-07-05-progc-gate3-qwen-{24l,12l}`.

> **Verdict.** Full 24-layer coverage roughly **doubles** the SSM's cross-layer
> reconstruction advantage over the no-state baseline: **knee gap at λ=1 is +0.072
> (24L) vs +0.043 (12L prefix)**. So the old chanind-12L fig06 point (+0.041)
> *understated* Qwen — it was a coverage artefact, not Qwen's true depth gain. At
> matched full depth the Qwen gap (**+0.072**) even **exceeds** Pythia-410m's at the
> same 24 layers (+0.050), consistent with the "cross-layer propagation is
> load-bearing at depth" reading and removing the figure's ambiguity in the
> depth-tracking direction.

## Knee gaps (SSM − no-state baseline, held-out test R² at λ_L1=1)

| coverage | SSM test R² | baseline test R² | **knee gap** | SSM test L0 |
|---|---|---|---|---|
| **24 layers (full)** | 0.8330 | 0.7612 | **+0.0718** | 0.031 |
| **12 layers (prefix)** | 0.8367 | 0.7940 | **+0.0427** | 0.019 |

Full λ sweep in each run's `tables/pareto.csv`. Both trained-L1 dictionaries
reconstruct their activations well (recon R² 0.96–0.97, R4-clean). The 12L prefix's
SSM collapses at λ≥3 (test R² 0.65 → 0.08 — the short 12-layer state over-sparsifies),
so its frontier knee is genuinely at λ=1; the 24L run is healthy across the sweep.

## Implication for fig06 (D2 decision — figure left unchanged)

Per the write-up plan the figure itself is **not** modified here (a plotting/D2
decision): the point is recorded so the caption can cite it. The honest options are
(a) keep Qwen at x=12 with the "12L prefix of a 24L model" annotation *and* add the
full-coverage +0.072 in the caption as the coverage-corrected value, or (b) move the
Qwen point to x=24 at +0.072. Either way the coverage contrast closes the 4th-pass
ambiguity: the sub-410m Qwen gap in the old figure was SAE-coverage-limited, and at
true depth Qwen sits *above* the Pythia depth line, not below it.

## Scope / caveats

Single seed 0 (seeds 1–2 optional hardening if the contrast is ever contested; the
+0.072 vs +0.043 separation is well outside per-λ noise here). Trained-L1 sources on
cached Qwen activations — an on-box run (the "cloud" tag was a mis-file, like E3).
The `gate3_pareto` CSV now persists incrementally (this box froze 3× on 2026-07-05
under memory pressure and lost whole runs that wrote only at the end).
