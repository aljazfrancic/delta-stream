# v1.1 post-hoc E1 + E2 — what "seed" means in the TopK stability claim

**Model:** Pythia-410m-deduped (L=24, D=1024), query-position residuals, N = 1,555.
**Status of this memo:** design and verdict criteria **frozen 2026-11-15, before any
E1/E2 run exists**. Everything here is **post-hoc** relative to the pre-registered
protocol (`docs/`, untouched) and to paper v1.0.

> **⚠ STATUS (2026-11-15): criteria frozen, runs NOT started.** The activation store
> (`data/pythia-410m-deduped/residual_stream.h5`), the recorded seed-0 dictionaries and
> their weights are not on the machine (`data/` is empty in a fresh clone), so neither
> experiment can run yet. The code path was exercised end to end on a synthetic store
> only. Results will be appended below under a dated heading; this section and the
> verdict table stay as written.

## Why

Paper v1.0 says TopK dictionaries make single-feature directions "seed-unidentifiable"
where matched L1 dictionaries are "perfectly stable" (match@0.7 = 0.08 / 1.00), and
warns pipelines to check their TopK features "survive a re-seed". What the code
measures is narrower:

- `scripts/step4_sanity.py` re-seeds the **SSM** (`LowRankSSM(dims, seed=s)`, `_fit`),
  re-fits it by alternating minimisation from the **same initial SAE codes** (the codes
  are then refined inside each fit), and compares each top-100 feature's write direction
  `D1[ℓ] @ D2[ℓ][j]` across the three fits, **paired by feature index**
  (`decompose_sanity.decoder_directions` / `direction_cosines`).
- The dictionary is trained **once** per mechanism
  (`runs/2026-06-29-ph4-e2-train-{l1,topk}-410md`, trainer default `--seed 0`). Nothing
  under `runs/` trains a 410m SAE under a second seed. The R4 train-only L1 variant is a
  second dictionary but not an independent draw: same seed, 80% of the same rows.
- The frozen plan asked for more: Phase-3 Step 4 specifies re-running "the whole
  pipeline with 3 different random seeds (for the SAE init and the optimiser)" with
  max-cosine matching. With frozen public SAEs the implementation specialised this to
  an SSM-only re-seed with identity matching (documented in the
  `decompose_sanity` module docstring, not in the paper's deviation list).

So match@0.7 answers *"are the SSM's per-feature write directions identified when it is
fit over a given dictionary's codes?"* It does not answer *"do independently trained
SAEs converge to the same dictionary?"* — which is what "re-seed" means to most SAE
readers. E1 measures the second; E2 repeats the first over n = 3 dictionary draws.

**Regime, stated up front.** The trained dictionaries have M = 8,192 latents fit
full-batch (Adam, lr 1e-3, 1,000 steps) on N = 1,555 vectors per layer — **M > N** —
with no dead-latent auxiliary loss (alive: 1,178 L1 / 660 TopK at the widest layer).
Whatever E1 finds is a statement about this regime, not about SAEs trained at scale.

## E1 — dictionary-level seed stability

`scripts/sae_seed_stability.py` (logic + tests:
`src/layer_dynamics/diagnostics/sae_seed_stability.py`,
`tests/test_sae_seed_stability.py`). Inputs per seed: the trainer's
`sources_*.npz` (`u`, `feat_index`, `alive_count`) and `.weights.pkl`
(`state_dicts[ℓ]["W_dec"]`, `[D, M]`, unit-norm columns, standardised space).
Seeds 0, 1, 2 per mechanism, recorded seed-0 hyperparameters (below). For each trained
layer 1..23 and each of the three seed pairs, on **alive latents only**, signed cosines:

- **best-match**: each latent's max cosine against every alive latent of the other
  seed, both directions, averaged with equal weight;
- **one-to-one (PW-MCC)**: optimal assignment (`scipy.optimize.linear_sum_assignment`)
  on **absolute** decoder cosines, mean over matched pairs, `n` = the smaller alive
  count — Song et al.'s PW-MCC restricted to alive latents. (Best-match cosines stay
  signed: SAE codes are non-negative, so `d` and `−d` are different features.)

**Primary statistic (fixed here):** match@0.7 over the **top-100 latents by activation
mass per layer**, best-match, averaged over layers, then over the three seed pairs.
"Stable" means primary ≥ 0.7 — the bar Gate 3 used.

**Secondary:** the same over all alive latents; mean max-cosine (top-100 and all);
one-to-one mean cosine; per-layer breakdown; alive counts.

**Nulls** (this repo has been burned by a statistic random inputs also satisfy — the
sorted-modulus cosine of 0.999):

1. `null_random` — a random isotropic unit-norm dictionary of matched size;
2. `null_xlayer_cross_seed` — the *other* seed's dictionary at the layer half the stack
   away: a different initialisation and the wrong layer.

**The primary statistic is declared uninformative for a mechanism** — no stable/unstable
verdict drawn, the row reported as such — if either null reaches 0.7, or if the real
value clears 0.7 by less than 0.2 over the worst null.

**Initialisation diagnostics (not nulls).** For a given seed every layer's SAE starts
from the *same* matrix (`sae_train.initial_decoder`). Two numbers say how much of a
trained dictionary is still that matrix: `diag_xlayer_same_seed` (same seed, distant
layer), and **init retention** — each alive latent's cosine with its own initial
column (mean, and fraction ≥ 0.7). These are reported whatever they show; if
dictionaries turn out initialisation-dominated, that is itself the finding about this
regime and bounds what cross-seed agreement could be.

## E2 — SSM-level stability across dictionary draws

`scripts/step4_sanity.py` on each new dictionary with the exact recorded arguments
(`scripts/run_e2.sh`: `--rank 32 --lambda-l1 1.0 --outer-iters 10 --batch-size 128
--device cuda --seed 0`), then `scripts/ssm_draw_stability.py` tabulates match@0.7 per
draw, mean and range per mechanism. Draw 0 is the **recorded** run
(`runs/2026-06-29-ph4-e2-step4-l1-410md`, `runs/2026-06-30-ph4-e2-step4-topk-410md`);
draws 1 and 2 are new. The retrained seed 0 is run too, as a reproduction check of the
recorded 1.00 / 0.08; it is listed and **not** counted as a draw.

## Seed-0 reproduction gate (before anything else is trusted)

The seed-0 weights are gone, so seed 0 is retrained to a **new path** and must
reproduce the tracked tables
(`runs/2026-06-29-ph4-e2-train-{l1,topk}-410md/tables/sources_trained_{mode}.json`):
M\* within 2% (1,178 L1 / 660 TopK), every layer's alive count within 5%, mean recon R²
within 0.003 (0.931 / 0.973). Tolerances fixed here. If the gate fails the run aborts:
the store or the hyperparameters differ and no cross-seed number means anything. The
learning rate is not in the v1.0 tables; 1e-3 is the trainer default since its first
commit and no tracked command overrides it (`scripts/run_review_r4_r6.sh` is the
tracked instance of the recipe) — the gate is what confirms it.

## Verdict table (frozen before the runs)

| E1: L1 | E1: TopK | What v1.1 says |
|---|---|---|
| stable | unstable | The warning holds at both levels in this regime. Keep it, state the regime, cite Paulo & Belrose as consistent and Song et al. as the contrasting large-data result. |
| unstable | unstable | Neither mechanism gives reproducible dictionaries here. "Perfectly stable L1" is restricted to SSM refits over a fixed dictionary; the practitioner warning covers downstream fits only; add to Limitations. |
| stable | stable | TopK dictionaries reproduce; the instability lives in fitting per-feature readouts over TopK codes. Narrow the warning to that. |
| unstable | stable | Suspect a bug first. If it survives checking, report it as found. |

**E2 reads independently:** if all three L1 draws are ≥ 0.7 and all three TopK draws
below, "it is the sparsity mechanism" stands at n = 3. Anything mixed is reported per
draw and the sentence is downgraded to what the draws show.

Whichever row applies, the result gets the prominence it would have had if it
flattered the paper. The claim wording is the author's call and is applied only after
the numbers are in.

## Commands

    scripts/run_v11_seeds.sh --smoke    # 2 layers x 50 epochs, end to end
    scripts/run_v11_seeds.sh            # full run, sequential, runs/v11-seeds-overnight.log

which expands to, per mechanism and seed s ∈ {0, 1, 2}:

    uv run python scripts/train_410m_saes.py --config configs/pythia-410m-deduped.yaml \
        --mode l1 --l1-coeff 0.5 --epochs 1000 --lr 1e-3 --seed $s \
        --out-npz data/pythia-410m-deduped/v11/sources_trained_l1_s$s.npz \
        --out runs/<date>-v11-train-l1-s$s-410md
    uv run python scripts/train_410m_saes.py --config configs/pythia-410m-deduped.yaml \
        --mode topk --k 32 --epochs 1000 --lr 1e-3 --seed $s \
        --out-npz data/pythia-410m-deduped/v11/sources_trained_topk_s$s.npz \
        --out runs/<date>-v11-train-topk-s$s-410md
    uv run python scripts/sae_seed_stability.py --config configs/pythia-410m-deduped.yaml \
        --out runs/<date>-v11-e1-sae-seed-stability --seed 0 \
        --l1 data/pythia-410m-deduped/v11/sources_trained_l1_s{0,1,2}.npz \
        --topk data/pythia-410m-deduped/v11/sources_trained_topk_s{0,1,2}.npz \
        --l1-ref runs/2026-06-29-ph4-e2-train-l1-410md/tables/sources_trained_l1.json \
        --topk-ref runs/2026-06-29-ph4-e2-train-topk-410md/tables/sources_trained_topk.json
    uv run python scripts/step4_sanity.py --config configs/pythia-410m-deduped.yaml \
        --out runs/<date>-v11-e2-step4-{l1,topk}-s$s-410md \
        --sources-npz data/pythia-410m-deduped/v11/sources_trained_{l1,topk}_s$s.npz \
        --rank 32 --lambda-l1 1.0 --outer-iters 10 --batch-size 128 --device cuda --seed 0
    uv run python scripts/ssm_draw_stability.py --config configs/pythia-410m-deduped.yaml \
        --out runs/<date>-v11-e2-ssm-draw-stability --seed 0 --draw ... --check ...

Every dictionary goes to an explicit per-seed path under `data/<model>/v11/`; the
recorded `data/<model>/sources_trained_<mode>.npz` is never written, and an existing
per-seed file is reused, not overwritten.

## Changes to the plan made before any real run (recorded so they are not post-hoc)

1. **The "same seed, different layer" comparison is a diagnostic, not a null.** A
   synthetic end-to-end smoke test (random sparse data, D = 64, M = 256) scored 1.00 on
   it for an L1 dictionary whose latents had barely left their shared initialisation
   (init retention 0.95), while the cross-seed statistic was at random level. A shared
   initialisation makes it a measure of init-dominance; it says nothing about whether
   the cross-seed statistic is vacuous. The cross-layer *null* is therefore taken
   across seeds, and init retention was added as the direct measurement.
2. **Seed 0 is retrained** (the recorded weights no longer exist) and gated on
   reproducing the tracked tables, as above.
3. The trainer now writes `lr` and `seed` into its tracked table and `seed` into the
   weights pickle (v1.0 recorded neither).
4. **PW-MCC takes absolute cosines** (amended the same day, still before any run, after
   reading Song et al.'s definition): the one-to-one statistic was first written with
   signed cosines, which is not the published metric it is reported "for comparability"
   with. The primary statistic and the verdict table are unaffected — both use the
   signed best-match cosine.

## Results

*Pending — no runs yet (see the status note at the top).*
