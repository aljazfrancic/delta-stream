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

> **STATUS UPDATE (2026-11-15, evening — the note above is the morning's record, left as
> written).** The store has been **rebuilt and verified bit-level** against the tracked
> records (`runs/2026-11-15-v11-recollect-verify-410md/memo.md`; the author signed off on
> re-collecting, there being no backup). A **smoke test on the real store passed end to
> end** (3 layers × 50 epochs through training, E1, six SSM-refit runs and the E2
> summary — its numbers are from deliberately under-trained dictionaries and are **not
> results**). A **full-length preview of the seed-0 gate passed exactly**: seed 0 retrained
> at 1,000 epochs on layers 1–2 gives alive counts 345 / 777 (L1) and 152 / 190 (TopK)
> and reconstruction R² identical to the recorded tables to the last digit, which also
> confirms the unrecorded learning rate (1e-3) and that training is deterministic on this
> machine. **The full run has still not been started**; criteria and verdict table below
> are unchanged. Launch: `scripts/launch_v11_overnight.sh` (detached, resumable).

> **STATUS UPDATE (2026-11-16 — the two notes above are earlier records, left as
> written).** The full run is **complete** (00:24–06:50 CET, run date 2026-11-16, commit
> `a225592`, no failed step). Both hard stops passed exactly. Results are under
> *Results → 2026-11-16* below. Criteria and verdict table are unchanged.

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

    scripts/launch_v11_overnight.sh check    # preflight only; starts nothing
    scripts/launch_v11_overnight.sh smoke    # 3 layers x 50 epochs end to end + gate preview
    scripts/launch_v11_overnight.sh          # full run, detached, runs/v11-seeds-overnight.log
    scripts/launch_v11_overnight.sh status   # / stop  (resumable: start again)

(the launcher runs `scripts/run_v11_seeds.sh [--smoke]` as a systemd user unit so the
run outlives the terminal that started it)

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
4. **The store is a rebuild, verified identical** (2026-11-15): see the status update at
   the top. Two hard stops were added to the runner accordingly — store verification and
   the seed-0 reproduction gate — plus atomic artefact writes and completion markers
   after a smoke test killed mid-write left a truncated weights file.
5. **PW-MCC takes absolute cosines** (amended the same day, still before any run, after
   reading Song et al.'s definition): the one-to-one statistic was first written with
   signed cosines, which is not the published metric it is reported "for comparability"
   with. The primary statistic and the verdict table are unaffected — both use the
   signed best-match cosine.

## Results

*Pending — no runs yet (see the status note at the top).*

(The line above is the pre-run placeholder, left as written.)

### 2026-11-16 — full run

**Run.** `scripts/launch_v11_overnight.sh` (systemd user unit), 00:24–06:50 CET, run date
2026-11-16, commit `a225592`, no failed step; log `runs/v11-seeds-overnight.log`
(git-ignored). Each dictionary took 43.8–44.7 min. Each SSM refit took 23–24 min (L1) or
16–17 min (TopK), against the 30–60 min estimate. The E1 and E2 summary `run.log`s say
`(dirty)`: the only changes to the tree at that point were this sweep's own new,
untracked run directories. No tracked file was modified.

**Hard stops.**
- *Store:* verified on 2026-11-15
  (`runs/2026-11-15-v11-recollect-verify-410md`), not re-run.
- *Seed-0 reproduction gate:* **passed exactly.** Retrained seed 0 gives per-layer alive
  counts identical to the tracked tables on all 23 layers (M\* 1,178 L1 / 660 TopK) and
  reconstruction R² identical on every layer (difference 0; means 0.9311 / 0.9726).
- *Seed-0 SSM-refit check:* `step4_sanity.py` over the retrained seed-0 dictionaries
  writes `sanity_stats.json`, `seed_pairs.csv` and `feature_themes.json`
  **byte-identical** to the recorded runs (`runs/2026-06-29-ph4-e2-step4-l1-410md`,
  `runs/2026-06-30-ph4-e2-step4-topk-410md`). The v1.0 statistic (1.00 / 0.08) is
  therefore reproduced, not just approached.

**E1 — dictionary-level** (`runs/2026-11-16-v11-e1-sae-seed-stability/tables/`; seeds 0, 1,
2; 23 layers; three seed pairs)

| statistic | L1 | TopK |
|---|---|---|
| **primary**: match@0.7, top-100 by mass, best-match | **0.254** (pairs 0.251 / 0.255 / 0.255) | **0.035** (pairs 0.035 / 0.040 / 0.030) |
| primary, per-layer range (mean over pairs) | 0.08 (ℓ18) – 0.42 (ℓ6) | 0.003 (ℓ8) – 0.15 (ℓ18) |
| null_random / null_xlayer_cross_seed (primary) | 0.000 / 0.000 | 0.000 / 0.000 |
| mean max-cosine, top-100 (null_random) | 0.515 (0.098) | 0.313 (0.091) |
| match@0.7, all alive latents | 0.033 | 0.037 |
| mean max-cosine, all alive | 0.213 | 0.354 |
| one-to-one PW-MCC (abs cos), alive | 0.168 (null 0.101) | 0.337 (null 0.094) |
| diag_xlayer_same_seed, primary | 0.055 | 0.014 |
| init retention, all alive: mean cos / frac ≥ 0.7 | 0.876 / 0.880 | 0.401 / 0.323 |
| init retention, top-100: mean cos / frac ≥ 0.7 | 0.475 / 0.308 | 0.570 / 0.558 |
| alive per layer, min/mean/max over the 3 seeds | 267–1,244 (mean ≈ 820) | 152–686 (mean ≈ 387) |

Both rows are **informative** under the frozen rule: worst null 0.000 for each, so
neither null reaches 0.7. The 0.2-margin clause applies only to a real value that
clears 0.7, and neither does. **Neither mechanism is stable.**

**Verdict row: unstable | unstable**: "Neither mechanism gives reproducible
dictionaries here."

Notes for the write-up (observations, not new criteria):
- **The two mechanisms are ranked differently by different statistics.** On the primary
  statistic L1 is well above TopK (0.25 vs 0.035). Over all alive latents they are equal
  (0.033 / 0.037). On one-to-one PW-MCC TopK is higher (0.34 vs 0.17). "L1 dictionaries
  are more reproducible than TopK" therefore holds for the high-mass latents only and
  should not be stated more broadly.
- **The L1 dictionaries are largely initialisation-dominated.** 88% of alive L1 latents
  keep cosine ≥ 0.7 with their own seed-specific initial column. Different seeds start
  from different matrices, so these latents cannot match across seeds, which explains
  the low all-alive figures. The high-mass L1 latents move further (31% retained). For
  TopK the high-mass latents are the *more* init-retaining ones (56%). As the memo
  anticipated, this is a statement about the regime (M = 8,192 > N = 1,555,
  full-batch, 1,000 Adam steps at lr 1e-3, no dead-latent auxiliary loss), not about
  SAEs trained at scale.
- Every real cosine statistic sits well above its random null, so the dictionaries
  share *some* structure across seeds, but on no layer does either mechanism come near
  0.7 on the primary statistic.

**E2 — SSM-level across dictionary draws**
(`runs/2026-11-16-v11-e2-ssm-draw-stability/tables/`; per-draw runs
`runs/2026-11-16-v11-e2-step4-{l1,topk}-s{1,2}-410md`)

| draw (dictionary seed) | L1 match@0.7 (SSM-seed pairs) | TopK match@0.7 (SSM-seed pairs) |
|---|---|---|
| 0 (recorded run) | 1.00 (1.00–1.00) | 0.08 (0.07–0.09) |
| 1 | 1.00 (1.00–1.00) | 0.057 (0.05–0.06) |
| 2 | 1.00 (1.00–1.00) | 0.08 (0.07–0.09) |
| mean [min, max] | **1.00 [1.00, 1.00]** | **0.072 [0.057, 0.08]** |
| retrained seed 0, *check, not a draw* | 1.00, byte-identical tables | 0.08, byte-identical tables |

Mean cosine of the paired write directions: L1 0.950 / 0.951 / 0.951, TopK 0.397 / 0.326 /
0.394. SSM train R²: L1 0.875–0.877, TopK 0.859–0.864.

All three L1 draws are ≥ 0.7 and all three TopK draws are below, so under the frozen
rule **"it is the sparsity mechanism" stands at n = 3** for the SSM-level statistic.

**Together.** The v1.0 number is correct and reproducible as a statement about
*SSM refits over a fixed dictionary*: identified over L1 codes, unidentified over TopK
codes, on each of three independently trained dictionaries per mechanism. It is **not**
a statement about the dictionaries. Re-seeding the SAE itself gives non-reproducible
dictionaries under both mechanisms in this regime. L1 retains more of its high-mass
directions than TopK, but it is nowhere near "perfectly stable". Per the frozen table
(row 2, with E2 standing): restrict "perfectly stable L1" to SSM refits over a fixed
dictionary, scope the practitioner warning to downstream fits, and add dictionary-level
non-reproducibility (both mechanisms, this regime) to Limitations. The exact wording is
the author's call and is not applied here.
