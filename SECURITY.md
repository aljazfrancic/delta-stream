# Security

This is a single-user mechanistic-interpretability research pipeline. It runs
locally, runs the language models **inference-only** (the only things it trains
are its own small decompositions and SAEs, on its own cached activations), and
loads public weights from a fixed, code-pinned set of Hugging Face Hub
repositories (EleutherAI Pythia, `openai-community` GPT-2, Qwen2-0.5B, the
`tim-lawson` MLSAE suites, `jbloom/GPT2-Small-SAEs-Reformatted`, `chanind` Qwen
SAEs, and — only with the optional `.[lens]` extra — the AlignmentResearch Tuned
Lens). No repository ID ever comes from user input or external data. It does not
accept untrusted input, does not run as a service, and is not deployed anywhere.

## Where weights are deserialized

What the torch/transformers advisories below hinge on is *which files* get
unpickled, so the complete list (verified 2026-11-06):

- **Language models** load through TransformerLens → `transformers`
  `from_pretrained`; every pinned model repository ships a single
  `model.safetensors` (no pickle, no sharded index).
- **SAE dictionaries** load with `safetensors.numpy.load_file`.
- **`torch.load` is called in exactly two places:**
  1. `src/layer_dynamics/checkpoint.py` (`load_ssm`) — reloads SSM checkpoints
     that this pipeline itself wrote with `save_ssm` under `runs/`. Local,
     self-produced files; never a downloaded checkpoint.
  2. The optional **`.[lens]` extra** — the `tuned-lens` package loads
     `params.pt` from the `AlignmentResearch/tuned-lens` Hub Space with
     `torch.load` (a full unpickle under torch 2.5). This is the **one path
     where a downloaded pickle is deserialized**: its safety rests on that
     single, fixed publisher. It is used only by the Tuned Lens baseline
     (Phase 1 Step 3 / Phase 4 Baseline C) and is not part of
     `scripts/reproduce.sh`.

## Dependabot advisories

### Resolved — lockfile updated (2026-11-06)

Transitive dependencies bumped to patched versions in `uv.lock`; no direct pin
changed and no numerically load-bearing library moved. Verified with the full
`scripts/reproduce.sh --with-runs` (208 tests; all 16 paper figures
byte-identical) plus offline smoke tests of the model, CounterFact, Tuned Lens,
and git-provenance loading paths.

| Package | From | To |
|---|---|---|
| aiohttp | 3.14.1 | 3.14.5 |
| certifi | 2026.6.17 | 2026.10.23 |
| filelock | 3.29.4 | 4.2.9 |
| GitPython | 3.1.50 | 3.2.1 |
| multidict | 6.7.1 | 6.9.1 |
| Pillow | 12.2.0 | 13.0.0 |
| setuptools | 82.0.1 | 84.2.1 |
| urllib3 | 2.7.0 | 2.8.0 |
| yarl (follows aiohttp) | 1.24.2 | 1.25.1 |

### Accepted risk — pinned dependencies

The remaining advisories sit in dependencies whose patched versions **cannot be
adopted without breaking documented, load-bearing constraints**, and whose
vulnerable code paths have no entry point here. The pins are an intentional risk
acceptance, not an oversight. Each alert is dismissed in the Security tab with a
comment linking this file.

### `torch == 2.5.1` (held at `~=2.5.0`)

- **Why pinned:** the dev box is a GTX 1050 (Pascal, compute capability 6.1).
  torch 2.6+ dropped Pascal/Maxwell (`sm_6x`) from the default CUDA wheels;
  2.5.x is the last line that still ships kernels this GPU can run. See the
  rationale in [pyproject.toml](pyproject.toml) and
  [src/layer_dynamics/device.py](src/layer_dynamics/device.py). Upgrading to the
  first patched version (2.6.0) removes GPU support entirely on this hardware.
- **Exposure:** the advisories are overwhelmingly *local-only* DoS / memory
  corruption triggered by crafting specific tensor ops (`pad_packed_sequence`,
  `unpack_sequence`, `lstm_cell`, quantization `scale`/`zero_point`, …) or
  compiling adversarial models — code paths this pipeline never feeds untrusted
  input. Alerts #4, #5, #8–#12 dismissed as tolerable risk.

#### CVE-2025-32434 / GHSA-53q9-r3pm-6pq6 and CVE-2026-24747 / GHSA-63cw-57p8-fm3p (reviewed 2026-11-06) — accepted

- **What they are:** RCE / memory corruption from a *malicious checkpoint file*
  loaded with `torch.load(..., weights_only=True)` — bugs in the restricted
  unpickler (fixed in 2.6.0 and 2.10.0 respectively).
- **Exposure:** the restricted unpickler is not this project's safety boundary:
  both `torch.load` sites (see [above](#where-weights-are-deserialized)) do a
  full unpickle, so what matters is the *source* of the file — the pipeline's
  own SSM checkpoints, or (optional extra only) the fixed Tuned Lens publisher.
  No attacker-supplied checkpoint is ever loaded. Alerts #6 and #135 dismissed
  as tolerable risk.

#### CVE-2025-3000 / GHSA-rrmf-rvhw-rf47 (reviewed 2026-07-20) — accepted

- **What it is:** local memory corruption in `torch.jit.script` (CWE-787;
  severity *low*). Vulnerable range `<= 2.12.1`, fixed in **2.13.0** — far
  beyond the 2.6.0 line that already drops Pascal, so the torch pin cannot
  reach it for the same hardware reason above.
- **Exposure:** exploitation requires calling `torch.jit.script` on
  attacker-influenced code; this pipeline never uses TorchScript
  (`grep -rn torch.jit src scripts tests` is empty) and takes no untrusted
  input. No entry point. Alert #15 dismissed as tolerable risk.

### `transformers == 4.57.6` (held at `>=4.44,<5`)

- **Why pinned:** `transformer-lens` 2.x (the residual-stream hook layer this
  whole pipeline is built on) uses `transformers.TRANSFORMERS_CACHE`, which was
  removed in transformers 5.0. Every fix below is 5.x-only and breaks
  TransformerLens.
- **CVE-2026-1839** (`Trainer._load_rng_state`; no training through the
  Trainer happens here, and the issue needs the Trainer) — not reachable.
  Alert #7 dismissed.

#### CVE-2026-4372 / GHSA-29pf-2h5f-8g72 (reviewed 2026-07-02) — accepted

- **What it is:** RCE via a *malicious model repository* — a crafted
  `config.json` sets `_attn_implementation_internal` to an attacker-controlled
  Hub repo ID, and `from_pretrained()` then downloads and executes the
  attacker's kernel code, bypassing `trust_remote_code`. Fixed only in
  **5.3.0**, which the `<5` pin cannot reach (every 4.x is in the vulnerable
  range, so an in-range bump does not help).
- **Exposure:** exploitation requires loading an **attacker-controlled model
  repo**. This pipeline loads only the fixed, code-pinned set of trusted
  repositories listed at the top of this file; no repo ID is derived from user
  input or external data, and no new repos are added without a code change
  that this file's re-review triggers cover. The malicious-`config.json`
  vector therefore has no entry point here. Alerts #1 (pyproject.toml) and #13
  (uv.lock) — same CVE, two manifests — dismissed as tolerable risk.

#### CVE-2026-5241 / GHSA-fgcw-684q-jj6r (reviewed 2026-07-14) — accepted

- **What it is:** arbitrary code execution via a *malicious model repository*,
  through the **LightGlue** (keypoint-matching vision model) loading path —
  `LightGlueConfig.__init__` reads `trust_remote_code` from the untrusted
  `config.json` and forwards it into a nested
  `AutoConfig.from_pretrained(keypoint_detector_config["_name_or_path"], ...)`,
  so a crafted repo can execute attacker code even when the caller passed
  `trust_remote_code=False` (CWE-829; CVSS 8.0). Verified present in the
  locked 4.57.6 (`models/lightglue/configuration_lightglue.py`). Fixed only in
  **5.5.0**, which the `<5` pin cannot reach.
- **Exposure:** same vector class as CVE-2026-4372 above, and specifically
  routed through the LightGlue config path. This pipeline loads only the
  fixed, code-pinned set of trusted text-LM and SAE repositories and never
  instantiates LightGlue (or any vision model). No entry point. Alerts #2 and
  #14 (same CVE, two manifests) dismissed as tolerable risk.

#### CVE-2026-9856 / GHSA-xrqw-3rrv-vx5w (reviewed 2026-11-06) — not used

- **What it is:** path traversal → arbitrary file write in the `save_pretrained()`
  methods of tokenizers/processors, via `chat_template` keys in a crafted
  `tokenizer_config.json`. Fixed in 5.10.0.
- **Exposure:** the pipeline never calls `save_pretrained` (or saves any
  tokenizer/processor), and loads only the trusted repositories above. Alerts #3
  and #49 (two manifests) dismissed as vulnerable code not used.

#### CVE-2026-80047 / GHSA-x9r9-c232-4q39 (reviewed 2026-11-06) — not used

- **What it is:** `GenerativePreTrainedModel.load_custom_generate()` caches a
  repository's `custom_generate/generate.py` to disk *before* the
  `trust_remote_code` consent check (execution itself stays gated). No fix yet.
- **Exposure:** the pipeline never uses Hugging Face `generate` / custom
  generation (all forward passes go through TransformerLens). Alert #63
  dismissed as vulnerable code not used.

### `datasets == 3.6.0` (held at `~=3.2`)

- **Why pinned:** `datasets` loads the inputs the paper's dataset was built
  from (`NeelNanda/counterfact-tracing`, `NeelNanda/pile-10k`); the only fix is
  **5.0.1**, two major versions up (4.0 changed dataset loading), so moving it
  means re-validating the frozen data pipeline for an unreachable issue.
- **CVE-2026-66007 / GHSA-379c-qx7v-6h59 (reviewed 2026-11-06) — not used:**
  path traversal through `file_name` metadata in *folder-based* builders
  (ImageFolder/AudioFolder-style), leaking files via `save_to_disk` /
  `push_to_hub`. The pipeline only calls `load_dataset` on the two fixed Hub
  datasets above and never uses folder builders, `save_to_disk`, or
  `push_to_hub`. Alert #65 dismissed as vulnerable code not used.

### `fsspec == 2025.3.0` (transitive; capped by `datasets` 3.x at `<=2025.3.0`)

- **CVE-2026-104851 / GHSA-27vj-qcqg-25rc (reviewed 2026-11-06) — not used:**
  server-side template injection → RCE in `ReferenceFileSystem` (Kerchunk
  `reference://` JSON). Nothing in the pipeline (or its use of `datasets`)
  opens a `reference://` filesystem. Fix (2026.6.0) is blocked by the
  `datasets` pin above. Alert #66 dismissed as vulnerable code not used.

### `accelerate == 1.14.0` (transitive, via transformers / TransformerLens / tuned-lens)

- **CVE-2026-69112 / GHSA-4j2p-28q2-5m79 (reviewed 2026-11-06) — not used:**
  path traversal / DoS through `weight_map` entries of a *sharded* checkpoint
  index in `load_checkpoint_in_model` / `load_checkpoint_and_dispatch`. No
  fixed release exists. The pipeline never calls these, passes no
  `device_map`, and every model it loads is a single-file safetensors
  checkpoint (no index). Alert #54 dismissed as vulnerable code not used.

## Re-review triggers

Revisit this acceptance if any of the following change:

- the project starts loading **untrusted** model checkpoints, runs as a
  **service**, or accepts untrusted input;
- a **new Hub repository** is added as a weight or data source (re-check it is a
  trusted publisher — CVE-2026-4372 makes the model repo itself the attack
  vector), or the Tuned Lens source changes;
- a model with a **sharded** checkpoint, Hugging Face `generate`, or
  `save_pretrained` enters the pipeline;
- the GPU is retargeted (newer card or CPU-only), freeing the torch pin;
- `transformer-lens` ships a transformers-5-compatible release, freeing the
  transformers pin;
- the `datasets` pin is lifted (then also bump `fsspec`).

New advisories in the pinned packages: review against this file, then dismiss
in the GitHub Security tab (Security → Dependabot alerts) as "risk is
tolerable" or "vulnerable code is not used", with a comment linking this file.
Advisories in unconstrained (transitive) packages: bump them in `uv.lock`
(`uv lock --upgrade-package <name>`) and re-run `scripts/reproduce.sh`.
