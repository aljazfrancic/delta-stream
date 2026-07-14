# Security

This is a single-user mechanistic-interpretability research pipeline. It runs
locally, performs **inference only**, and loads **trusted, public model weights**
from a fixed, code-pinned set of Hugging Face Hub repositories (EleutherAI
Pythia + SAEs, the matching Tuned Lens, `tim-lawson` MLSAE suites,
`jbloom/GPT2-Small-SAEs-Reformatted`, `chanind` Qwen SAEs, `openai-community`
GPT-2, Qwen2-0.5B) as safetensors. No repository ID ever comes from user input
or external data. It does not accept untrusted input, does not run as a
service, and is not deployed anywhere.

## Dependabot / known advisories — accepted risk

GitHub Dependabot flags advisories in two pinned dependencies. As of the last
review the patched versions **cannot be adopted without breaking documented,
load-bearing constraints**, and the vulnerable code paths are **not used** by
this project. The pins are an intentional risk acceptance, not an oversight.

### `torch == 2.5.1` (held at `~=2.5.0`)

- **Why pinned:** the dev box is a GTX 1050 (Pascal, compute capability 6.1).
  torch 2.6+ dropped Pascal/Maxwell (`sm_6x`) from the default CUDA wheels;
  2.5.x is the last line that still ships kernels this GPU can run. See the
  rationale in [pyproject.toml](pyproject.toml) and
  [src/layer_dynamics/device.py](src/layer_dynamics/device.py). Upgrading to the
  first patched version (2.6.0) removes GPU support entirely on this hardware.
- **Exposure:** the advisories are overwhelmingly *local-only* DoS / memory
  corruption triggered by crafting specific tensor ops or compiling adversarial
  models with TorchInductor — code paths this pipeline never exercises. The one
  meaningful class (RCE via `torch.load`, PYSEC-2025-41) requires loading an
  untrusted checkpoint; this project does not call `torch.load` and loads only
  trusted EleutherAI weights via safetensors. No realistic exposure.

### `transformers == 4.57.6` (held at `>=4.44,<5`)

- **Why pinned:** `transformer-lens` 2.x (the residual-stream hook layer this
  whole pipeline is built on) uses `transformers.TRANSFORMERS_CACHE`, which was
  removed in transformers 5.0. The only fix version (5.0.0rc3) is out of range
  and breaks TransformerLens.
- **Exposure:** both advisories require code paths this project does not use —
  `Trainer._load_rng_state` (CVE-2026-1839; no training happens here, and the
  issue is only exploitable with torch < 2.6 *and* the Trainer) and X-CLIP
  checkpoint conversion (PYSEC-2025-217). Neither is reachable.

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
  vector therefore has no entry point here. Accepted; both Dependabot alerts
  (#13 uv.lock, #14 pyproject.toml — same CVE, two manifests) dismissed as
  tolerable risk with a comment linking this file.

#### CVE-2026-5241 / GHSA-fgcw-684q-jj6r (reviewed 2026-07-14) — accepted

- **What it is:** arbitrary code execution via a *malicious model repository*,
  through the **LightGlue** (keypoint-matching vision model) loading path —
  `LightGlueConfig.__init__` reads `trust_remote_code` from the untrusted
  `config.json` and forwards it into a nested
  `AutoConfig.from_pretrained(keypoint_detector_config["_name_or_path"], ...)`,
  so a crafted repo can execute attacker code even when the caller passed
  `trust_remote_code=False` (CWE-829; CVSS 8.0). Verified present in the
  locked 4.57.6 (`models/lightglue/configuration_lightglue.py`). Fixed only in
  **5.5.0**, which the `<5` pin cannot reach (every 4.x with LightGlue is in
  the vulnerable range, so an in-range bump does not help).
- **Exposure:** same vector class as CVE-2026-4372 above — exploitation
  requires loading an **attacker-controlled model repo**, and specifically one
  routed through the LightGlue config path. This pipeline loads only the
  fixed, code-pinned set of trusted text-LM and SAE repositories listed at the
  top of this file and never instantiates LightGlue (or any vision model); no
  repo ID is derived from user input or external data. The vector has no entry
  point here. Accepted; both Dependabot alerts (#15 uv.lock, #16
  pyproject.toml — same CVE, two manifests) dismissed as tolerable risk with a
  comment linking this file.

## Re-review triggers

Revisit this acceptance if any of the following change:

- the project starts loading **untrusted** model checkpoints, runs as a
  **service**, or accepts untrusted input;
- a **new Hub repository** is added as a weight source (re-check it is a
  trusted publisher — CVE-2026-4372 makes the model repo itself the attack
  vector);
- the GPU is retargeted (newer card or CPU-only), freeing the torch pin;
- `transformer-lens` ships a transformers-5-compatible release, freeing the
  transformers pin.

To stop these specific alerts from recurring, dismiss them in the GitHub
Security tab (Settings → Security → Dependabot alerts) as "risk is tolerable",
linking this file.
