# Security

This is a single-user mechanistic-interpretability research pipeline. It runs
locally, performs **inference only**, and loads **trusted, public model weights**
(EleutherAI Pythia and the matching Tuned Lens) from the Hugging Face Hub as
safetensors. It does not accept untrusted input, does not run as a service, and
is not deployed anywhere.

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

## Re-review triggers

Revisit this acceptance if any of the following change:

- the project starts loading **untrusted** model checkpoints, runs as a
  **service**, or accepts untrusted input;
- the GPU is retargeted (newer card or CPU-only), freeing the torch pin;
- `transformer-lens` ships a transformers-5-compatible release, freeing the
  transformers pin.

To stop these specific alerts from recurring, dismiss them in the GitHub
Security tab (Settings → Security → Dependabot alerts) as "risk is tolerable",
linking this file.
