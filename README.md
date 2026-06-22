# delta-stream

Code for **[Structured Cross-Layer Propagation of Features in Transformers](docs/README.md)** —
an empirical study of whether the layer-to-layer residual-stream update
`Δ_ℓ = v_{ℓ+1} − v_ℓ` has structured, predictable, approximately stationary dynamics across depth.

- **Research proposal:** [docs/README.md](docs/README.md)
- **Pre-registered protocol (frozen):** [docs/implementation-plan/](docs/implementation-plan/README.md)
- **Glossary:** [docs/glossary.md](docs/glossary.md)
- **Live progress tracker:** [progress.md](progress.md) — the only mutable record of progress.

## Layout

```
src/layer_dynamics/   importable package (run-logging backbone, later: diagnostics & decomposition)
scripts/              CLI run scripts; each takes --config / --out / --seed
configs/              YAML run configs
data/                 datasets and activation stores (git-ignored)
runs/                 per-run outputs: config.yaml, meta.json, figures/, tables/ (git-ignored)
figures/             paper figures regenerated from saved run intermediates (git-ignored)
tests/                unit tests for the backbone
```

## Setup

Managed with [uv](https://docs.astral.sh/uv/) on **Python 3.12** (the system 3.14 has no
torch wheels). One command does everything — installs uv, fetches the interpreter, syncs the
pinned deps, and verifies the GPU:

```bash
scripts/setup.sh          # then: source .venv/bin/activate, or prefix commands with `uv run`
```

Optional extras: `.[cca]` (regularised CCA for Phase 1 Step 4), `.[lens]` (Tuned Lens baseline).

### Retargeted to the dev box

This box is a **GTX 1050 (3 GB, Pascal sm_61)**, used for development and the full Phase 1
diagnostic. Decisions baked into [pyproject.toml](pyproject.toml) / [src/layer_dynamics/device.py](src/layer_dynamics/device.py):

- **Pythia-410m stays the scientific primary** (L=24) — collection runs on the GPU in fp32
  at batch 4 (peaks ~2 GB; verified). **Pythia-70m** is the fast dev lane (L=6, seconds/iter).
- **torch held on the 2.5 line + cu124 wheels** — 2.6+ drops Pascal from the default builds.
- **transformers `<5`** — transformer-lens 2.x needs the pre-5.0 cache API.
- **Gemma-2-2B / Llama-3.1-8B don't fit in 3 GB** — the Phase 2 Step 5 cross-model replication
  and Phase 4 cross-model baselines are deferred to a larger/cloud GPU.

## Run contract

Every experiment is a script, never a notebook. Each one records full provenance
before doing any work — the resolved config, git commit + dirty flag, seed, and
the versions of the numerically load-bearing libraries — into the run directory:

```bash
uv run python scripts/smoke_run.py --desc smoke-test --config configs/pythia-70m.yaml --seed 0
# -> runs/<date>-smoke-test/{config.yaml, meta.json, run.log, tables/result.json}
```

Confirm the package is importable and see tracked provenance:

```bash
uv run layer-dynamics      # or: python -m layer_dynamics.cli
```

## Status

**Phase 1, Step 0 — Environment.** Skeleton + run-logging backbone in place. See
[progress.md](progress.md) for the live state and the go/no-go gates.
