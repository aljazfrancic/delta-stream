#!/usr/bin/env bash
# Reproduce the local toolchain (Phase 1 Step 0, retargeted to the dev box:
# GTX 1050 / 3 GB / Pascal sm_61). Idempotent — safe to re-run.
#
# Targets a CUDA torch build held on the 2.5 line (last series shipping Pascal
# kernels) via the cu124 wheel index configured in pyproject.toml. For a
# CPU-only or different-CUDA machine, edit [tool.uv.index] there and re-run.
set -euo pipefail
cd "$(dirname "$0")/.."

# 1. uv (user-local, no root). The box has no curl, so fetch with wget.
if ! command -v uv >/dev/null 2>&1; then
  echo ">> installing uv"
  wget -qO /tmp/uv-install.sh https://astral.sh/uv/install.sh
  sh /tmp/uv-install.sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version

# 2. Pinned interpreter + full environment (creates .venv).
echo ">> installing Python 3.12 and syncing dependencies"
uv python install 3.12
uv sync --extra dev --python 3.12

# 3. Verify the GPU is usable and torch sees it.
echo ">> verifying GPU"
uv run python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0),
          "| capability:", torch.cuda.get_device_capability(0))
PY

echo ">> done. Activate with: source .venv/bin/activate  (or prefix commands with 'uv run')"
