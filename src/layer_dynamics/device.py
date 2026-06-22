"""Device and dtype selection for model runs.

Centralises the retarget decisions for the local box (Phase 1 Step 0): a GTX
1050 (Pascal, compute capability 6.1, 3 GB). Two consequences are baked in
here so individual scripts don't re-derive them:

- **fp32, not fp16.** Pascal has no usable half-precision tensor cores; fp16
  *compute* is slower than fp32, and Pythia-410m's fp32 weights (~1.6 GB) fit
  in 3 GB for inference. Activations are stored fp32 first per the plan and
  downcast downstream only if disk-bound.
- **Small batches.** A batch of 4 through Pythia-410m peaks at ~2 GB; callers
  default low and can raise it when running on a larger card.
"""

from __future__ import annotations

from typing import Any


def resolve_device(prefer: str = "auto") -> str:
    """Resolve ``"auto" | "cuda" | "cpu"`` to a concrete device string.

    ``"auto"`` and ``"cuda"`` fall back to CPU when no GPU is visible, so the
    same config runs on the dev box and on a CPU-only machine.
    """
    if prefer == "cpu":
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def collection_dtype() -> Any:
    """The dtype for activation collection: fp32 (see module docstring)."""
    import torch

    return torch.float32


def describe(device: str) -> str:
    """Human-readable one-liner for logging (name + free VRAM if CUDA)."""
    if device != "cuda":
        return device
    import torch

    name = torch.cuda.get_device_name(0)
    free, total = torch.cuda.mem_get_info()
    return f"cuda:{name} ({free // 2**20}/{total // 2**20} MiB free)"
