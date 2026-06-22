"""Model loading via TransformerLens, shared across phases.

Centralises the from_pretrained call + dtype/device handling so Step 1
(filtering) and Step 2 (activation collection) load the model the same way.
"""

from __future__ import annotations

from typing import Any

from layer_dynamics.device import resolve_device


def load_model(name: str, *, device: str = "auto", dtype: str = "float32") -> Any:
    """Load a HookedTransformer in eval mode on the resolved device.

    fp32 by default (Pascal has no usable fp16 path — see device.py).
    """
    import torch
    from transformer_lens import HookedTransformer

    dev = resolve_device(device)
    model = HookedTransformer.from_pretrained(name, device=dev)
    if dtype == "float32":
        model = model.to(torch.float32)
    elif dtype in ("float16", "half"):
        model = model.to(torch.float16)
    elif dtype in ("bfloat16", "bf16"):
        model = model.to(torch.bfloat16)
    model.eval()
    return model
