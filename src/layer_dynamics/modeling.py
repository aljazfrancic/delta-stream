"""Model loading via TransformerLens, shared across phases.

Centralises the from_pretrained call + dtype/device handling so Step 1
(filtering) and Step 2 (activation collection) load the model the same way.
"""

from __future__ import annotations

from typing import Any

from layer_dynamics.device import resolve_device

_DTYPES = {
    "float32": "float32", "float16": "float16", "half": "float16",
    "bfloat16": "bfloat16", "bf16": "bfloat16",
}


def load_model(name: str, *, device: str = "auto", dtype: str = "float32") -> Any:
    """Load a HookedTransformer in eval mode on the resolved device.

    fp32 by default (Pascal has no usable fp16 path — see device.py). The dtype is
    passed **into** ``from_pretrained`` so large fp16 models load directly in half
    precision instead of materialising fp32 weights on the GPU first (gpt2-large's
    fp32 weights are ~3.1 GB and OOM the 3 GB card during a load-then-cast).
    """
    import torch
    from transformer_lens import HookedTransformer

    dev = resolve_device(device)
    torch_dtype = getattr(torch, _DTYPES.get(dtype, "float32"))
    model = HookedTransformer.from_pretrained(name, device=dev, dtype=torch_dtype)
    model.eval()
    return model
