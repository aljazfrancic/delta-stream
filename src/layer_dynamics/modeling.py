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


def load_untrained_model(
    name: str, *, device: str = "auto", dtype: str = "float32", seed: int = 0
) -> Any:
    """Load a HookedTransformer with the pretrained *architecture* but **random weights**.

    The T4 untrained-model control (Phase 4 H1-validity). Same config as
    ``load_model(name)`` — identical ``n_layers`` / ``d_model`` / tokenizer — but the
    weights are freshly initialised (TransformerLens default init) from ``seed``
    instead of the trained checkpoint. A forward pass still produces a running-sum
    residual stream, so this is the control that separates H1's depth structure from
    residual-stream *plumbing*: any AR-on-Δ predictability an untrained net shows —
    especially after T1's persistence strip — is a property of the architecture, not
    of learning.

    Reproducible on CUDA (the primary device): TransformerLens weight init draws from
    the Python, NumPy *and* torch RNGs, so ``set_seed`` is called immediately before
    construction (seeding only torch is insufficient — verified). CPU init is not
    bit-reproducible in TransformerLens, but the scientific conclusion is robust to
    the particular random draw.
    """
    import torch
    from transformer_lens import HookedTransformer
    from transformer_lens.loading_from_pretrained import get_pretrained_model_config

    from layer_dynamics.seeding import set_seed

    dev = resolve_device(device)
    torch_dtype = getattr(torch, _DTYPES.get(dtype, "float32"))
    cfg = get_pretrained_model_config(name, device=dev, dtype=torch_dtype)
    cfg.init_weights = True  # the config defaults this off (pretrained loads weights);
    #                          without it HookedTransformer leaves params uninitialised
    #                          (zero on a fresh CUDA alloc) → a degenerate all-zero stream.
    set_seed(seed)  # TL init draws from random+numpy+torch; seed all three
    model = HookedTransformer(cfg)  # random weights from the current seed
    model.eval()
    return model
