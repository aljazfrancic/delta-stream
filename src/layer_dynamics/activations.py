"""Residual-stream collection (Phase 1 Step 2).

Collects the residual state ``v_ℓ`` at every depth ``ℓ ∈ {0, …, L}`` for a list
of prompts, at the final prompt token. ``v_0`` is ``blocks.0.hook_resid_pre``
(the embedding output, before any block writes) and ``v_1 … v_L`` are
``blocks.{ℓ}.hook_resid_post``. Δ_ℓ = v_{ℓ+1} − v_ℓ is computed downstream, not
here (Step 2 stores only ``v`` to leave the door open for analyses on ``v``).
"""

from __future__ import annotations

import numpy as np


def resid_hook_names(n_layers: int) -> list[str]:
    """Ordered hook names for v_0 .. v_L (length L+1)."""
    return ["blocks.0.hook_resid_pre"] + [
        f"blocks.{ell}.hook_resid_post" for ell in range(n_layers)
    ]


def _content_len(model, prompt: str) -> int:
    """Content-token count (excludes the prepended BOS). The final prompt token
    sits at index = content_len (BOS at 0, content at 1..n)."""
    return len(model.tokenizer(prompt, add_special_tokens=False)["input_ids"])


def collect_trajectories(
    model,
    prompts: list[str],
    *,
    batch_size: int = 8,
    progress: bool = True,
) -> np.ndarray:
    """Collect ``[N, L+1, D]`` residual trajectories at the final prompt token.

    Right-padding is safe: each row is read at its true final token, which under
    causal attention depends only on real tokens (same argument as filtering).
    """
    import torch

    model.tokenizer.padding_side = "right"
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    names = resid_hook_names(n_layers)
    nameset = set(names)

    out = np.empty((len(prompts), n_layers + 1, d_model), dtype=np.float32)

    rng = range(0, len(prompts), batch_size)
    if progress:
        from tqdm import tqdm

        rng = tqdm(rng, desc="collecting", unit="batch")

    with torch.no_grad():
        for start in rng:
            batch = prompts[start : start + batch_size]
            last_idx = torch.tensor([_content_len(model, p) for p in batch])
            tokens = model.to_tokens(batch, prepend_bos=True)
            _, cache = model.run_with_cache(
                tokens, names_filter=lambda n: n in nameset, return_type=None
            )
            rows = torch.arange(len(batch))
            for li, name in enumerate(names):
                out[start : start + len(batch), li, :] = (
                    cache[name][rows, last_idx].float().cpu().numpy()
                )

    return out


def collect_trajectories_from_ids(
    model,
    id_lists: list[list[int]],
    *,
    batch_size: int = 8,
    progress: bool = True,
) -> np.ndarray:
    """Like :func:`collect_trajectories` but from pre-tokenized content-id lists.

    Each entry in ``id_lists`` is the content token ids (no BOS); a BOS is prepended
    exactly as ``model.to_tokens(..., prepend_bos=True)`` would, and the trajectory is
    read at that row's final content token (index ``len(ids)``, with BOS at 0). Used by
    the context-perturbation control (Step 5 Null 2), where the wrapped and perturbed
    variants must be built at the token level so the subject-relation span sits at
    identical absolute positions in both — a string round-trip cannot guarantee that.
    """
    import torch

    device = next(model.parameters()).device
    bos_id = int(model.to_tokens("", prepend_bos=True)[0, 0])
    pad_id = model.tokenizer.pad_token_id
    pad_id = int(pad_id) if pad_id is not None else bos_id
    n_layers, d_model = model.cfg.n_layers, model.cfg.d_model
    names = resid_hook_names(n_layers)
    nameset = set(names)

    out = np.empty((len(id_lists), n_layers + 1, d_model), dtype=np.float32)

    rng = range(0, len(id_lists), batch_size)
    if progress:
        from tqdm import tqdm

        rng = tqdm(rng, desc="collecting(ids)", unit="batch")

    with torch.no_grad():
        for start in rng:
            batch = id_lists[start : start + batch_size]
            lens = [len(ids) for ids in batch]
            width = max(lens) + 1  # + BOS
            toks = torch.full((len(batch), width), pad_id, dtype=torch.long)
            for i, ids in enumerate(batch):
                toks[i, 0] = bos_id
                toks[i, 1 : 1 + len(ids)] = torch.tensor(ids, dtype=torch.long)
            toks = toks.to(device)
            _, cache = model.run_with_cache(
                toks, names_filter=lambda n: n in nameset, return_type=None
            )
            last_idx = torch.tensor(lens)  # final content token (BOS at 0)
            rows = torch.arange(len(batch))
            for li, name in enumerate(names):
                out[start : start + len(batch), li, :] = (
                    cache[name][rows, last_idx].float().cpu().numpy()
                )

    return out


def collect_trajectories_at_positions(
    model,
    prompts: list[str],
    abs_positions: np.ndarray,
    *,
    batch_size: int = 8,
    progress: bool = True,
) -> np.ndarray:
    """Collect ``[N, P, L+1, D]`` residual trajectories at ``P`` token positions each.

    ``abs_positions`` is ``[N, P]`` of **absolute** token indices (BOS at 0, content
    at 1..n) — the indices the token-position probe (Phase 2 Step 3) resolves. One
    forward pass per batch reads all ``P`` positions, so adding positions does not
    multiply compute. Right-padding is safe: every read index is ≤ that row's final
    real token, which under causal attention depends only on real tokens.

    Out-of-range / placeholder indices (e.g. an invalid ``mid``) should be passed as a
    valid in-range index (caller clamps); the caller's ``valid`` mask decides which
    reads to trust downstream.
    """
    import torch

    model.tokenizer.padding_side = "right"
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    names = resid_hook_names(n_layers)
    nameset = set(names)
    p = abs_positions.shape[1]

    out = np.empty((len(prompts), p, n_layers + 1, d_model), dtype=np.float32)

    rng = range(0, len(prompts), batch_size)
    if progress:
        from tqdm import tqdm

        rng = tqdm(rng, desc="collecting(pos)", unit="batch")

    with torch.no_grad():
        for start in rng:
            batch = prompts[start : start + batch_size]
            pos = torch.as_tensor(
                abs_positions[start : start + len(batch)], dtype=torch.long
            )
            tokens = model.to_tokens(batch, prepend_bos=True)
            _, cache = model.run_with_cache(
                tokens, names_filter=lambda n: n in nameset, return_type=None
            )
            rows = torch.arange(len(batch))[:, None]  # [b, 1] broadcast over P
            for li, name in enumerate(names):
                gathered = cache[name][rows, pos]  # [b, P, D]
                out[start : start + len(batch), :, li, :] = gathered.float().cpu().numpy()

    return out


def recovers_top1(model, v_last: np.ndarray) -> int:
    """ln_final + unembed of a single final residual ``v_L`` -> top-1 token id.

    The Step 2 sanity check: this should recover the model's actual top-1 token.
    Skipping ln_final (unembedding v_L directly) is the usual reason a correct
    extraction looks wrong.
    """
    import torch

    v = torch.tensor(v_last, device=next(model.parameters()).device).reshape(1, 1, -1)
    logits = model.unembed(model.ln_final(v))[0, -1]
    return int(logits.argmax())
