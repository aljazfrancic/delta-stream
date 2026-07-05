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


def stream_write_hook_names(n_layers: int) -> dict[str, list[str]]:
    """Per-layer residual-write hooks for the R8 attn-vs-MLP Δ split.

    ``hook_attn_out`` / ``hook_mlp_out`` are the tensors the block *adds* to the
    residual stream, so ``Δ_ℓ = attn_out_ℓ + mlp_out_ℓ`` exactly (sequential and
    parallel-attn/MLP architectures alike — TransformerLens sums these terms
    verbatim into ``resid_post``)."""
    return {
        "attn": [f"blocks.{ell}.hook_attn_out" for ell in range(n_layers)],
        "mlp": [f"blocks.{ell}.hook_mlp_out" for ell in range(n_layers)],
    }


def build_pseudo_stream(v0: np.ndarray, writes: np.ndarray) -> np.ndarray:
    """Cumulative pseudo-stream ``w`` whose diffs are exactly one write stream.

    ``w_0 = v_0`` (the embedding) and ``w_{ℓ+1} = w_ℓ + write_ℓ``, so downstream
    ``deltas(w)`` recovers ``writes`` and every Δ-based diagnostic (AR, Granger,
    stationarity, k*) runs unchanged on a single stream's per-layer residual
    writes. Only the diffs are meaningful — ``w_ℓ`` for ℓ ≥ 1 is *not* a real
    model state (the true stream also accumulates the other sublayer's writes).
    Accumulation in float64, returned as float32 (the store dtype).
    """
    n, n_layers, d = writes.shape
    assert v0.shape == (n, d)
    w = np.empty((n, n_layers + 1, d), dtype=np.float64)
    w[:, 0] = v0
    np.cumsum(writes.astype(np.float64), axis=1, out=w[:, 1:])
    w[:, 1:] += v0[:, None, :]
    return w.astype(np.float32)


def stream_additivity_stats(
    attn: np.ndarray, mlp: np.ndarray, dlt: np.ndarray
) -> dict:
    """How exactly ``attn + mlp`` reproduces ``Δ`` (fp32 rounding is the only slack).

    Returns the max absolute error, the relative Frobenius error, and per-layer
    share diagnostics for the memo: mean ‖attn_ℓ‖ / ‖mlp_ℓ‖ / ‖Δ_ℓ‖ and the
    attention energy share ‖attn‖² / (‖attn‖² + ‖mlp‖²).
    """
    err = attn + mlp - dlt
    max_abs = float(np.abs(err).max())
    rel_fro = float(np.linalg.norm(err) / np.linalg.norm(dlt))
    a2 = (attn.astype(np.float64) ** 2).sum(axis=2)  # [N, L]
    m2 = (mlp.astype(np.float64) ** 2).sum(axis=2)
    d2 = (dlt.astype(np.float64) ** 2).sum(axis=2)
    return {
        "max_abs_err": max_abs,
        "rel_frobenius_err": rel_fro,
        "mean_norm_attn": np.sqrt(a2).mean(axis=0).tolist(),
        "mean_norm_mlp": np.sqrt(m2).mean(axis=0).tolist(),
        "mean_norm_delta": np.sqrt(d2).mean(axis=0).tolist(),
        "attn_energy_share": (a2.sum(axis=0) / (a2.sum(axis=0) + m2.sum(axis=0))).tolist(),
    }


def collect_stream_writes(
    model,
    prompts: list[str],
    *,
    batch_size: int = 8,
    progress: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect per-layer attn/MLP residual writes + the residual trajectory.

    One forward pass per batch reads, at the final prompt token, the attention
    write ``attn_out_ℓ``, the MLP write ``mlp_out_ℓ`` (each ``[N, L, D]``) and the
    residual states ``v_0 … v_L`` (``[N, L+1, D]``, identical to
    :func:`collect_trajectories`) — so the R8 additivity check
    ``attn + mlp = Δ`` runs against the same forward pass, not a stored file.
    """
    import torch

    model.tokenizer.padding_side = "right"
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    resid_names = resid_hook_names(n_layers)
    stream_names = stream_write_hook_names(n_layers)
    attn_names, mlp_names = stream_names["attn"], stream_names["mlp"]
    nameset = set(resid_names) | set(attn_names) | set(mlp_names)

    attn = np.empty((len(prompts), n_layers, d_model), dtype=np.float32)
    mlp = np.empty((len(prompts), n_layers, d_model), dtype=np.float32)
    resid = np.empty((len(prompts), n_layers + 1, d_model), dtype=np.float32)

    rng = range(0, len(prompts), batch_size)
    if progress:
        from tqdm import tqdm

        rng = tqdm(rng, desc="collecting(streams)", unit="batch")

    with torch.no_grad():
        for start in rng:
            batch = prompts[start : start + batch_size]
            last_idx = torch.tensor([_content_len(model, p) for p in batch])
            tokens = model.to_tokens(batch, prepend_bos=True)
            _, cache = model.run_with_cache(
                tokens, names_filter=lambda n: n in nameset, return_type=None
            )
            rows = torch.arange(len(batch))
            sl = slice(start, start + len(batch))
            for li, name in enumerate(resid_names):
                resid[sl, li, :] = cache[name][rows, last_idx].float().cpu().numpy()
            for li, (a_name, m_name) in enumerate(zip(attn_names, mlp_names, strict=True)):
                attn[sl, li, :] = cache[a_name][rows, last_idx].float().cpu().numpy()
                mlp[sl, li, :] = cache[m_name][rows, last_idx].float().cpu().numpy()

    return attn, mlp, resid


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
