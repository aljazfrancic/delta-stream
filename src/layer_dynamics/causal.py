"""Causal existence-proof machinery (Phase 4 Step 3).

Two halves, kept separate so the SSM-side maths unit-tests without a GPU or a real model:

1. :func:`feature_delta_contribution` — the **estimated** per-layer contribution
   ``δ_ℓ`` of one Branch B feature ``(source layer ℓ*, input coordinate m)``, obtained by
   running the trained SSM forward twice (natural vs. that one coordinate zeroed) and
   differencing: ``δ_ℓ = Δ̂_ℓ^natural − Δ̂_ℓ^ablated``. By construction ``δ_ℓ = 0`` for
   ``ℓ < ℓ*`` (the input only enters at its home layer and afterwards), nonzero at the home
   layer via the feedthrough ``D_{ℓ*}`` and at later layers via the state chain
   ``B_{ℓ*} → z → C_ℓ``. The sign convention is **with-feature minus without-feature**
   ("what the feature added"), identical for the ablation (item 4) and activation (item 6)
   splices — only the splice *sign* differs.

2. :func:`run_prompts_with_resid_edit` — splice ``δ`` into the **real** model. At each block
   ``ℓ``'s ``hook_resid_post`` (which holds ``v_{ℓ+1}``) the per-layer ``sign·δ_ℓ`` is added at
   the query token position. Because the residual stream is cumulative, a per-layer add of
   ``−δ_ℓ`` at every ``ℓ ≥ ℓ*`` removes exactly ``Σ_{j=ℓ*}^{ℓ} δ_j`` from the stream after
   block ``ℓ`` — i.e. "subtract δ_ℓ from Δ_ℓ at every layer". The model recomputes all
   downstream attention/MLP from the edited stream (that recomputation *is* the causal
   effect we measure); we only remove the decomposition's *estimate* of the feature, leaving
   the reconstruction error in place (per the plan's "estimated contribution, not zero-replace").

``sign = -1`` ablates (removes the feature); ``sign = +1`` injects it (item 6's activation
test, where ``δ`` is built from a silent example with the coordinate set to the feature's mean
activation). The query position is each prompt's final content token (BOS at 0), matching the
Phase-1 collection convention.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from layer_dynamics.ssm import LowRankSSM


def feature_delta_contribution(
    model: LowRankSSM,
    u: Tensor,
    feature: int,
    source_layer: int,
    *,
    batch_size: int = 256,
) -> Tensor:
    """Per-example, per-layer contribution ``δ[N, L, D]`` of feature ``(source_layer, feature)``.

    Runs the SSM on ``u`` and on ``u`` with coordinate ``[:, source_layer, feature]`` zeroed,
    and differences the two reconstructions. Minibatched over examples so the ``[N, L, M]``
    input is never doubled. Returns ``δ`` on the model device (caller moves/splices it).
    """
    device = next(model.parameters()).device
    n = u.shape[0]
    out = torch.empty(n, model.dims.n_layers, model.dims.d_model, device=device)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            ub = u[start : start + batch_size].to(device)
            natural = model(ub)
            abl = ub.clone()
            abl[:, source_layer, feature] = 0.0
            out[start : start + batch_size] = natural - model(abl)
    if was_training:
        model.train()
    return out


def group_delta_contribution(
    model: LowRankSSM, u: Tensor, k: int, *, batch_size: int = 256
) -> Tensor:
    """δ[N, L, D] from zeroing each example's **top-k** |u| coordinates (distributed-code probe).

    Where :func:`feature_delta_contribution` removes one fixed feature, this removes each
    example's own k strongest-firing inputs (across all layers/coordinates) — the k features
    the decomposition says most explain *that* example's updates. Sweeping k traces how the
    behavioural effect grows as more of the distributed code is ablated together.
    """
    device = next(model.parameters()).device
    n = u.shape[0]
    out = torch.empty(n, model.dims.n_layers, model.dims.d_model, device=device)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            ub = u[start : start + batch_size].to(device)
            flat = ub.reshape(ub.shape[0], -1)
            idx = flat.abs().topk(min(k, flat.shape[1]), dim=1).indices
            uz = flat.scatter(1, idx, 0.0).reshape(ub.shape)
            out[start : start + batch_size] = model(ub) - model(uz)
    if was_training:
        model.train()
    return out


def group_random_delta_contribution(
    model: LowRankSSM,
    u: Tensor,
    k: int,
    *,
    seed: int = 0,
    match: str = "count",
    batch_size: int = 256,
) -> tuple[Tensor, Tensor, Tensor]:
    """δ[N, L, D] from a **random** group ablation — control for :func:`group_delta_contribution`.

    Where :func:`group_delta_contribution` removes each example's **top-k** |u| coordinates (its
    strongest co-firing features), this removes a *random* subset of the **same example's active
    (non-zero) coordinates** — the matched-random-group control. Two matching modes:

    * ``match="count"`` — remove ``min(k, n_active)`` randomly-chosen active coordinates: the same
      *number* of real features as top-k, randomly selected (shares the top-k's ``k`` axis).
    * ``match="mass"`` — remove randomly-chosen active coordinates until their summed |u| first
      reaches the top-k coordinates' summed |u|: the same total contribution *mass* removed,
      randomly selected (kills the "you just removed more residual-stream mass" confound).

    If the top-k ablation drops the answer while the matched-random one does not, the behavioural
    effect is carried specifically by the high-magnitude co-firing features, not by the count or
    sheer mass of residual-stream content removed — i.e. the distributed code is *structured*, not
    "any ~k directions span the answer". Inactive (zero) coordinates are never selected (ablating
    one is a no-op). Selection is deterministic in ``seed`` and independent of ``batch_size``.
    Returns ``(δ, n_ablated[N], mass_ablated[N])`` so the caller can report the matched count/mass.
    """
    if match not in ("count", "mass"):
        raise ValueError(f"match must be 'count' or 'mass', got {match!r}")
    device = next(model.parameters()).device
    n = u.shape[0]
    flat = u.reshape(n, -1).cpu()  # selection runs on CPU (batch-size- and device-independent)
    absu = flat.abs()
    p = absu.shape[1]
    kk = min(int(k), p)
    active = absu > 0
    # random order over coordinates with inactive ones pushed to the end (rand∈[0,1) > -1)
    g = torch.Generator().manual_seed(int(seed))
    keys = torch.where(active, torch.rand(n, p, generator=g), torch.full((n, p), -1.0))
    order = keys.argsort(dim=1, descending=True)  # active coords first, in random order
    active_ord = active.gather(1, order)
    if match == "count":
        rank = torch.arange(p).expand(n, p)
        take_ord = (rank < kk) & active_ord
    else:  # "mass": prefix whose cumulative |u| first reaches the top-k mass
        target = absu.topk(kk, dim=1).values.sum(dim=1, keepdim=True)
        absu_ord = absu.gather(1, order)
        cum_before = absu_ord.cumsum(dim=1) - absu_ord  # mass strictly before each position
        take_ord = (cum_before < target) & active_ord
    sel_mask = torch.zeros_like(take_ord).scatter(1, order, take_ord)  # back to coord positions
    n_ablated = sel_mask.sum(dim=1)
    mass_ablated = (absu * sel_mask).sum(dim=1)

    out = torch.empty(n, model.dims.n_layers, model.dims.d_model, device=device)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            sl = slice(start, start + batch_size)
            ub = u[sl].to(device)
            fb = ub.reshape(ub.shape[0], -1)
            mb = sel_mask[sl].to(device)
            uz = fb.masked_fill(mb, 0.0).reshape(ub.shape)
            out[sl] = model(ub) - model(uz)
    if was_training:
        model.train()
    return out, n_ablated, mass_ablated


def _content_len(model, prompt: str) -> int:
    """Content-token count (excludes the prepended BOS); query position = this index."""
    return len(model.tokenizer(prompt, add_special_tokens=False)["input_ids"])


def run_prompts_with_resid_edit(
    model,
    prompts: list[str],
    *,
    edit: Tensor | None = None,
    sign: float = -1.0,
    batch_size: int = 4,
) -> Tensor:
    """Final-query-token logits ``[N, vocab]``, optionally with a per-layer residual edit.

    ``edit[N, L, D]`` (or ``None`` for a clean run) is added as ``sign·edit[:, ℓ]`` at each
    block ``ℓ``'s ``hook_resid_post`` at the query token. Right-padding is safe — every read
    and edit is at the row's true final content token, which under causal attention depends
    only on real tokens. No-grad; logits returned on CPU (float32).
    """
    model.tokenizer.padding_side = "right"
    n_layers = model.cfg.n_layers
    device = next(model.parameters()).device
    edit_dev = None if edit is None else edit.to(device)

    out: Tensor | None = None  # sized from the first batch's actual logits width
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            last_idx = torch.tensor([_content_len(model, p) for p in batch], device=device)
            tokens = model.to_tokens(batch, prepend_bos=True)
            rows = torch.arange(len(batch), device=device)

            hooks = []
            if edit_dev is not None:
                eb = edit_dev[start : start + len(batch)]  # [b, L, D]
                hooks = [
                    (f"blocks.{ell}.hook_resid_post",
                     _resid_edit_hook(ell, rows, last_idx, eb, sign))
                    for ell in range(n_layers)
                ]

            logits = model.run_with_hooks(tokens, fwd_hooks=hooks, return_type="logits")
            final = logits[rows, last_idx].float().cpu()
            if out is None:
                out = torch.empty(len(prompts), final.shape[-1], dtype=torch.float32)
            out[start : start + len(batch)] = final
    return out


def _resid_edit_hook(ell: int, rows: Tensor, last_idx: Tensor, eb: Tensor, sign: float):
    """Build a ``hook_resid_post`` hook that adds ``sign·eb[:, ell]`` at the query token.

    All batch-local tensors are bound as arguments (not captured from the loop) so the
    closure is unambiguous — each layer's hook edits exactly its own ``eb[:, ell]`` slice.
    """

    def hook(resid, hook):  # noqa: ARG001  (TransformerLens passes the HookPoint)
        resid[rows, last_idx] = resid[rows, last_idx] + sign * eb[:, ell]
        return resid

    return hook


def answer_probs(logits: Tensor, answer_ids: np.ndarray) -> np.ndarray:
    """Softmax prob of each row's ``answer_ids`` token from final-token ``logits[N, vocab]``."""
    probs = torch.softmax(logits, dim=-1)
    return probs[torch.arange(len(answer_ids)), torch.as_tensor(answer_ids)].numpy()


def max_token_logprob(logits: Tensor, token_ids: list[int]) -> np.ndarray:
    """Per-row max log-probability over ``token_ids`` from final-token ``logits[N, vocab]``.

    The Step-3 item-6 activation metric: log-prob of the semantic token *set* (e.g. capital
    names), maxed over the set, so a near-miss that doesn't cross top-1 still registers.
    """
    logp = torch.log_softmax(logits, dim=-1)
    return logp[:, torch.as_tensor(token_ids)].max(dim=-1).values.numpy()


def single_token_ids(model, words: list[str]) -> tuple[list[int], list[str]]:
    """Token ids for ``words`` that are **single** tokens with a leading space; rest dropped.

    Per the item-6 tokenisation pre-registration: restrict the semantic set to single-token,
    leading-space realisations (the natural-language "The capital is …" position). Returns the
    kept ids and the kept word strings (so the caller can log/abort if too few survive).
    """
    ids, kept = [], []
    for w in words:
        toks = model.tokenizer(" " + w.strip(), add_special_tokens=False)["input_ids"]
        if len(toks) == 1:
            ids.append(int(toks[0]))
            kept.append(w.strip())
    return ids, kept


def top_and_control_examples(
    activations: np.ndarray, *, n: int = 100
) -> tuple[np.ndarray, np.ndarray, str]:
    """Select the ``n`` strongest-firing examples and ``n`` matched control examples.

    ``activations`` is the candidate feature's per-example value (signed). Top examples are
    the ``n`` largest by magnitude. Controls follow the Step-3 control-set pre-registration:
    sample from exactly-zero-activation examples; if fewer than ``n`` are available, fall back
    to the ``n`` smallest-magnitude *non-zero* examples (below the 5th percentile of non-zero
    magnitudes). Returns ``(top_idx, control_idx, rule)`` where ``rule`` records which branch
    was taken (logged in the paper, per the pre-registration).
    """
    mag = np.abs(activations)
    order = np.argsort(mag)[::-1]
    top_idx = order[:n]
    zero_idx = np.flatnonzero(mag == 0)
    if len(zero_idx) >= n:
        control_idx = zero_idx[:n]
        rule = "exact-zero"
    else:
        nz = mag[mag > 0]
        thresh = np.percentile(nz, 5) if len(nz) else 0.0
        small = np.flatnonzero((mag > 0) & (mag <= thresh))
        control_idx = small[np.argsort(mag[small])][:n] if len(small) else order[-n:]
        rule = "5th-percentile-fallback"
    return top_idx, control_idx, rule


def magnitude_matched_features(
    u: np.ndarray,
    candidate_layer: int,
    candidate_feature: int,
    top_idx: np.ndarray,
    *,
    n_rand: int = 10,
    feat_layers: np.ndarray | None = None,
    tol: float = 0.2,
    rng: np.random.Generator | None = None,
) -> tuple[list[tuple[int, int]], float]:
    """Random ``(layer, feature)`` controls matched on mean |activation| over ``top_idx``.

    Per Step-3 item 5: each random feature's mean activation magnitude on the *candidate's*
    top-100 examples must match the candidate's own mean magnitude there (±``tol``). Searches
    every ``(layer, feature)`` slot except the candidate; widens ``tol`` 0.2→0.5 only if too
    few qualify (recorded), and returns however many were found (caller flags N/A if <``n_rand``).
    Returns ``(features, tol_used)``.
    """
    rng = rng or np.random.default_rng(0)
    target = float(np.abs(u[top_idx, candidate_layer, candidate_feature]).mean())
    mean_mag = np.abs(u[top_idx]).mean(axis=0)  # [L, M] mean |activation| over top examples
    mean_mag[candidate_layer, candidate_feature] = np.nan  # exclude the candidate itself
    if feat_layers is not None:  # only real (alive) slots: home layer must equal this layer
        layer_grid = np.arange(u.shape[1])[:, None]
        mean_mag = np.where(feat_layers[None, :] == layer_grid, mean_mag, np.nan)

    for tol_used in (tol, 0.5):
        lo, hi = target * (1 - tol_used), target * (1 + tol_used)
        ell, m = np.where((mean_mag >= lo) & (mean_mag <= hi))
        if len(ell) >= n_rand:
            pick = rng.choice(len(ell), size=n_rand, replace=False)
            return [(int(ell[i]), int(m[i])) for i in pick], tol_used
    return [(int(a), int(b)) for a, b in zip(ell, m, strict=True)], 0.5
