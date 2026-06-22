"""Model-based filtering of candidate triples (Phase 1 Step 1).

Computes, for each prompt, the model's top-1 next-token id and its probability,
batched. The build script keeps a candidate when the top-1 token equals the
object's first token and the probability clears the threshold.

Correctness note on batching: prompts are right-padded and the prediction is
read at each sequence's *true* last token (index = number of content tokens,
since a BOS is prepended). Causal attention means the logit at that position
attends only to real tokens, so right-padding does not affect it.
"""

from __future__ import annotations

import numpy as np


def first_token_id(model, text: str) -> int:
    """First token id of ``text`` (keep its leading space; no BOS)."""
    ids = model.tokenizer(text, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError(f"{text!r} tokenises to nothing")
    return int(ids[0])


def _content_len(model, prompt: str) -> int:
    """Number of content tokens (excludes the prepended BOS)."""
    return len(model.tokenizer(prompt, add_special_tokens=False)["input_ids"])


def top1_next_token(
    model,
    prompts: list[str],
    *,
    batch_size: int = 32,
    progress: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(top1_ids, top1_probs)`` for the next token after each prompt."""
    import torch

    model.tokenizer.padding_side = "right"
    top1_ids = np.empty(len(prompts), dtype=np.int64)
    top1_probs = np.empty(len(prompts), dtype=np.float32)

    rng = range(0, len(prompts), batch_size)
    if progress:
        from tqdm import tqdm

        rng = tqdm(rng, desc="filtering", unit="batch")

    with torch.no_grad():
        for start in rng:
            batch = prompts[start : start + batch_size]
            # last content index per row: BOS at 0, content at 1..n -> index n.
            last_idx = [_content_len(model, p) for p in batch]
            tokens = model.to_tokens(batch, prepend_bos=True)  # right-padded
            logits = model(tokens, return_type="logits")
            rows = torch.arange(len(batch))
            last = torch.tensor(last_idx)
            final_logits = logits[rows, last]  # [B, vocab]
            probs = torch.softmax(final_logits.float(), dim=-1)
            p, idx = probs.max(dim=-1)
            top1_ids[start : start + len(batch)] = idx.cpu().numpy()
            top1_probs[start : start + len(batch)] = p.cpu().numpy()

    return top1_ids, top1_probs
