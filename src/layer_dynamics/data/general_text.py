"""Generic-text windows for the E4 generality control (Phase 4).

Phase 1 established H1 (above-null cross-layer Δ structure) on *factual-recall*
prompts read at the query position. E4 asks whether that structure is specific to
factual recall or general to transformer depth, by re-running the identical
AR-on-Δ + Granger + scrambled-null analysis on **generic text** (Pile documents)
read at an arbitrary final token.

To keep the comparison about *factual recall vs generic text* and not about
*context length*, the generic windows are length-matched to the factual set: the
caller passes the multiset of factual content-token lengths and gets back one
window per length, each drawn from a **distinct** document (no document-boundary
crossing) at a random in-document offset. The windows are returned as content-id
lists, fed to ``activations.collect_trajectories_from_ids`` exactly as the
factual prompts are (BOS prepended, residual read at the final content token).
"""

from __future__ import annotations

import numpy as np


def length_matched_windows(
    doc_token_lists: list[list[int]],
    target_lengths: list[int],
    *,
    seed: int,
) -> list[list[int]]:
    """Pick one length-matched generic window per target length.

    For each length ``L`` in ``target_lengths`` (matched, in order), consume the
    next document in a shuffled order that has at least ``L`` tokens and return a
    contiguous ``L``-token window from a random offset within it. Each window
    comes from a distinct document, so windows never cross a document boundary and
    diversity matches the "one window per fact" factual set.

    Raises ``ValueError`` if the documents run out before every target length is
    filled (the caller should supply more documents).
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(doc_token_lists))
    windows: list[list[int]] = []
    di = 0
    for length in target_lengths:
        while di < len(order) and len(doc_token_lists[order[di]]) < length:
            di += 1
        if di >= len(order):
            raise ValueError(
                f"ran out of documents long enough for length {length} "
                f"({len(windows)}/{len(target_lengths)} windows filled); "
                f"supply more / longer documents"
            )
        doc = doc_token_lists[order[di]]
        di += 1
        offset = int(rng.integers(0, len(doc) - length + 1))
        windows.append([int(t) for t in doc[offset : offset + length]])
    return windows
