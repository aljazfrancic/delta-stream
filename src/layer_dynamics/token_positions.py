"""Probe-position resolution for the token-position-invariance test (Phase 2 Step 3).

Phase 1/2 collected residual trajectories at a single token — the final prompt
token. Step 3 asks whether the AR-on-Δ structure is a property of the depth
dynamics or specific to that query position, by recollecting at three more
deterministic positions per example and refitting the kernel at each.

This module locates those positions in **content-token coordinates** (the token
index *excluding* the prepended BOS, so absolute index = content index + 1 — the
indexing the collection path in :mod:`layer_dynamics.activations` already uses).
The four positions, in fixed order:

    0 ``subj_first`` — first subject token
    1 ``subj_last``  — last subject token (== ``subj_first`` for single-token subjects)
    2 ``mid``        — a mid-prompt non-subject token (the median non-subject,
                       non-final content token; deterministic and reproducible)
    3 ``final``      — final prompt token (the Phase 1 position)

The subject character span is taken from the relation template's ``{}`` slot
(``prompt == template.format(subject)`` for CounterFact, verified at collection),
with a ``prompt.find(subject)`` fallback. The token span is then read off the
fast tokenizer's offset mapping, so it is robust to sub-word splitting.

Resolution is kept free of any model/tokenizer object (it takes the offset
mapping as input) so it is unit-testable with hand-built offsets.
"""

from __future__ import annotations

POSITION_NAMES = ("subj_first", "subj_last", "mid", "final")


def subject_char_span(prompt: str, subject: str, template: str | None) -> tuple[int, int] | None:
    """Character span ``[start, end)`` of ``subject`` within ``prompt``.

    Prefers the template's ``{}`` slot (exact for ``prompt == template.format(subject)``),
    falling back to the first verbatim occurrence of ``subject``. Returns ``None`` if
    the subject cannot be located.
    """
    if template and "{}" in template:
        start = len(template.split("{}", 1)[0])
        if prompt[start : start + len(subject)] == subject:
            return start, start + len(subject)
    idx = prompt.find(subject)
    if idx >= 0:
        return idx, idx + len(subject)
    return None


def token_span_from_offsets(
    offsets: list[tuple[int, int]], char_start: int, char_end: int
) -> tuple[int, int] | None:
    """First/last content-token indices overlapping the char span ``[char_start, char_end)``.

    ``offsets`` is the fast tokenizer's ``offset_mapping`` for the content tokens
    (``add_special_tokens=False``). A token ``[o0, o1)`` overlaps the subject when
    ``o1 > char_start and o0 < char_end``; zero-width offsets are ignored. Returns
    ``None`` when no token overlaps.
    """
    hits = [
        i
        for i, (o0, o1) in enumerate(offsets)
        if o1 > o0 and o1 > char_start and o0 < char_end
    ]
    if not hits:
        return None
    return hits[0], hits[-1]


def resolve_probe_positions(
    offsets: list[tuple[int, int]], subj_span: tuple[int, int]
) -> dict | None:
    """Resolve the four probe positions (content indices) from a subject token span.

    Returns a dict with content indices ``subj_first/subj_last/mid/final``, the
    ``multi_token_subject`` flag (subject spans >1 token), and a per-position
    ``valid`` mask. ``mid`` is the **median** non-subject, non-final content token —
    a deterministic, reproducible rule — and is ``None`` (with ``valid['mid']`` False)
    when no such token exists (the subject abuts the final token). Returns ``None``
    if the prompt has no content tokens.
    """
    n = len(offsets)
    if n == 0:
        return None
    sf, sl = subj_span
    final = n - 1
    subject_idx = set(range(sf, sl + 1))
    cand = [i for i in range(n) if i not in subject_idx and i != final]
    mid = cand[len(cand) // 2] if cand else None
    valid = {
        "subj_first": True,
        "subj_last": True,
        "mid": mid is not None,
        "final": True,
    }
    return {
        "subj_first": sf,
        "subj_last": sl,
        "mid": mid,
        "final": final,
        "multi_token_subject": sl > sf,
        "valid": valid,
    }
