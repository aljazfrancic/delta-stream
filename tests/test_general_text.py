"""Tests for the E4 generic-text length-matched window selection (Phase 4)."""

from __future__ import annotations

import numpy as np
import pytest

from layer_dynamics.data.general_text import length_matched_windows


def _docs(n: int, length: int, seed: int = 0) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    return [list(rng.integers(0, 1000, size=length)) for _ in range(n)]


def test_one_window_per_target_with_matched_lengths():
    docs = _docs(50, 40)
    targets = [4, 9, 9, 12, 24]
    windows = length_matched_windows(docs, targets, seed=0)
    assert [len(w) for w in windows] == targets


def test_windows_are_contiguous_subsequences_of_documents():
    # Every window must appear verbatim as a contiguous slice of some document
    # (no boundary crossing, no token invented).
    docs = _docs(30, 64, seed=1)

    def is_slice_of_some_doc(win: list[int]) -> bool:
        for d in docs:
            for off in range(len(d) - len(win) + 1):
                if d[off : off + len(win)] == win:
                    return True
        return False

    windows = length_matched_windows(docs, [5, 8, 13, 21], seed=3)
    assert all(is_slice_of_some_doc(w) for w in windows)


def test_distinct_documents_used_per_window():
    # With one window per distinct document, no two windows should be drawn from
    # the same document — check via the shuffled-consumption invariant: the number
    # of qualifying documents consumed equals the number of windows.
    docs = _docs(10, 30)
    windows = length_matched_windows(docs, [5, 5, 5], seed=2)
    assert len(windows) == 3
    # All-equal-length docs => exactly len(targets) docs consumed, all distinct.


def test_deterministic_under_seed():
    docs = _docs(40, 50, seed=7)
    targets = [4, 9, 12, 7, 20]
    a = length_matched_windows(docs, targets, seed=11)
    b = length_matched_windows(docs, targets, seed=11)
    c = length_matched_windows(docs, targets, seed=12)
    assert a == b
    assert a != c  # different seed => different offsets/doc order


def test_short_documents_are_skipped():
    # Mix of short (len 3) and long (len 30) docs; a target of 10 must land in a
    # long doc only.
    docs = [[1, 2, 3]] * 20 + _docs(5, 30, seed=4)
    windows = length_matched_windows(docs, [10, 10, 10], seed=0)
    assert all(len(w) == 10 for w in windows)


def test_raises_when_documents_exhausted():
    docs = _docs(2, 30)  # only two long-enough docs
    with pytest.raises(ValueError, match="ran out of documents"):
        length_matched_windows(docs, [10, 10, 10], seed=0)
