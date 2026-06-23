"""Tests for the Phase 2 Step 3 probe-position resolver."""

from __future__ import annotations

from layer_dynamics.token_positions import (
    POSITION_NAMES,
    resolve_probe_positions,
    subject_char_span,
    token_span_from_offsets,
)


def test_subject_char_span_from_template():
    # "Apple A5 was created by" — subject at the {} slot (char 0..8).
    span = subject_char_span("Apple A5 was created by", "Apple A5", "{} was created by")
    assert span == (0, 8)


def test_subject_char_span_template_midsentence():
    prompt = "The capital of France is"
    span = subject_char_span(prompt, "France", "The capital of {} is")
    assert prompt[span[0] : span[1]] == "France"


def test_subject_char_span_fallback_and_missing():
    # No template -> verbatim find; absent subject -> None.
    assert subject_char_span("foo bar baz", "bar", None) == (4, 7)
    assert subject_char_span("foo bar baz", "qux", None) is None


def test_token_span_from_offsets_overlap():
    # Offsets for ["Apple"," A","5"," was"," created"," by"]; subject chars [0,8).
    offsets = [(0, 5), (5, 7), (7, 8), (8, 12), (12, 20), (20, 23)]
    assert token_span_from_offsets(offsets, 0, 8) == (0, 2)        # multi-token subject
    assert token_span_from_offsets(offsets, 8, 12) == (3, 3)       # single token
    assert token_span_from_offsets(offsets, 100, 200) is None      # no overlap


def test_resolve_probe_positions_multitoken():
    offsets = [(0, 5), (5, 7), (7, 8), (8, 12), (12, 20), (20, 23)]  # 6 content tokens
    pos = resolve_probe_positions(offsets, (0, 2))
    assert pos["subj_first"] == 0 and pos["subj_last"] == 2
    assert pos["final"] == 5
    assert pos["multi_token_subject"] is True
    # Non-subject, non-final candidates are {3, 4}; median index -> cand[1] == 4.
    assert pos["mid"] == 4
    assert all(pos["valid"][n] for n in POSITION_NAMES)


def test_resolve_probe_positions_singletoken():
    offsets = [(0, 6), (6, 9), (9, 12), (12, 15)]  # 4 tokens, subject = token 0
    pos = resolve_probe_positions(offsets, (0, 0))
    assert pos["subj_first"] == pos["subj_last"] == 0
    assert pos["multi_token_subject"] is False
    assert pos["final"] == 3
    # candidates {1, 2} -> median cand[1] == 2.
    assert pos["mid"] == 2


def test_resolve_probe_positions_no_mid():
    # Subject is everything but the final token -> no mid candidate.
    offsets = [(0, 3), (3, 6), (6, 9)]
    pos = resolve_probe_positions(offsets, (0, 1))
    assert pos["mid"] is None
    assert pos["valid"]["mid"] is False
    assert pos["valid"]["final"] is True


def test_resolve_probe_positions_empty():
    assert resolve_probe_positions([], (0, 0)) is None
