"""Tests for the dataset-assembly module (Phase 1 Step 1)."""

from __future__ import annotations

import pandas as pd

from layer_dynamics.data import relations, sources


def test_relation_class_values_are_known():
    # Every mapped class must be one of the declared classes.
    assert set(relations.RELATION_CLASS.values()) <= set(relations.RELATION_CLASSES)


def test_relation_class_covers_34_counterfact_relations():
    assert len(relations.RELATION_CLASS) == 34
    assert relations.relation_class("P103") == "language"
    assert relations.relation_class("P30") == "geography"
    assert relations.relation_class("P176") == "organization"
    assert relations.relation_class("P-unknown") == "other"


def test_load_sources_unions_and_dedups(monkeypatch):
    def fake_a():
        return pd.DataFrame(
            {
                "subject": ["Paris", "Tokyo"],
                "relation": ["P36", "P36"],
                "relation_template": ["{} capital", "{} capital"],
                "object": [" France", " Japan"],
                "prompt": ["Paris is the capital of", "Tokyo is the capital of"],
                "relation_class": ["geography", "geography"],
            }
        )

    def fake_b():
        # One genuine duplicate of fake_a's first row, one new row.
        return pd.DataFrame(
            {
                "subject": ["Paris", "Berlin"],
                "relation": ["P36", "P36"],
                "relation_template": ["{} capital", "{} capital"],
                "object": [" France", " Germany"],
                "prompt": ["Paris is the capital of", "Berlin is the capital of"],
                "relation_class": ["geography", "geography"],
            }
        )

    monkeypatch.setitem(sources.SOURCES, "a", fake_a)
    monkeypatch.setitem(sources.SOURCES, "b", fake_b)

    df = sources.load_sources(["a", "b"])
    assert len(df) == 3  # 4 rows, one (subject, relation, prompt) duplicate removed
    assert list(df.columns) == sources.CANONICAL_COLUMNS


def test_unimplemented_source_raises():
    import pytest

    with pytest.raises(NotImplementedError):
        sources.SOURCES["pararel"]()
