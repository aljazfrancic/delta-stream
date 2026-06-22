"""Factual-recall dataset sources (Phase 1 Step 1).

Each source loader returns a pandas DataFrame of *candidate* triples in the
canonical s-r-o schema below, before any model-based filtering. The build
script unions the requested sources, deduplicates, and then filters by whether
the primary model actually predicts the object.

Canonical candidate columns:
    subject, relation, relation_template, object, prompt, relation_class

- ``prompt`` is already in s-r-o order ("Paris is the capital of") with the
  subject inserted; ``object`` is the target continuation *with its leading
  space* (e.g. " France") so its first token matches what the model predicts.
- ``relation`` is the Wikidata P-code; ``relation_class`` the broad category.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from layer_dynamics.data.relations import relation_class

CANONICAL_COLUMNS = [
    "subject",
    "relation",
    "relation_template",
    "object",
    "prompt",
    "relation_class",
]


def load_counterfact() -> pd.DataFrame:
    """Load CounterFact via ``NeelNanda/counterfact-tracing``.

    That release is already reformatted for the s-r-o convention used here —
    ``prompt`` has the subject inserted and ``target_true`` is the
    space-prefixed object — so no reverse-direction reformatting is needed.
    """
    from datasets import load_dataset

    ds = load_dataset("NeelNanda/counterfact-tracing", split="train")
    df = pd.DataFrame(
        {
            "subject": [s.strip() for s in ds["subject"]],
            "relation": list(ds["relation_id"]),
            "relation_template": list(ds["relation"]),
            # Keep the leading space CounterFact stores: it is the first token
            # the model must predict after the prompt.
            "object": list(ds["target_true"]),
            "prompt": list(ds["prompt"]),
        }
    )
    df["relation_class"] = df["relation"].map(relation_class)
    return df[CANONICAL_COLUMNS]


def _not_implemented(name: str) -> Callable[[], pd.DataFrame]:
    def loader() -> pd.DataFrame:
        raise NotImplementedError(
            f"source '{name}' is registered but not yet implemented. CounterFact "
            f"alone yields well over the 5,000-example target after filtering; "
            f"add '{name}' here when a cross-source union is needed (plan Step 1)."
        )

    return loader


# Source registry. Add new loaders here; the build script reads this by name.
SOURCES: dict[str, Callable[[], pd.DataFrame]] = {
    "counterfact": load_counterfact,
    "pararel": _not_implemented("pararel"),
    "trex": _not_implemented("trex"),
}


def load_sources(names: list[str]) -> pd.DataFrame:
    """Union the named sources into one deduplicated candidate frame.

    Dedup key is (subject, relation, prompt) so the same fact phrased
    identically is not double-counted across sources.
    """
    frames = [SOURCES[name]() for name in names]
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["subject", "relation", "prompt"]).reset_index(drop=True)
    return df
