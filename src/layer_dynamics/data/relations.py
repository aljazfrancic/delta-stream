"""Relation taxonomy for the CounterFact relations (Phase 1 Step 1).

Each of the 34 CounterFact relation ids (Wikidata P-codes) is mapped to a broad
``relation_class``. Classes are grouped by the *type of the predicted object*
— the thing the factual-recall circuit must produce — because that is what the
cross-class generalisation question in Phase 2 Step 4 is really about ("does the
kernel that predicts a *place* also predict a *language*?").

``relation_class`` is the stratification unit (Phase 1 Step 1 item 4: keep
classes with >=500 surviving examples) and the Phase 2 Step 4 split unit (>=3
classes with >=1000 examples). The finer P-code lives in the ``relation``
column for per-relation breakdowns.
"""

from __future__ import annotations

# P-code -> relation_class. Covers all 34 CounterFact relations exactly.
RELATION_CLASS: dict[str, str] = {
    # geography: object is a place (country, city, continent, region)
    "P30": "geography",   # continent
    "P17": "geography",   # located in country
    "P131": "geography",  # located in administrative entity
    "P276": "geography",  # location
    "P159": "geography",  # headquarters location
    "P19": "geography",   # place of birth
    "P20": "geography",   # place of death
    "P937": "geography",  # work location
    "P740": "geography",  # location of formation
    "P36": "geography",   # capital
    "P190": "geography",  # twinned administrative body
    "P27": "geography",   # country of citizenship
    "P495": "geography",  # country of origin
    "P138": "geography",  # named after (object is usually a place)
    # language: object is a language
    "P1412": "language",  # languages spoken/written
    "P103": "language",   # native language
    "P37": "language",    # official language
    "P364": "language",   # original language of work
    "P407": "language",   # language of work
    # occupation: object is a role / field / instrument / sport / genre
    "P106": "occupation",  # occupation
    "P413": "occupation",  # position played on team
    "P641": "occupation",  # sport
    "P39": "occupation",   # position held
    "P101": "occupation",  # field of work
    "P1303": "occupation", # instrument
    "P136": "occupation",  # genre
    # organization: object is a company / org / label / network
    "P176": "organization",  # manufacturer
    "P178": "organization",  # developer
    "P127": "organization",  # owned by
    "P108": "organization",  # employer
    "P463": "organization",  # member of
    "P264": "organization",  # record label
    "P449": "organization",  # original broadcaster
    # religion
    "P140": "religion",  # religion
}

RELATION_CLASSES: tuple[str, ...] = (
    "geography",
    "language",
    "occupation",
    "organization",
    "religion",
)


def relation_class(relation_id: str) -> str:
    """Map a P-code to its ``relation_class`` (``"other"`` if unknown)."""
    return RELATION_CLASS.get(relation_id, "other")
