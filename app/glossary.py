"""Customer words to policy words.

Customers describe cover in everyday language; policy wordings use industry terms, and
embeddings do not always connect the two ("accidental damage" vs "all risks"). Before a
question is embedded, matching policy terms are appended to it.

The list is general UK home insurance vocabulary, not tuned to individual test questions.
It is measured on a held-out question set that was not used to write it.
"""
import re

import json
from functools import lru_cache

from ingest.parse import ROOT

MAX_POLICY_USES = 5  # a customer word found in more chunks than this is already policy vocabulary


@lru_cache(maxsize=1)
def _policy_texts() -> tuple:
    path = ROOT / "data" / "chunks_400.jsonl"
    return tuple(json.loads(line)["text"].lower() for line in path.open())


def policy_uses(word: str) -> int:
    """How many chunks of the policies contain this exact word or phrase."""
    pattern = re.compile(rf"\b{re.escape(word.lower())}\b")
    return sum(1 for t in _policy_texts() if pattern.search(t))

# (pattern for the customer's words, policy terms to add)
GLOSSARY = [
    (r"\baccidental(ly)? damag", "accidental damage, all risks"),
    (r"\b(empty|vacant|nobody (will be )?(at home|living)|travelling|six weeks away)\b", "unoccupied"),
    (r"\b(furniture|belongings|possessions|my stuff|my things)\b", "contents, personal possessions"),
    (r"\b(house|structure|walls|roof)\b", "buildings"),
    (r"\b(car|van)\b", "vehicle"),
    (r"\b(bike|bicycle)s?\b", "pedal cycle"),
    (r"\b(burst pipe|leak(ing)?|water damage)\b", "escape of water"),
    (r"\b(break-?in|burglar(y|ies)?|stolen|steal(s|ing)?|nicked|robbed)\b", "theft"),
    (r"\b(boiler|heating)\b", "home emergency, central heating"),
    (r"\b(lawyer|solicitor|legal costs?|sue|court)\b", "legal expenses, legal protection"),
    (r"\b(changed? my mind|cancel(l?ing)?|money back|refund)\b", "cooling-off period, cancellation"),
    (r"\b(ring|necklace|jewel+e?ry|watch(es)?)\b", "jewellery, valuables"),
    (r"\b(pay (anything )?myself|deductible|out of my own pocket)\b", "excess"),
    (r"\bcash\b", "money"),
    (r"\b(abroad|overseas|on holiday|outside the uk)\b", "anywhere in the world, territorial limits"),
    (r"\b(cracks?|sinking)\b", "subsidence, heave"),
]
_COMPILED = [(re.compile(p, re.I), terms) for p, terms in GLOSSARY]


def expand(question: str) -> str:
    """Append policy terms for everyday words the policies themselves do not use.
    Words the policies already use (heating, roof, house) are left alone: expanding them
    only pulls the search towards general pages (query drift)."""
    terms = []
    for pattern, add in _COMPILED:
        m = pattern.search(question)
        if m and policy_uses(m.group(0)) <= MAX_POLICY_USES:
            terms += [t.strip() for t in add.split(",") if t.strip().lower() not in question.lower()]
    terms = list(dict.fromkeys(terms))  # de-duplicate, keep order
    return f"{question} Policy terms: {', '.join(terms)}." if terms else question