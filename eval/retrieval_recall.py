"""Retrieval recall: for each answerable eval question, is a chunk from the right insurer and
the right PDF page among the top-k results? Compares every chunk-size collection.

Run:  python -m eval.retrieval_recall
"""
import json
import math
import re
from collections import defaultdict

from app.retrieve import retrieve
from ingest.parse import ROOT

QUESTIONS = ROOT / "eval" / "questions.jsonl"

# (label, collection, routing on/off): each fix is measured against the one before it
CONFIGS = [
    ("baseline: vector search", "chunks_400", False),
    ("+ insurer routing", "chunks_400", True),
]

KS = [5, 10]
SHORT = {"PE": "Policy Expert", "Policy Expert": "Policy Expert", "Uinsure": "Uinsure", "Covea": "Covea"}


def expected_pages(q: dict) -> dict:
    """{insurer: {pdf pages}} from the question's pdf_page field.
    Single insurer: '4, 20'.  Cross-document: 'PE 10; Uinsure 6; Covea 19'."""
    field = str(q.get("pdf_page", ""))
    if q["insurer"] != "All":
        return {q["insurer"]: {int(n) for n in re.findall(r"\d+", field)}}
    out = {}
    for part in field.split(";"):
        m = re.match(r"\s*(PE|Policy Expert|Uinsure|Covea)\s+(.+)", part)
        if m:
            out[SHORT[m.group(1)]] = {int(n) for n in re.findall(r"\d+", m.group(2))}
    return out


def found(hits: list, expected: dict) -> set:
    """Which expected insurers have at least one retrieved chunk on an expected page."""
    return {h["insurer"] for h in hits
            if h["insurer"] in expected and set(h["pdf_pages"]) & expected[h["insurer"]]}


def wilson(hits: int, n: int, z: float = 1.96) -> tuple:
    """95% Wilson interval for a proportion: honest error bars on a small sample."""
    if n == 0:
        return (0.0, 0.0)
    p = hits / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def main():
    questions = [json.loads(l) for l in QUESTIONS.open()]
    questions = [q for q in questions if q["behaviour"] != "refuse" and expected_pages(q)]
    print(f"{len(questions)} answerable questions (refusal questions have no correct page)\n")

    last_misses = []
    for label, coll, routing in CONFIGS:
        hits_by_q = {q["id"]: retrieve(q["question"], k=max(KS), collection_name=coll, routing=routing)
                     for q in questions}
        for k in KS:
            ok, misses = 0, []
            for q in questions:
                exp = expected_pages(q)
                top = hits_by_q[q["id"]][:k]
                if found(top, exp) == set(exp):   # cross-document questions need every insurer found
                    ok += 1
                else:
                    misses.append((q, exp, [(h["insurer"], h["pdf_pages"]) for h in top[:5]]))
            lo, hi = wilson(ok, len(questions))
            print(f"{label:<26} recall@{k:<2} {ok}/{len(questions)} = {ok / len(questions):.0%}"
                  f"   (95% CI {lo:.0%} to {hi:.0%})")
            if k == 5:
                last_misses = misses
        print()

    print(f"--- misses at k=5 for: {CONFIGS[-1][0]}")
    for q, exp, top in last_misses:
        print(f"  {q['id']}  {q['question'][:70]}")
        print(f"        expected {dict((i, sorted(p)) for i, p in exp.items())}")
        print(f"        got      {top}")
        

if __name__ == "__main__":
    main()