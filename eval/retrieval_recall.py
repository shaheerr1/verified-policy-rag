"""Retrieval recall: for each answerable eval question, is a chunk from the right insurer and
the right PDF page among the top-k results? Compares retrieval configurations side by side.

Run:  python -m eval.retrieval_recall
"""
import json
import math
import re

from app.retrieve import retrieve
from ingest.parse import ROOT

QUESTIONS = ROOT / "eval" / "questions.jsonl"
DEV = ROOT / "eval" / "glossary_dev_questions.jsonl"  # written alongside the glossary: development only
HELDOUT = ROOT / "eval" / "heldout_questions.jsonl"   # written independently, never used to tune anything


# (label, collection, routing, neighbour expansion, glossary, policy context)
# policy context = the system knows which insurer the customer is with (login, policy number, UI),
# as it would in production. Without it, the insurer must be guessed from the question text.
CONFIGS = [
    ("baseline: vector search", "chunks_400", False, False, False, False),
    ("+ insurer routing", "chunks_400", True, False, False, False),
    ("+ neighbour expansion", "chunks_400", True, True, False, False),
    ("+ glossary", "chunks_400", True, True, True, False),
    ("+ policy context", "chunks_400", True, True, True, True),
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


def evaluate(questions: list, configs: list, title: str) -> None:
    print(f"== {title}: {len(questions)} answerable questions\n")
    last_misses = []
    for label, coll, routing, expand, glossary, context in configs:
        for k in KS:
            ok, misses, sizes = 0, [], []
            for q in questions:
                exp = expected_pages(q)
                insurer = q["insurer"] if context and q["insurer"] != "All" else None
                hits = retrieve(q["question"], k=k, insurer=insurer, collection_name=coll,
                                routing=routing, expand=expand, glossary=glossary)
                sizes.append(len(hits))
                if found(hits, exp) == set(exp):   # cross-document questions need every insurer found
                    ok += 1
                else:
                    misses.append((q, exp, [(h["insurer"], h["pdf_pages"]) for h in hits[:6]]))
            lo, hi = wilson(ok, len(questions))
            print(f"{label:<26} k={k:<2} recall {ok}/{len(questions)} = {ok / len(questions):.0%}"
                  f"   (95% CI {lo:.0%} to {hi:.0%})   avg chunks returned {sum(sizes) / len(sizes):.1f}")
            if k == 5:
                last_misses = misses
        print()
    print(f"--- misses at k=5 for: {configs[-1][0]}")
    for q, exp, top in last_misses:
        print(f"  {q['id']}  {q['question'][:70]}")
        print(f"        expected {dict((i, sorted(p)) for i, p in exp.items())}")
        print(f"        got      {top}")
    print()


def load(path) -> list:
    rows = [json.loads(l) for l in path.open() if l.strip()]
    return [q for q in rows if q["behaviour"] != "refuse" and expected_pages(q)]


def main():
    evaluate(load(QUESTIONS), CONFIGS, "Main eval set")
    # glossary on vs off only: development set (leaky, written with the glossary) and the
    # independent held-out set, which is the real test of whether the glossary generalises
    evaluate(load(DEV), CONFIGS[-2:], "Glossary development set (written with the glossary)")
    evaluate(load(HELDOUT), CONFIGS[-2:], "Independent held-out set")


if __name__ == "__main__":
    main()