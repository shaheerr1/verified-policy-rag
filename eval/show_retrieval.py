"""Show exactly what retrieval returns for each question, marking chunks on an expected page.

Run:  python -m eval.show_retrieval                       (all held-out questions)
      python -m eval.show_retrieval u07 u08 u11           (only these ids)
      python -m eval.show_retrieval --main q05            (from the main eval set)
Uses the full pipeline with the insurer known from policy context, as in production.
"""
import json
import sys

from app.retrieve import retrieve
from eval.retrieval_recall import expected_pages
from ingest.parse import ROOT

args = sys.argv[1:]
path = ROOT / "eval" / ("questions.jsonl" if "--main" in args else "heldout_questions.jsonl")
ids = {a for a in args if not a.startswith("--")}

found_count = total = 0
for line in path.open():
    q = json.loads(line)
    if (ids and q["id"] not in ids) or q["behaviour"] == "refuse":
        continue
    exp = expected_pages(q)
    insurer = q["insurer"] if q["insurer"] != "All" else None
    hits = retrieve(q["question"], k=5, insurer=insurer)
    ok = any(h["insurer"] in exp and set(h["pdf_pages"]) & exp[h["insurer"]] for h in hits)
    found_count += ok
    total += 1

    print("=" * 100)
    print(f"{q['id']}  [{q['insurer']}]  {'FOUND' if ok else 'MISSED'}")
    print(f"Q: {q['question']}")
    print(f"Expected answer: {q.get('expected', '')}")
    print(f"Expected PDF pages: {dict((i, sorted(p)) for i, p in exp.items())}")
    for h in hits:
        mark = "*" if h["insurer"] in exp and set(h["pdf_pages"]) & exp[h["insurer"]] else " "
        kind = "  +next" if h.get("neighbour_of") else ""
        snippet = h["text"].replace("\n", " ")[:150]
        print(f"  {mark} {h['id']}  {h['insurer']:<13} p{h['printed_pages']} (PDF {h['pdf_pages']}){kind}")
        print(f"      {snippet}")

print("=" * 100)
print(f"{found_count}/{total} questions had a chunk from an expected page in the top 5 (+ neighbours)")