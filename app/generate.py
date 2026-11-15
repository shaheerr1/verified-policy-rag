"""Generate a cited answer from retrieved policy clauses with a local model."""
import json
import sys

import ollama

from app.retrieve import retrieve
from app.privacy import mask
MODEL = "qwen2.5:3b"


def page_of(hit):
    pages = hit.get("printed_pages") or hit.get("pdf_pages") or ""
    if isinstance(pages, list):
        pages = ", ".join(map(str, pages))
    return str(pages)


def build_context(hits):
    blocks = []
    for h in sorted(hits, key=lambda h: h["id"]):
        header = f"[{h['id']}] {h.get('insurer', '')} | {h.get('section', '')} | page {page_of(h)}"
        blocks.append(header + "\n" + h["text"].strip())
    return "\n\n".join(blocks)


SCHEMA = {
    "type": "object",
    "properties": {
        "relevant_clauses": {"type": "array", "maxItems": 10, "items": {"type": "string"}},
        "sentences": {
            "type": "array",
            "minItems": 1,
            "maxItems": 5,
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "citations": {"type": "array", "maxItems": 4, "items": {"type": "string"}},
                },
                "required": ["text", "citations"],
            },
        },
        "not_in_policy": {"type": "boolean"},
    },
    "required": ["relevant_clauses", "sentences", "not_in_policy"],
}

SYSTEM = """You answer customer questions about their home insurance policy using ONLY the policy clauses provided.

Rules:
1. Every sentence must be supported by the clauses. Do not use outside knowledge.
2. For each sentence, list the clause IDs (like c0412) that support it in "citations". Only use IDs that appear in the clauses.
3. Copy amounts, limits, day counts and percentages exactly as written in the clause.
4. If the answer differs by policy tier or cover option and the customer's tier is not given, state the answer for each tier.
5. Include any condition or exclusion that changes the answer.
6. Ignore clauses that are not relevant to the question.
7. If no clause answers the question, write one sentence saying the policy documents provided do not answer this, with no citations, and set "not_in_policy" to true.
8. Answer the customer's actual question in 2 to 5 short sentences. The first sentence must answer it directly (yes, no, or it depends on...). Then add only the limits, conditions or exclusions that change the answer. Use plain words, but do not copy whole clauses. Do not put clause IDs inside the sentence text.
9. Work in order: first list the IDs of every clause relevant to the question in "relevant_clauses", then write the answer, then decide "not_in_policy".
10. Keep the policy's own terms for who and what is covered (for example "a member of your family", not "your son"; "contents", not "his laptop"). Each sentence makes one claim and is under 30 words. Do not mention section names or say where something is stated; citations do that."""

def generate(question, insurer=None, k=5):
    question = mask(question)
    hits = retrieve(question, k=k, insurer=insurer)
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Policy clauses:\n\n{build_context(hits)}\n\nCustomer question: {question}"},
        ],
        format=SCHEMA,
        # options={"temperature": 0, "num_ctx": 8192, "num_predict": 700},
        options={"temperature": 0, "seed": 42, "num_ctx": 8192, "num_predict": 700},
    )
    raw = response["message"]["content"]
    try:
        answer = json.loads(raw)
    except json.JSONDecodeError:
        print("UNPARSEABLE OUTPUT (first 800 chars):\n", raw[:800])
        raise
    return {"question": question, "insurer": insurer, "hits": hits, "answer": answer}


def show(result):
    by_id = {h["id"]: h for h in result["hits"]}
    a = result["answer"]
    print(f"\nQ: {result['question']}")
    if a["not_in_policy"]:
        print("NOT IN POLICY")
    for s in a["sentences"]:
        refs = ", ".join(
            f"{c} p{page_of(by_id[c])}" if c in by_id else f"{c} NOT IN CONTEXT"
            for c in s["citations"]
        ) or "no citation"
        print(f"- {s['text']}\n    [{refs}]")


if __name__ == "__main__":
    # python -m app.generate Uinsure "question..."
    insurer, question = sys.argv[1], " ".join(sys.argv[2:])
    show(generate(question, insurer=insurer))