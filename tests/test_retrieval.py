"""Regression tests for retrieval. Run after every change to retrieval:

    pytest -q tests/test_retrieval.py

Two groups:
  logic   fast, no models needed: insurer routing, brand removal on every path, the glossary's
          rare-word rule, and neighbour expansion staying inside its section
  recall  needs Ollama and the Chroma index: recall must not fall below the levels already reached
          (skipped automatically if Ollama is not running)
Each test exists because of a real problem found while building; see the comment on each.
"""
import json
from pathlib import Path

import pytest

import app.retrieve as r
from app.glossary import expand

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "eval" / "questions.jsonl"
HELDOUT = ROOT / "eval" / "heldout_questions.jsonl"
BRANDS = ("policy expert", "uinsure", "covea", "covéa")


def load(path):
    return [json.loads(line) for line in path.open() if line.strip()]


# ------------------------------------------------------------------ logic
def test_route_finds_the_named_insurer():
    """Routing must pick the right insurer for every eval question that names one."""
    for q in load(MAIN):
        text, insurers = r.route(q["question"])
        if q["insurer"] == "All":
            assert len(insurers) >= 2, q["id"]
        else:
            assert insurers == [q["insurer"]], q["id"]


def test_route_removes_brand_names():
    """Brand names in the embedded text pull in welcome and complaints pages (problem 13)."""
    for q in load(MAIN):
        text, _ = r.route(q["question"])
        assert not any(b in text.lower() for b in BRANDS), q["id"]


def test_brand_removed_when_insurer_comes_from_context(monkeypatch):
    """Regression for problem 21: the policy-context path once skipped brand removal."""
    seen = {}
    monkeypatch.setattr(r, "embed_query", lambda text: seen.setdefault("text", text) and [0.0])
    monkeypatch.setattr(r, "_search", lambda *a, **k: [])
    r.retrieve("What is the limit on Covea's legal protection cover?", insurer="Covea",
               expand=False, glossary=False)
    assert "covea" not in seen["text"].lower()


@pytest.mark.skipif(not (ROOT / "data" / "chunks_400.jsonl").exists(), reason="run ingest.chunk first")
def test_glossary_leaves_policy_vocabulary_alone():
    """Expanding words the policies already use caused query drift (problem 17)."""
    assert expand("My heating failed, am I covered?") == "My heating failed, am I covered?"
    assert expand("Will they pay for a loose roof tile?") == "Will they pay for a loose roof tile?"


@pytest.mark.skipif(not (ROOT / "data" / "chunks_400.jsonl").exists(), reason="run ingest.chunk first")
def test_glossary_expands_real_vocabulary_gaps():
    assert "cooling-off period" in expand("Can I get my money back?")
    assert "all risks" in expand("Is accidental damage included?")


def test_neighbours_never_cross_document_or_section(monkeypatch):
    """Neighbour expansion adds the next chunk, but only from the same document and section."""
    meta = lambda src, sec, p: {"insurer": "Covea", "section": sec, "source": src, "headings": "",
                                "pdf_pages": str(p), "printed_pages": str(p - 1)}
    store = {"c0101": ("condition", meta("cov.pdf", "Section 2 Buildings", 35)),
             "c0201": ("other doc", meta("pe.pdf", "Section 1 Buildings", 20)),
             "c0301": ("other section", meta("cov.pdf", "Section 3 Liability", 39))}

    class FakeCollection:
        def get(self, ids, include):
            ids = [i for i in ids if i in store]
            return {"ids": ids, "documents": [store[i][0] for i in ids],
                    "metadatas": [store[i][1] for i in ids]}

    monkeypatch.setattr(r, "collection", lambda name=None: FakeCollection())
    hits = [r._to_hit("c0100", "cover", meta("cov.pdf", "Section 2 Buildings", 34), 0.8),
            r._to_hit("c0200", "x", meta("cov.pdf", "Section 2 Buildings", 30), 0.7),
            r._to_hit("c0300", "y", meta("cov.pdf", "Section 2 Buildings", 38), 0.6)]
    ids = [h["id"] for h in r.add_neighbours(hits, "x")]
    assert ids == ["c0100", "c0101", "c0200", "c0300"]


# ----------------------------------------------------------------- recall
def _ollama_running() -> bool:
    try:
        import ollama
        ollama.list()
        return (ROOT / "chroma").exists()
    except Exception:
        return False


needs_index = pytest.mark.skipif(not _ollama_running(), reason="Ollama or Chroma index not available")


def _recall(questions, k=5, context=False):
    from eval.retrieval_recall import expected_pages, found
    qs = [q for q in questions if q["behaviour"] != "refuse" and expected_pages(q)]
    ok = 0
    for q in qs:
        insurer = q["insurer"] if context and q["insurer"] != "All" else None
        hits = r.retrieve(q["question"], k=k, insurer=insurer)
        exp = expected_pages(q)
        ok += found(hits, exp) == set(exp)
    return ok, len(qs)


@needs_index
def test_main_set_recall_floor():
    """Recall reached 32/34 at k=5; any change that drops below that is a regression."""
    ok, n = _recall(load(MAIN))
    assert ok >= 32, f"main set recall@5 fell to {ok}/{n}"


@needs_index
def test_main_set_recall_floor_with_policy_context():
    """Guards problem 21: supplying the insurer must not lower main-set recall."""
    ok, n = _recall(load(MAIN), context=True)
    assert ok >= 32, f"main set recall@5 with policy context fell to {ok}/{n}"


@needs_index
def test_customer_questions_recall_floor_with_policy_context():
    """Independent customer questions reached 5/6 at k=5 with the insurer known (problem 20)."""
    ok, n = _recall(load(HELDOUT), context=True)
    assert ok >= 6, f"customer questions recall@5 fell to {ok}/{n}"