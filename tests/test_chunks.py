"""Regression tests for the chunk files. Run after every change to parsing or chunking:

    pytest -q tests/test_chunks.py

Three groups:
  structure      every chunk is well formed and within size limits
  answerability  each eval fact survives parsing and chunking, on the right insurer and PDF page
  pairing        text that must stay together (cover + its exclusions, label + its value) shares a chunk
Every fact below was checked against the PDF text before being written here, so a failure
means the chunks lost or split something, not that the test is wrong.
"""
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FILES = sorted((ROOT / "data").glob("chunks_*.jsonl"))
PAGE_COUNTS = {"Policy Expert": 42, "Uinsure": 64, "Covea": 76}
NLI_LIMIT, MIN_TOKENS = 512, 10
GENERIC = re.compile(r"^(we don't cover|we do not cover|what is (not )?covered|"
                     r"what you are (not )?covered for|not covered|exclusions?)\b", re.I)


def norm(s: str) -> str:
    """Lowercase, straight apostrophes, no whitespace: so line breaks never cause a false failure."""
    return re.sub(r"\s+", "", s.lower().replace("\u2019", "'"))


@pytest.fixture(params=FILES, ids=lambda p: p.stem, scope="module")
def chunks(request):
    return [json.loads(line) for line in request.param.open()]


def test_chunk_files_exist():
    assert FILES, "no data/chunks_*.jsonl found: run python -m ingest.chunk first"


# ---------------------------------------------------------------- structure
def test_required_fields(chunks):
    for c in chunks:
        for field in ("id", "insurer", "section", "source", "pdf_pages", "printed_pages", "tokens", "text"):
            assert c.get(field) not in (None, ""), f"{c.get('id')} missing {field}"


def test_unique_ids(chunks):
    ids = [c["id"] for c in chunks]
    assert len(ids) == len(set(ids))


def test_token_limits(chunks):
    too_big = [c["id"] for c in chunks if c["tokens"] > NLI_LIMIT]
    too_small = [c["id"] for c in chunks if c["tokens"] < MIN_TOKENS]
    assert not too_big, f"over the DeBERTa window: {too_big}"
    assert not too_small, f"below the minimum: {too_small}"


def test_pages_in_range(chunks):
    for c in chunks:
        assert all(1 <= p <= PAGE_COUNTS[c["insurer"]] for p in c["pdf_pages"]), c["id"]


def test_no_template_junk(chunks):
    assert not [c["id"] for c in chunks if "sub-section text" in c["text"].lower()]


def test_no_orphan_exclusions(chunks):
    """No chunk may start with 'We don't cover' without saying which cover it belongs to."""
    orphans = []
    for c in chunks:
        heads = [h for h in c["headings"] if h != c["section"]]
        if heads and GENERIC.match(heads[0]):
            orphans.append(c["id"])
    assert not orphans, f"exclusions without their cover: {orphans}"


# ------------------------------------------------------------ answerability
# (question id, insurer, PDF pages where the fact is printed, text that must survive)
FACTS = [
    ("q02", "Policy Expert", [20], "£1,000"),
    ("q03", "Policy Expert", [10], "30 days"),
    ("q04", "Policy Expert", [9], "55mph"),
    ("q05", "Policy Expert", [15], "seven days"),
    ("q06", "Policy Expert", [10, 27], "£2,000"),
    ("q07", "Policy Expert", [42], "loose roof tiles"),
    ("q08", "Uinsure", [6], "60 days"),
    ("q09", "Uinsure", [31], "£2,500"),
    ("q10", "Uinsure", [32], "£1000"),
    ("q11", "Uinsure", [52], "15 years"),
    ("q12", "Uinsure", [55], "48 hours"),
    ("q13", "Uinsure", [47], "48 hours"),
    ("q14", "Uinsure", [42, 44], "51%"),
    ("q15", "Covea", [28, 37], "£250"),
    ("q16", "Covea", [30], "£10,000"),
    ("q17", "Covea", [50, 53], "£1,000"),
    ("q19", "Covea", [57], "£100,000"),
    ("q20", "Covea", [32, 68, 73], "£75,000"),
    ("q23", "Policy Expert", [30], "accidental damage"),
    ("q24", "Uinsure", [16, 30], "accidental damage"),
    ("q25", "Covea", [28, 37], "£15,000"),
    ("q26", "Policy Expert", [27], "replacement title deeds"),
    ("q27", "Uinsure", [50], "£1,500"),
    ("q31", "Covea", [71], "30 days"),
    ("q34", "Covea", [67], "cyber"),
    # hand-checked limits table (Policy Expert pages 4-5)
    ("q01", "Policy Expert", [4], "Basic: £600,000"),
    ("q21", "Policy Expert", [4], "deducted from the rebuild cost"),
    ("q36", "Policy Expert", [5], "Theft from vehicle | Basic: not covered"),
    ("q37", "Policy Expert", [5], "Money | Basic: not covered"),
    ("q38", "Policy Expert", [5], "Gold: 45 days"),
]


@pytest.mark.parametrize("qid,insurer,pages,snippet", FACTS, ids=[f[0] for f in FACTS])
def test_fact_survives(chunks, qid, insurer, pages, snippet):
    on_page = [c for c in chunks if c["insurer"] == insurer and set(c["pdf_pages"]) & set(pages)]
    hits = [c["id"] for c in on_page if norm(snippet) in norm(c["text"])]
    assert hits, (f"{qid}: '{snippet}' not found in {insurer} chunks on PDF pages {pages} "
                  f"({len(on_page)} chunks on those pages: {[c['id'] for c in on_page][:8]})")


# ------------------------------------------------------------------ pairing
# (name, insurer, terms that must all appear in ONE chunk)
PAIRS = [
    ("PE buildings flood keeps its exception", "Policy Expert",
     ["Section 1 Buildings", "Flood", "unless your home has been damaged at the same time"]),
    ("PE contents flood keeps its exclusion", "Policy Expert",
     ["Section 2 Contents", "Flood", "contents in the open"]),
    ("Covea counselling keeps its number", "Covea", ["Counselling Service", "0330 134 8165"]),
    ("Uinsure subsidence keeps its exclusions", "Uinsure", ["Subsidence", "compaction of infill"]),
    ("Uinsure pedal cycles keeps its limit", "Uinsure", ["pedal cycles", "60 days anywhere else in the world"]),
    ("Uinsure boiler keeps its age limit", "Uinsure", ["boiler", "15 years"]),
]


@pytest.mark.parametrize("name,insurer,terms", PAIRS, ids=[p[0] for p in PAIRS])
def test_kept_together(chunks, name, insurer, terms):
    hits = [c["id"] for c in chunks if c["insurer"] == insurer and all(norm(t) in norm(c["text"]) for t in terms)]
    assert hits, f"{name}: no single chunk contains all of {terms}"


def test_flood_sections_distinct(chunks):
    """The buildings and contents Flood covers must be distinguishable by section."""
    sections = {c["section"] for c in chunks
                if c["insurer"] == "Policy Expert" and "2.flood" in norm(c["text"][:120])}
    assert {"Section 1 Buildings", "Section 2 Contents"} <= sections, sections