"""Step 3 of ingestion: chunk the parsed documents. Fast, so run it as often as needed.

Loads Docling's saved JSON (no re-parsing), follows the limits table decision from
check_numbers.py, and sizes chunks to fit the DeBERTa checker (512 token window).

Policy wordings use generic headings ("We don't cover:", "What is not covered?") that
Docling treats as headings of their own, which would cut exclusions off from the cover
they belong to. Those chunks are merged back into the preceding cover chunk, or, if that
would exceed the size limit, labelled with the cover's heading. Every chunk is also
labelled with its policy section (Buildings, Contents...), so identical cover names in
different sections can be told apart.

Run from the repo root:
  python -m ingest.chunk          # 400 tokens (default)
  python -m ingest.chunk 256      # smaller chunks, for the recall comparison
"""
import csv
import json
import re
import sys
from pathlib import Path

from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.types.doc import DoclingDocument
from transformers import AutoTokenizer

from ingest.parse import PARSED, PDFS, ROOT, insurer_for

NLI_MODEL = "cross-encoder/nli-deberta-v3-base"  # the checker; its tokenizer sets the size limit
NLI_LIMIT = 512
MIN_TOKENS = 10
LIMITS_CSV = ROOT / "data" / "policy_expert_limits.csv"
LIMITS_PAGES = {4, 5}
# printed page = PDF page - offset (checked by hand against each PDF's footer)
PAGE_OFFSET = {"Policy Expert": 0, "Uinsure": 2, "Covea": 1}
# section each PDF page belongs to: (first PDF page, section name), checked against page headers/footers
SECTIONS = {
    "Policy Expert": [(1, "General terms and definitions"), (19, "Section 1 Buildings"),
                      (27, "Section 2 Contents"), (37, "Cancellation and complaints")],
    "Uinsure": [(1, "General terms and definitions"), (10, "Section 1 Buildings"),
                (17, "Section 2 Contents"), (33, "Policy conditions"),
                (36, "Section 3 Family Legal Expenses"), (47, "Section 4 Home Emergency"),
                (59, "Complaints and privacy")],
    "Covea": [(1, "General information and definitions"), (20, "Section 1 Contents, Art and Jewellery"),
              (31, "Section 2 Buildings"), (39, "Section 3 Liability"), (46, "Section 4 Family Protection"),
              (47, "Section 4A Lifestyle"), (49, "Section 4B Home Emergency"),
              (55, "Section 4C Legal Protection"), (65, "Further information"),
              (67, "Section 4D Cyber"), (68, "General conditions")],
}

# headings that only make sense attached to the cover above them
GENERIC_HEADING = re.compile(
    r"^(we don't cover|we do not cover|we will not (cover|pay)|what is (not )?covered|"
    r"what you are (not )?covered for|not covered|exclusions?|conditions?|claims)\b",
    re.I,
)
TEMPLATE_JUNK = re.compile(r"Sub-section\s*-\s*Sub-section text\s*", re.I)  # left in Uinsure's PDF


def clean(s: str) -> str:
    return TEMPLATE_JUNK.sub("", s).replace("\u2019", "'").strip()


def is_generic(headings: list) -> bool:
    return bool(headings) and bool(GENERIC_HEADING.match(headings[0]))


def section_for(insurer: str, pdf_page: int) -> str:
    """The section a PDF page belongs to: the last section starting on or before it."""
    current = ""
    for start, name in SECTIONS.get(insurer, []):
        if pdf_page >= start:
            current = name
    return current


def printed(insurer: str, pdf_pages: list) -> list:
    """Convert PDF page numbers to the numbers printed on the pages."""
    off = PAGE_OFFSET.get(insurer, 0)
    return [p - off for p in pdf_pages if p - off > 0]



TIER_NAMES = ["Basic", "Bronze", "Silver", "Gold"]


def describe(value: str) -> str:
    """Turn one table cell into plain words."""
    v = value.strip()
    if v.lower() == "not covered":
        return "is not covered"
    if v.lower() == "covered":
        return "is covered"
    if v.lower().startswith("see your schedule"):
        return "has its limit shown in your schedule"
    if v.startswith("£"):
        return f"is covered up to {v}"
    return f"is {v}"


def row_sentence(row: dict) -> str:
    """Write a limits table row as plain sentences, grouping tiers that share a value.
    Customers ask in sentences; a row of cells barely resembles their question."""
    item = row["item"].split(" (")[0]
    qualifier = row["item"][len(item):].strip(" ()")
    subject = f"{item.lower()} ({qualifier} cover)" if qualifier else item.lower()
    groups = []
    for tier, key in zip(TIER_NAMES, ["basic", "bronze", "silver", "gold"]):
        if groups and groups[-1][1] == row[key]:
            groups[-1][0].append(tier)
        else:
            groups.append(([tier], row[key]))
    parts = []
    for tiers, value in groups:
        names = tiers[0] if len(tiers) == 1 else ", ".join(tiers[:-1]) + " and " + tiers[-1]
        parts.append(f"On Policy Expert {names}, {subject} {describe(value)}.")
    return " ".join(parts)


def csv_chunks(tokenizer) -> list:
    """One chunk per row of the hand-checked Policy Expert limits table."""
    out = []
    for row in csv.DictReader(LIMITS_CSV.open()):
        page = int(row["pdf_page"])
        text = row_sentence(row) + "\n" + row["chunk_text"]  # plain sentences + the exact row
        out.append({
            "insurer": "Policy Expert",
            "source": "policy_expert_limits.csv",
            "headings": ["Policy limits", row["section"]],
            "section": "Policy limits",
            "pdf_pages": [page],
            "printed_pages": printed("Policy Expert", [page]),
            "tokens": tokenizer.count_tokens(text),
            "text": text,
        })
    return out


def attach_generic(chunks: list, tokenizer, max_tokens: int) -> tuple:
    """Merge 'We don't cover:'-style chunks into the cover before them, or label them with it."""
    out, merged, labelled = [], 0, 0
    for c in chunks:
        prev = out[-1] if out else None
        if prev is None or not is_generic(c["headings"]):
            out.append(c)
            continue
        combined = prev["text"] + "\n" + c["text"]
        n = tokenizer.count_tokens(combined)
        if n <= max_tokens:
            prev["text"] = combined
            prev["tokens"] = n
            prev["pdf_pages"] = sorted(set(prev["pdf_pages"]) | set(c["pdf_pages"]))
            merged += 1
        else:
            parent = next((h for h in reversed(prev["headings"]) if not GENERIC_HEADING.match(h)), None)
            if parent:
                c["headings"] = [parent] + c["headings"]
                c["text"] = parent + "\n" + c["text"]
                c["tokens"] = tokenizer.count_tokens(c["text"])
            out.append(c)
            labelled += 1
    return out, merged, labelled


def chunk_document(pdf: Path, chunker, tokenizer, skip_pages: set, max_tokens: int) -> list:
    """Chunk one parsed document, keeping headings, section and page numbers on every chunk."""
    insurer = insurer_for(pdf.name)
    doc = DoclingDocument.load_from_json(PARSED / f"{pdf.stem}.json")
    raw = []
    for chunk in chunker.chunk(dl_doc=doc):
        pdf_pages = sorted({p.page_no for item in chunk.meta.doc_items for p in item.prov})
        if pdf_pages and set(pdf_pages) <= skip_pages:
            continue  # these pages are replaced by the hand-checked CSV
        text = clean(chunker.contextualize(chunk=chunk))  # heading path + chunk text
        raw.append({
            "insurer": insurer,
            "source": pdf.name,
            "headings": [clean(h) for h in (chunk.meta.headings or []) if clean(h)],
            "pdf_pages": pdf_pages,
            "tokens": tokenizer.count_tokens(text),
            "text": text,
        })

    chunks, merged, labelled = attach_generic(raw, tokenizer, max_tokens)
    out = []
    for c in chunks:
        if c["tokens"] < MIN_TOKENS:
            continue  # cover lines and stray labels: nothing a question could be answered from
        section = section_for(insurer, c["pdf_pages"][0]) if c["pdf_pages"] else ""
        if section and section.lower() not in c["text"][:120].lower():
            c["headings"] = [section] + c["headings"]
            c["text"] = section + "\n" + c["text"]
            c["tokens"] = tokenizer.count_tokens(c["text"])
        c["section"] = section
        c["printed_pages"] = printed(insurer, c["pdf_pages"])
        out.append(c)
    print(f"{insurer}: {len(out)} chunks ({merged} exclusion blocks merged into their cover, "
          f"{labelled} labelled with it)")
    return out


def main():
    max_tokens = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(NLI_MODEL), max_tokens=max_tokens)
    chunker = HybridChunker(tokenizer=tokenizer, merge_peers=True)

    decision = json.loads((PARSED / "limits_decision.json").read_text())
    use_csv = decision["use_csv_for_pe_pages_4_5"]

    chunks = []
    for pdf in sorted(PDFS.glob("*.pdf")):
        insurer = insurer_for(pdf.name)
        skip = LIMITS_PAGES if (insurer == "Policy Expert" and use_csv) else set()
        chunks += chunk_document(pdf, chunker, tokenizer, skip, max_tokens)
    if use_csv:
        rows = csv_chunks(tokenizer)
        chunks += rows
        print(f"Policy Expert limits: {len(rows)} rows from hand-checked CSV")

    out = ROOT / "data" / f"chunks_{max_tokens}.jsonl"
    with out.open("w") as f:
        for i, c in enumerate(chunks):
            f.write(json.dumps({"id": f"c{i:04d}", **c}, ensure_ascii=False) + "\n")

    sizes = [c["tokens"] for c in chunks]
    too_big = sum(s > NLI_LIMIT for s in sizes)
    generic_left = sum(is_generic([h for h in c["headings"] if h != c.get("section")]) for c in chunks)
    print(f"\n{len(chunks)} chunks -> {out.relative_to(ROOT)}")
    print(f"tokens: avg {sum(sizes) // len(sizes)}, max {max(sizes)}, over {NLI_LIMIT}: {too_big}")
    print(f"chunks still starting with a generic heading: {generic_left}")


if __name__ == "__main__":
    main()