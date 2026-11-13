"""Step 3 of ingestion: chunk the parsed documents. Fast, so run it as often as needed.

Loads Docling's saved JSON (no re-parsing), follows the limits table decision from
check_numbers.py, and sizes chunks to fit the DeBERTa checker (512 token window).

Run from the repo root:
  python -m ingest.chunk          # 400 tokens (default)
  python -m ingest.chunk 256      # smaller chunks, for the recall comparison
"""
import csv
import json
import sys
from pathlib import Path

from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.types.doc import DoclingDocument
from transformers import AutoTokenizer

from ingest.parse import PARSED, PDFS, ROOT, insurer_for

NLI_MODEL = "cross-encoder/nli-deberta-v3-base"  # the checker; its tokenizer sets the size limit
NLI_LIMIT = 512
LIMITS_CSV = ROOT / "data" / "policy_expert_limits.csv"
LIMITS_PAGES = {4, 5}
# printed page = PDF page - offset (checked by hand against each PDF's footer)
PAGE_OFFSET = {"Policy Expert": 0, "Uinsure": 2, "Covea": 1}


def printed(insurer: str, pdf_pages: list) -> list:
    """Convert PDF page numbers to the numbers printed on the pages."""
    off = PAGE_OFFSET.get(insurer, 0)
    return [p - off for p in pdf_pages if p - off > 0]


def csv_chunks(tokenizer) -> list:
    """One chunk per row of the hand-checked Policy Expert limits table."""
    out = []
    for row in csv.DictReader(LIMITS_CSV.open()):
        page = int(row["pdf_page"])
        text = row["chunk_text"]
        out.append({
            "insurer": "Policy Expert",
            "source": "policy_expert_limits.csv",
            "headings": ["Policy limits", row["section"]],
            "pdf_pages": [page],
            "printed_pages": printed("Policy Expert", [page]),
            "tokens": tokenizer.count_tokens(text),
            "text": text,
        })
    return out


def chunk_document(pdf: Path, chunker, tokenizer, skip_pages: set) -> list:
    """Chunk one parsed document, keeping headings and page numbers on every chunk."""
    insurer = insurer_for(pdf.name)
    doc = DoclingDocument.load_from_json(PARSED / f"{pdf.stem}.json")
    out = []
    for chunk in chunker.chunk(dl_doc=doc):
        pdf_pages = sorted({p.page_no for item in chunk.meta.doc_items for p in item.prov})
        if pdf_pages and set(pdf_pages) <= skip_pages:
            continue  # these pages are replaced by the hand-checked CSV
        text = chunker.contextualize(chunk=chunk)  # heading path + chunk text
        out.append({
            "insurer": insurer,
            "source": pdf.name,
            "headings": chunk.meta.headings or [],
            "pdf_pages": pdf_pages,
            "printed_pages": printed(insurer, pdf_pages),
            "tokens": tokenizer.count_tokens(text),
            "text": text,
        })
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
        doc_chunks = chunk_document(pdf, chunker, tokenizer, skip)
        print(f"{insurer}: {len(doc_chunks)} chunks")
        chunks += doc_chunks
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
    print(f"\n{len(chunks)} chunks -> {out.relative_to(ROOT)}")
    print(f"tokens: avg {sum(sizes) // len(sizes)}, max {max(sizes)}, over {NLI_LIMIT}: {too_big}")


if __name__ == "__main__":
    main()