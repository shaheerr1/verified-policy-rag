"""Step 1 of ingestion: parse the policy PDFs with Docling and save the results.
Slow (several minutes), so it runs once. Chunking happens later in chunk.py.
Run from the repo root:  python -m ingest.parse
Add --force to re-parse files that were already parsed.
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import TableItem, TextItem

ROOT = Path(__file__).resolve().parents[1]
PDFS = ROOT / "data" / "pdfs"
PARSED = ROOT / "data" / "parsed"

def insurer_for(name: str) -> str:
    n = name.lower()
    if "pebasic" in n or "policyexpert" in n:
        return "Policy Expert"
    if "uinsure" in n:
        return "Uinsure"
    if "executive" in n or "covea" in n:
        return "Covea"
    return Path(name).stem

def build_converter() -> DocumentConverter:
    opts = PdfPipelineOptions()
    opts.do_ocr = False                      # all three PDFs have a real text layer
    opts.do_table_structure = True           # tables are where the numbers are
    opts.table_structure_options.mode = TableFormerMode.ACCURATE
    opts.table_structure_options.do_cell_matching = True
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})


def page_texts(doc) -> dict:
    """Docling's text for each PDF page, with tables as markdown. Used by check_numbers.py."""
    pages = defaultdict(list)
    for item, _ in doc.iterate_items():
        if not getattr(item, "prov", None):
            continue
        page = item.prov[0].page_no
        if isinstance(item, TableItem):
            pages[page].append(item.export_to_markdown(doc=doc))
        elif isinstance(item, TextItem):
            pages[page].append(item.text)
    return {p: "\n".join(t) for p, t in sorted(pages.items())}


def main():
    force = "--force" in sys.argv
    PARSED.mkdir(parents=True, exist_ok=True)
    converter = build_converter()

    for pdf in sorted(PDFS.glob("*.pdf")):
        insurer = insurer_for(pdf.name)
        out_json = PARSED / f"{pdf.stem}.json"
        if out_json.exists() and not force:
            print(f"== {insurer}: already parsed, skipping (use --force to redo)")
            continue

        print(f"== {insurer}: parsing {pdf.name} ...", flush=True)
        start = time.time()
        doc = converter.convert(pdf).document

        doc.save_as_json(out_json)                                    # full structure, reloaded by chunk.py
        (PARSED / f"{pdf.stem}.md").write_text(doc.export_to_markdown())  # readable copy to eyeball
        (PARSED / f"{pdf.stem}.pages.json").write_text(
            json.dumps(page_texts(doc), ensure_ascii=False, indent=1))

        tables = sum(1 for item, _ in doc.iterate_items() if isinstance(item, TableItem))
        print(f"   {doc.num_pages()} pages, {tables} tables, {time.time() - start:.0f}s")

    print(f"\nDone. Outputs in {PARSED.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()