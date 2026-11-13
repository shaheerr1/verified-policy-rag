"""Step 2 of ingestion: check Docling's parse before anything is chunked.
  1. Figures check: every £ amount, phone number, % and day count in the PDF's own
     text layer must appear the same number of times in Docling's output, per page.
  2. Limits table check: Docling's Policy Expert tier table vs the hand-checked CSV.
Run after parse.py:  python -m ingest.check_numbers
Writes data/parsed/number_check.md and data/parsed/limits_decision.json
"""
import csv
import json
import re
from collections import Counter

import pdfplumber
from docling_core.types.doc import DoclingDocument, TableItem

from ingest.parse import PARSED, PDFS, ROOT, insurer_for

LIMITS_CSV = ROOT / "data" / "policy_expert_limits.csv"
LIMITS_PAGES = {4, 5}
TIERS = ["basic", "bronze", "silver", "gold"]


FIGURE = re.compile(
    r"£\s?\d[\d,]*(?:\.\d+)?"                 # money: £1,500  £250
    r"|\b0\d{3}\s?\d{3}\s?\d{4}\b"             # UK phone: 0330 134 8165
    r"|\b\d+(?:\.\d+)?\s?%"                    # percentages: 51%
    r"|\b\d+\s+(?:consecutive\s+)?days?\b",    # day counts: 60 consecutive days
    re.I,
)


def figures(text: str) -> Counter:
    return Counter(re.sub(r"\s+", "", m.group(0)).lower() for m in FIGURE.finditer(text or ""))


def figures_check(lines: list) -> int:
    total = 0
    for pdf in sorted(PDFS.glob("*.pdf")):
        insurer = insurer_for(pdf.name)
        pages_file = PARSED / f"{pdf.stem}.pages.json"
        if not pages_file.exists():
            print(f"skip {pdf.name}: run parse first")
            continue
        docling_pages = {int(k): v for k, v in json.loads(pages_file.read_text()).items()}
        flagged = []
        with pdfplumber.open(pdf) as doc:
            for i, page in enumerate(doc.pages, start=1):
                if insurer == "Policy Expert" and i in LIMITS_PAGES:
                    continue  # checked against the CSV instead (text layer is wrong there)
                a = figures(page.extract_text() or "")
                b = figures(docling_pages.get(i, ""))
                missing, extra = a - b, b - a
                if missing or extra:
                    flagged.append((i, dict(missing), dict(extra)))
        total += len(flagged)
        lines.append(f"## {insurer}: {len(flagged)} flagged pages\n")
        for page, missing, extra in flagged:
            lines.append(f"- PDF page {page}: only in text layer {missing or '-'}; only in Docling {extra or '-'}")
        lines.append("")
        print(f"{insurer}: {len(flagged)} pages with figure differences")
    return total


def norm_money(s: str) -> str:
    m = re.search(r"£\s?[\d,]+", s or "")
    return m.group(0).replace(" ", "") if m else ""


def limits_check(lines: list) -> list:
    pe = next((p for p in PDFS.glob("*.pdf") if insurer_for(p.name) == "Policy Expert"), None)
    doc = DoclingDocument.load_from_json(PARSED / f"{pe.stem}.json")
    docling_rows = {}
    for item, _ in doc.iterate_items():
        if isinstance(item, TableItem) and item.prov and item.prov[0].page_no in LIMITS_PAGES:
            for _, r in item.export_to_dataframe(doc=doc).iterrows():
                cells = [str(c).strip() for c in r.tolist()]
                if cells and cells[0]:
                    docling_rows[cells[0].lower()] = cells[1:]

    problems = []
    for row in csv.DictReader(LIMITS_CSV.open()):
        key = row["item"].lower().split(" (")[0]
        got_row = next((v for k, v in docling_rows.items() if key in k), None)
        if got_row is None:
            problems.append(f"{row['item']}: row not found in Docling's tables")
            continue
        for tier, got in zip(TIERS, got_row + [""] * 4):
            want = row[tier]
            if norm_money(want) and norm_money(want) != norm_money(got):
                problems.append(f"{row['item']} [{tier}]: CSV {want}, Docling '{got}'")
            if want.lower() == "not covered" and norm_money(got):
                problems.append(f"{row['item']} [{tier}]: CSV not covered, Docling '{got}'")

    lines.append(f"## Policy Expert limits table (pages 4-5): {len(problems)} mismatches vs CSV\n")
    lines += [f"- {p}" for p in problems] + [""]
    print(f"Limits table: {len(problems)} mismatches vs hand-checked CSV")
    return problems


def main():
    lines = ["# Parse check: PDF text layer vs Docling\n"]
    flagged = figures_check(lines)
    problems = limits_check(lines)

    use_csv = bool(problems)
    (PARSED / "limits_decision.json").write_text(json.dumps(
        {"use_csv_for_pe_pages_4_5": use_csv, "mismatches": len(problems)}, indent=1))
    (PARSED / "number_check.md").write_text("\n".join(lines))

    print(f"\n{flagged} pages to check by eye.")
    print("Limits table:", "using hand-checked CSV" if use_csv else "Docling is correct, using Docling")
    print("Full report: data/parsed/number_check.md")


if __name__ == "__main__":
    main()