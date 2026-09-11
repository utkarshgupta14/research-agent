#!/usr/bin/env python3
"""Demonstration script for parsing a sample research paper PDF using PyMuPDF.

Usage:
    python scripts/parse_paper.py [path/to/paper.pdf]
"""

import sys
from pathlib import Path

from app.ingestion import parse_pdf


def main() -> None:
    # Use provided path or default to a known sample paper in data/papers/
    default_paper = Path("data/papers/SlowFast_Networks_for_Video_Recognition.pdf")

    if len(sys.argv) > 1:
        pdf_path = Path(sys.argv[1])
    elif default_paper.exists():
        pdf_path = default_paper
    else:
        available_papers = list(Path("data/papers").glob("*.pdf"))
        if available_papers:
            pdf_path = available_papers[0]
        else:
            print("[ERROR] No sample PDF found in data/papers/ and no file argument was provided.")
            sys.exit(1)

    print(f"Parsing PDF: {pdf_path}")
    pages = parse_pdf(pdf_path)

    print(f"\nSuccessfully parsed {len(pages)} pages.")
    print(f"Paper ID: {pages[0].paper_id}")
    print(f"Source:   {pages[0].source_path}")
    print("-" * 60)

    for p in pages:
        status = "(empty page)" if p.is_empty else f"{p.char_count} chars"
        print(f"  Page {p.page_number:2d}: {status}")

    print("-" * 60)
    print("Preview of Page 1 text (first 300 characters):")
    print("-" * 60)
    print(pages[0].text[:300] + ("..." if len(pages[0].text) > 300 else ""))
    print("-" * 60)
    print("[SUCCESS] PDF parsed deterministically into page representations.")


if __name__ == "__main__":
    main()
