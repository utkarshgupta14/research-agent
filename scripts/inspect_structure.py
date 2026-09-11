#!/usr/bin/env python3
"""Inspection script for document structure extraction on real research papers.

Usage:
    python scripts/inspect_structure.py [path/to/paper.pdf]
"""

import sys
from pathlib import Path

from app.ingestion import extract_document_structure, parse_pdf


def inspect_paper(pdf_path: Path) -> None:
    print(f"\n{'=' * 70}")
    print(f"Inspecting Document Structure: {pdf_path.name}")
    print(f"{'=' * 70}")

    pages = parse_pdf(pdf_path)
    total_pages = len(pages)
    doc_struct = extract_document_structure(pages)

    print(f"Total Pages:      {total_pages}")
    print(f"Sections Detected: {len(doc_struct.sections)}")
    print("-" * 70)
    print(f"{'Section Title':<45} | {'Page Range':<12} | {'Pages'}")
    print("-" * 70)

    for block in doc_struct.section_blocks:
        page_range_str = f"pp. {block.page_start}–{block.page_end}"
        pages_list_str = ", ".join(str(p) for p in block.pages)
        print(f"{block.title:<45} | {page_range_str:<12} | {pages_list_str}")

    print("-" * 70)


def main() -> None:
    papers_to_check: list[Path] = []

    if len(sys.argv) > 1:
        papers_to_check.append(Path(sys.argv[1]))
    else:
        candidates = [
            Path("data/papers/SlowFast_Networks_for_Video_Recognition.pdf"),
            Path("data/papers/Carreira_Quo_Vadis_Action_CVPR_2017_paper.pdf"),
        ]
        papers_to_check = [p for p in candidates if p.exists()]
        if not papers_to_check:
            available = list(Path("data/papers").glob("*.pdf"))
            if available:
                papers_to_check = available[:2]

    if not papers_to_check:
        print("[ERROR] No sample PDFs found in data/papers/.")
        sys.exit(1)

    for paper in papers_to_check:
        inspect_paper(paper)


if __name__ == "__main__":
    main()
