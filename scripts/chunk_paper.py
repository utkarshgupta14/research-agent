#!/usr/bin/env python3
"""Demonstration script for chunking a parsed and structured PDF paper.

Usage:
    python scripts/chunk_paper.py [path/to/paper.pdf]
"""

import sys
from pathlib import Path

from app.ingestion import chunk_document, extract_document_structure, parse_pdf
from app.ingestion.chunker import ChunkerConfig, count_tokens


def inspect_paper_chunks(pdf_path: Path) -> None:
    print(f"\n{'=' * 75}")
    print(f"Chunking Paper: {pdf_path.name}")
    print(f"{'=' * 75}")

    pages = parse_pdf(pdf_path)
    doc_struct = extract_document_structure(pages)
    config = ChunkerConfig(chunk_size=600, chunk_overlap=100)
    chunks = chunk_document(doc_struct, config=config)

    token_counts = [count_tokens(c.text) for c in chunks]
    avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0

    print(f"Total Pages:         {len(pages)}")
    print(f"Sections Detected:   {len(doc_struct.sections)}")
    print(f"Total Chunks:        {len(chunks)}")
    print(f"Avg Tokens/Chunk:    {avg_tokens:.1f}")
    print(f"Min Tokens/Chunk:    {min(token_counts) if token_counts else 0}")
    print(f"Max Tokens/Chunk:    {max(token_counts) if token_counts else 0}")
    print("-" * 75)
    print(f"{'Chunk ID':<35} | {'Page':<5} | {'Toks':<5} | {'Section'}")
    print("-" * 75)

    # Print first 10 chunks as a sample
    for c in chunks[:10]:
        toks = count_tokens(c.text)
        print(f"{c.chunk_id:<35} | p. {c.page:<3} | {toks:<5} | {c.section}")

    if len(chunks) > 10:
        print(f"... and {len(chunks) - 10} more chunks")

    print("-" * 75)
    print("Preview of Chunk 1 Text:")
    print("-" * 75)
    if chunks:
        sample_text = chunks[0].text[:300] + ("..." if len(chunks[0].text) > 300 else "")
        print(sample_text)
    print("-" * 75)


def main() -> None:
    if len(sys.argv) > 1:
        pdf_path = Path(sys.argv[1])
    else:
        candidates = [
            Path("data/papers/SlowFast_Networks_for_Video_Recognition.pdf"),
            Path("data/papers/Carreira_Quo_Vadis_Action_CVPR_2017_paper.pdf"),
        ]
        existing = [p for p in candidates if p.exists()]
        pdf_path = existing[0] if existing else Path("data/papers").glob("*.pdf").__next__()

    inspect_paper_chunks(pdf_path)


if __name__ == "__main__":
    main()
