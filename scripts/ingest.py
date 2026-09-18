#!/usr/bin/env python3
"""Batch ingestion CLI script for ResearchPilot MVP 0 (Step 0.9).

Orchestrates full ingestion of research papers from data/papers/:
- Parses PDF pages
- Extracts document structure and sections
- Chunks text into sentence-level units
- Saves document JSON records and master manifest under data/documents/
- Embeds papers and chunks into dual FAISS IndexFlatIP indexes under data/index/

Usage:
    python scripts/ingest.py
    python scripts/ingest.py --limit 3
    python scripts/ingest.py --papers-dir data/papers --limit 5
"""

import argparse
import sys
import time
from pathlib import Path

from app.config import DOCUMENTS_DIR, INDEX_DIR, PAPERS_DIR
from app.ingestion.pipeline import IngestionPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest research paper PDFs into ResearchPilot document store and FAISS vector index."
    )
    parser.add_argument(
        "--papers-dir",
        type=Path,
        default=PAPERS_DIR,
        help=f"Directory containing PDF papers (default: {PAPERS_DIR})",
    )
    parser.add_argument(
        "--documents-dir",
        type=Path,
        default=DOCUMENTS_DIR,
        help=f"Output directory for document JSON files (default: {DOCUMENTS_DIR})",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=INDEX_DIR,
        help=f"Output directory for FAISS index binaries and metadata (default: {INDEX_DIR})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of papers to ingest (useful for testing)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.pdf",
        help="Glob pattern to match PDF files (default: *.pdf)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("\n" + "=" * 80)
    print(" ResearchPilot End-to-End Ingestion Pipeline (Step 0.9)")
    print("=" * 80)
    print(f"Papers Directory:    {args.papers_dir.resolve()}")
    print(f"Documents Directory: {args.documents_dir.resolve()}")
    print(f"Index Directory:     {args.index_dir.resolve()}")
    if args.limit:
        print(f"Limit:               {args.limit} papers")
    print("-" * 80)

    start_time = time.perf_counter()

    pipeline = IngestionPipeline(
        documents_dir=args.documents_dir,
        index_dir=args.index_dir,
    )

    print("\nStarting batch ingestion...")
    result = pipeline.run(
        papers_dir=args.papers_dir,
        glob_pattern=args.pattern,
        limit=args.limit,
    )

    elapsed = time.perf_counter() - start_time

    # Display per-paper summary
    print("\n" + "-" * 80)
    print(" Ingested Papers Summary")
    print("-" * 80)
    print(f"{'Paper ID':<40} | {'Pages':<5} | {'Secs':<5} | {'Chunks':<6} | Title")
    print("-" * 80)
    for p in result.papers:
        # Load corresponding document to get section/chunk counts
        doc_path = args.documents_dir / f"{p.paper_id}.json"
        num_pages = 0
        num_chunks = 0
        if doc_path.exists():
            import json
            doc_data = json.loads(doc_path.read_text(encoding="utf-8"))
            num_pages = doc_data.get("num_pages", 0)
            num_chunks = doc_data.get("num_chunks", 0)

        title_display = (p.title[:24] + "...") if len(p.title) > 27 else p.title
        print(f"{p.paper_id[:40]:<40} | {num_pages:<5} | {len(p.sections):<5} | {num_chunks:<6} | {title_display}")

    # Display failures if any
    if result.failed_files:
        print("\n" + "!" * 80)
        print(" Ingestion Warnings / Failed Files")
        print("!" * 80)
        for fail in result.failed_files:
            print(f"  • File:  {fail['file']}")
            print(f"    Error: {fail['error']}")

    # Display overall metrics
    total_vectors = len(result.papers) + result.total_chunks
    print("\n" + "=" * 80)
    print(" Ingestion Completed Successfully")
    print("=" * 80)
    print(f"Total Papers Ingested:    {len(result.papers)}")
    print(f"Total Pages Parsed:       {result.total_pages}")
    print(f"Total Sections Extracted: {result.total_sections}")
    print(f"Total Chunks Created:     {result.total_chunks}")
    print(f"Total Vectors Indexed:    {total_vectors} ({len(result.papers)} paper + {result.total_chunks} chunk)")
    print(f"Failed Files:             {len(result.failed_files)}")
    print(f"Elapsed Time:             {elapsed:.2f}s")
    print(f"Document Metadata:        {args.documents_dir.resolve()}")
    print(f"FAISS Retrieval Index:    {args.index_dir.resolve()}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
