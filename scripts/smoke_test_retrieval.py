#!/usr/bin/env python3
"""Demonstration and smoke-test script for FAISS dense retrieval (Step 0.8).

Ingests sample research papers, indexes them at both the paper and chunk levels,
and performs live semantic search and evidence retrieval using the configured
EmbeddingProvider. Also verifies index persistence (save and load).

Usage:
    python scripts/smoke_test_retrieval.py
    python scripts/smoke_test_retrieval.py [path/to/paper1.pdf] [path/to/paper2.pdf]
"""

import sys
import shutil
from pathlib import Path

from app.ingestion import (
    parse_pdf,
    extract_paper,
    extract_document_structure,
    chunk_document,
)
from app.retrieval.faiss_retriever import FAISSRetriever


def run_retrieval_smoke_test(pdf_paths: list[Path]) -> None:
    print("\n" + "=" * 75)
    print("ResearchPilot Retrieval Smoke Test (Step 0.8)")
    print("=" * 75)

    papers = []
    all_chunks = []

    print("\n1. Ingesting Papers & Chunks:")
    print("-" * 75)
    for p in pdf_paths:
        if not p.exists():
            print(f"Warning: File not found: {p}")
            continue

        pages = parse_pdf(p)
        paper = extract_paper(pages)
        papers.append(paper)

        doc_struct = extract_document_structure(pages)
        chunks = chunk_document(doc_struct)
        all_chunks.extend(chunks)

        print(f"  • {p.name}")
        print(f"    Title:    {paper.title}")
        print(f"    Sections: {len(paper.sections)} | Chunks: {len(chunks)}")

    if not papers:
        print("Error: No valid papers found to index.")
        return

    print(f"\nTotal Papers: {len(papers)} | Total Chunks: {len(all_chunks)}")

    # Initialize retriever
    retriever = FAISSRetriever()
    print(f"Using Embedding Provider: {type(retriever.embedding_provider).__name__}")
    print(f"Embedding Dimension:      {retriever.dimension}")

    print("\n2. Indexing into FAISS (IndexFlatIP):")
    print("-" * 75)
    retriever.index_papers(papers)
    print(f"  ✓ Indexed {len(papers)} papers into paper index")
    retriever.index_chunks(all_chunks)
    print(f"  ✓ Indexed {len(all_chunks)} chunks into chunk index")

    # 3. Paper Search
    query_paper = "spatiotemporal 3D convolutional architectures for video"
    print("\n3. Paper-Level Semantic Search:")
    print("-" * 75)
    print(f"Query: \"{query_paper}\"")
    paper_matches = retriever.search_papers(query_paper, k=min(3, len(papers)))
    for rank, p in enumerate(paper_matches, start=1):
        print(f"  [{rank}] Score: {p.score:.4f} | ID: {p.paper_id}")
        print(f"      Title: {p.title}")

    # 4. Evidence Retrieval
    query_evidence = "slow pathway operating at low frame rate"
    print("\n4. Chunk-Level Evidence Retrieval:")
    print("-" * 75)
    print(f"Query: \"{query_evidence}\"")
    evidence_matches = retriever.retrieve_evidence(query_evidence, k=2)
    for ev in evidence_matches:
        print(f"\n  [{ev.source_id}] Cosine Score: {ev.score:.4f}")
        print(f"      Paper:   {ev.paper_id}")
        print(f"      Section: {ev.section} (Page {ev.page})")
        preview = ev.text[:180].replace("\n", " ") + ("..." if len(ev.text) > 180 else "")
        print(f"      Text:    {preview}")

    # 5. Persistence Test (Save & Load)
    print("\n5. Persistence Verification (Save & Load):")
    print("-" * 75)
    scratch_dir = Path("data/scratch_retrieval_smoke_test")
    try:
        retriever.save(scratch_dir)
        print(f"  ✓ Saved indexes and metadata to {scratch_dir}")

        loaded_retriever = FAISSRetriever.from_saved(scratch_dir)
        print("  ✓ Successfully loaded retriever from disk")

        re_evidence = loaded_retriever.retrieve_evidence(query_evidence, k=2)
        assert len(re_evidence) == len(evidence_matches), "Mismatch in count of re-retrieved evidence"
        assert re_evidence[0].chunk_id == evidence_matches[0].chunk_id, "Mismatch in top chunk ID after reload"
        print("  ✓ Verified search parity after reload: Exact match")
    finally:
        if scratch_dir.exists():
            shutil.rmtree(scratch_dir)

    print("\n" + "=" * 75)
    print("Smoke Test Passed Successfully!")
    print("=" * 75 + "\n")


def main() -> None:
    if len(sys.argv) > 1:
        pdf_paths = [Path(arg) for arg in sys.argv[1:]]
    else:
        # Default to two representative papers
        candidates = [
            Path("data/papers/SlowFast_Networks_for_Video_Recognition.pdf"),
            Path("data/papers/Tran_Learning_Spatiotemporal_Features_ICCV_2015_paper.pdf"),
        ]
        pdf_paths = [p for p in candidates if p.exists()]
        if not pdf_paths:
            pdf_paths = list(Path("data/papers").glob("*.pdf"))[:2]

    run_retrieval_smoke_test(pdf_paths)


if __name__ == "__main__":
    main()
