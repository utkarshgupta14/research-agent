"""End-to-end ingestion pipeline module for ResearchPilot MVP 0 (Step 0.9).

Orchestrates:
PDF files -> Page parsing -> Document structure -> Chunking
-> Paper embeddings & Chunk embeddings -> FAISS dual index
-> Per-document JSON metadata & Master manifest catalog.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from pydantic import BaseModel, Field

from app.config import DOCUMENTS_DIR, INDEX_DIR, PAPERS_DIR
from app.ingestion.chunker import ChunkerConfig, chunk_document
from app.ingestion.parser import parse_pdf
from app.ingestion.structure import extract_document_structure
from app.models.chunk import Chunk
from app.models.paper import Paper
from app.retrieval.embeddings import EmbeddingProvider
from app.retrieval.faiss_retriever import FAISSRetriever

logger = logging.getLogger(__name__)


def derive_paper_id(file_path: Path | str) -> str:
    """Derive a stable, clean, deterministic paper identifier from a file path.

    Args:
        file_path: Path to the research paper PDF.

    Returns:
        A sanitized paper_id string.
    """
    stem = Path(file_path).stem
    # Replace non-alphanumeric characters (except dashes and dots) with underscores
    clean = re.sub(r"[^\w\-.]", "_", stem)
    clean = re.sub(r"_+", "_", clean).strip("_")
    return clean or "unknown_paper"


class PaperDocument(BaseModel):
    """Structured document record saved per paper under data/documents/."""

    paper_id: str
    paper: Paper
    num_pages: int
    num_sections: int
    num_chunks: int
    chunk_ids: list[str] = Field(default_factory=list)
    source_file: str
    ingested_at: str


class IngestionManifest(BaseModel):
    """Master manifest catalog summarizing an entire ingestion batch."""

    total_papers: int
    total_pages: int
    total_sections: int
    total_chunks: int
    total_indexed_vectors: int
    papers: list[dict[str, Any]] = Field(default_factory=list)
    failed_files: list[dict[str, str]] = Field(default_factory=list)
    created_at: str


class IngestionResult(BaseModel):
    """Telemetry returned after executing the ingestion pipeline."""

    papers: list[Paper] = Field(default_factory=list)
    chunks: list[Chunk] = Field(default_factory=list)
    total_pages: int = 0
    total_sections: int = 0
    total_chunks: int = 0
    failed_files: list[dict[str, str]] = Field(default_factory=list)
    documents_dir: Path
    index_dir: Path


class IngestionPipeline:
    """Coordinates batch ingestion, structure extraction, and retrieval indexing."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider | None = None,
        chunker_config: ChunkerConfig | None = None,
        documents_dir: Path | str | None = None,
        index_dir: Path | str | None = None,
        retriever: FAISSRetriever | None = None,
    ) -> None:
        """Initialize the ingestion pipeline.

        Args:
            embedding_provider: Embedding provider to use. Defaults to configured provider.
            chunker_config: Configuration for sentence-level chunking.
            documents_dir: Directory where paper JSON files and manifest are written.
            index_dir: Directory where FAISS binary indexes and metadata are written.
            retriever: Optional pre-configured FAISSRetriever instance.
        """
        self.chunker_config = chunker_config or ChunkerConfig()
        self.documents_dir = Path(documents_dir or DOCUMENTS_DIR)
        self.index_dir = Path(index_dir or INDEX_DIR)

        if retriever is not None:
            self.retriever = retriever
        else:
            self.retriever = FAISSRetriever(
                embedding_provider=embedding_provider,
                index_dir=self.index_dir,
            )

    def run(
        self,
        pdf_paths: Sequence[Path | str] | None = None,
        papers_dir: Path | str | None = None,
        glob_pattern: str = "*.pdf",
        limit: int | None = None,
    ) -> IngestionResult:
        """Execute the end-to-end ingestion pipeline over target PDF files.

        Args:
            pdf_paths: Explicit list of PDF files to ingest. If None, scans papers_dir.
            papers_dir: Directory to scan for PDFs if pdf_paths is not provided.
            glob_pattern: Glob pattern when scanning papers_dir (defaults to '*.pdf').
            limit: Maximum number of papers to ingest (useful for testing).

        Returns:
            An IngestionResult model containing counts, processed papers, and directories.
        """
        # 1. Resolve target files
        if pdf_paths is not None:
            resolved_paths = [Path(p) for p in pdf_paths]
        else:
            search_dir = Path(papers_dir or PAPERS_DIR)
            resolved_paths = sorted(search_dir.glob(glob_pattern))

        if limit is not None and limit > 0:
            resolved_paths = resolved_paths[:limit]

        # 2. Ensure directories exist
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        all_papers: list[Paper] = []
        all_chunks: list[Chunk] = []
        paper_summaries: list[dict[str, Any]] = []
        failed_files: list[dict[str, str]] = []

        total_pages = 0
        total_sections = 0

        # 3. Process each PDF file
        for path in resolved_paths:
            if not path.exists():
                logger.warning("Target file does not exist: %s", path)
                failed_files.append({"file": str(path), "error": "File does not exist"})
                continue

            paper_id = derive_paper_id(path)
            try:
                pages = parse_pdf(path, paper_id=paper_id)
                if not pages:
                    raise ValueError(f"No pages extracted from {path.name}")

                doc = extract_document_structure(pages)
                paper = doc.to_paper()
                chunks = chunk_document(doc, config=self.chunker_config)

                # Persist per-paper structured document record
                doc_record = PaperDocument(
                    paper_id=paper.paper_id,
                    paper=paper,
                    num_pages=len(pages),
                    num_sections=len(paper.sections),
                    num_chunks=len(chunks),
                    chunk_ids=[c.chunk_id for c in chunks],
                    source_file=str(path),
                    ingested_at=datetime.now(timezone.utc).isoformat(),
                )
                doc_json_path = self.documents_dir / f"{paper.paper_id}.json"
                doc_json_path.write_text(
                    doc_record.model_dump_json(indent=2),
                    encoding="utf-8",
                )

                all_papers.append(paper)
                all_chunks.extend(chunks)
                total_pages += len(pages)
                total_sections += len(paper.sections)

                paper_summaries.append({
                    "paper_id": paper.paper_id,
                    "title": paper.title,
                    "authors": paper.authors,
                    "year": paper.year,
                    "num_pages": len(pages),
                    "num_sections": len(paper.sections),
                    "num_chunks": len(chunks),
                    "source_path": str(path),
                })

            except Exception as exc:
                logger.error("Failed to ingest %s: %s", path.name, exc, exc_info=True)
                failed_files.append({"file": str(path), "error": str(exc)})

        # 4. Build and save FAISS indexes
        if all_papers or all_chunks:
            self.retriever.index_papers(all_papers)
            self.retriever.index_chunks(all_chunks)
            self.retriever.save(self.index_dir)

        # 5. Write master manifest
        manifest = IngestionManifest(
            total_papers=len(all_papers),
            total_pages=total_pages,
            total_sections=total_sections,
            total_chunks=len(all_chunks),
            total_indexed_vectors=len(all_papers) + len(all_chunks),
            papers=paper_summaries,
            failed_files=failed_files,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        manifest_path = self.documents_dir / "manifest.json"
        manifest_path.write_text(
            manifest.model_dump_json(indent=2),
            encoding="utf-8",
        )

        return IngestionResult(
            papers=all_papers,
            chunks=all_chunks,
            total_pages=total_pages,
            total_sections=total_sections,
            total_chunks=len(all_chunks),
            failed_files=failed_files,
            documents_dir=self.documents_dir,
            index_dir=self.index_dir,
        )


def ingest_papers(
    pdf_paths: Sequence[Path | str] | None = None,
    papers_dir: Path | str | None = None,
    documents_dir: Path | str | None = None,
    index_dir: Path | str | None = None,
    limit: int | None = None,
) -> IngestionResult:
    """Convenience helper to run the ingestion pipeline.

    Args:
        pdf_paths: Optional list of PDF paths to ingest.
        papers_dir: Directory containing PDFs.
        documents_dir: Target directory for document JSON files.
        index_dir: Target directory for FAISS indexes.
        limit: Optional maximum number of papers to ingest.

    Returns:
        IngestionResult summary.
    """
    pipeline = IngestionPipeline(
        documents_dir=documents_dir,
        index_dir=index_dir,
    )
    return pipeline.run(
        pdf_paths=pdf_paths,
        papers_dir=papers_dir,
        limit=limit,
    )
