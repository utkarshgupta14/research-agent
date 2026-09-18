"""Ingestion module for ResearchPilot."""

from app.ingestion.chunker import (
    ChunkerConfig,
    TextChunker,
    chunk_document,
)
from app.ingestion.parser import PDFParser, ParsedPage, parse_pdf
from app.ingestion.pipeline import (
    IngestionManifest,
    IngestionPipeline,
    IngestionResult,
    PaperDocument,
    derive_paper_id,
    ingest_papers,
)
from app.ingestion.structure import (
    DocumentStructureExtractor,
    SectionBlock,
    StructuredDocument,
    extract_document_structure,
    extract_paper,
    extract_sections,
)

__all__ = [
    "ChunkerConfig",
    "DocumentStructureExtractor",
    "IngestionManifest",
    "IngestionPipeline",
    "IngestionResult",
    "PaperDocument",
    "ParsedPage",
    "PDFParser",
    "SectionBlock",
    "StructuredDocument",
    "TextChunker",
    "chunk_document",
    "derive_paper_id",
    "extract_document_structure",
    "extract_paper",
    "extract_sections",
    "ingest_papers",
    "parse_pdf",
]

