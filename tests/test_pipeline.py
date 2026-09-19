"""Unit and integration tests for the End-to-End Ingestion Pipeline (Step 0.9).

Tests:
- Deterministic paper ID derivation.
- End-to-end execution with synthetic embeddings (offline, zero API tokens).
- Per-paper JSON documents and master manifest.json schema and counts.
- Graceful fault tolerance on corrupted or non-PDF files.
- Repeatability / idempotency across consecutive runs.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
import pytest
import fitz  # PyMuPDF

from app.ingestion.pipeline import (
    IngestionPipeline,
    derive_paper_id,
)
from app.retrieval.embeddings import EmbeddingProvider, _l2_normalize


class SyntheticEmbeddingProvider(EmbeddingProvider):
    """Deterministic, offline embedding provider for fast unit tests."""

    VOCAB: dict[str, int] = {
        "transformer": 0,
        "attention": 0,
        "residual": 1,
        "resnet": 1,
        "cnn": 1,
        "video": 2,
        "action": 2,
        "spatiotemporal": 2,
        "tracking": 3,
        "benchmark": 3,
        "cookie": 4,
        "baking": 4,
        "recipe": 4,
    }

    @property
    def dimension(self) -> int:
        return 6

    def _embed_text(self, text: str) -> list[float]:
        vec = [0.0] * self.dimension
        words = text.lower().split()
        for w in words:
            clean_w = w.strip(".,;:!?()[]\"'")
            if clean_w in self.VOCAB:
                vec[self.VOCAB[clean_w]] += 1.0
            else:
                vec[5] += 0.1
        if sum(vec) == 0.0:
            vec[5] = 1.0
        return _l2_normalize(vec)

    def embed_query(self, text: str) -> list[float]:
        return self._embed_text(text)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_text(t) for t in texts]


def create_mock_pdf(path: Path, title: str, sections_and_text: list[tuple[str, str]]) -> None:
    """Helper to generate a valid minimal PDF file on the fly for testing."""
    doc = fitz.open()
    page = doc.new_page()

    lines = [title, "Author One, Author Two", ""]
    for heading, body in sections_and_text:
        lines.append(heading)
        lines.append(body)
        lines.append("")

    full_text = "\n".join(lines)
    page.insert_text((50, 72), full_text, fontsize=11)
    doc.save(str(path))
    doc.close()


# ==============================================================================
# Unit Tests
# ==============================================================================

def test_derive_paper_id():
    """Verify deterministic and sanitized paper ID creation."""
    assert derive_paper_id("Attention Is All You Need.pdf") == "Attention_Is_All_You_Need"
    assert derive_paper_id("/path/to/2006.03876v4.pdf") == "2006.03876v4"
    assert derive_paper_id("SlowFast__Networks (CVPR).pdf") == "SlowFast_Networks_CVPR"
    assert derive_paper_id("complex   spaces   and---dashes.pdf") == "complex_spaces_and---dashes"


def test_pipeline_end_to_end_execution(tmp_path: Path):
    """Verify full pipeline runs from PDF parsing to document store and FAISS indexing."""
    papers_dir = tmp_path / "papers"
    documents_dir = tmp_path / "documents"
    index_dir = tmp_path / "index"
    papers_dir.mkdir()

    # Create 2 synthetic PDFs
    pdf1 = papers_dir / "paper_alpha.pdf"
    create_mock_pdf(
        pdf1,
        title="Alpha Transformer Models",
        sections_and_text=[
            ("Abstract", "We propose the transformer architecture based on self-attention mechanisms."),
            ("1. Introduction", "Attention models have achieved remarkable success in NLP benchmarks."),
            ("2. Method", "Multi-head attention maps queries and keys across representation subspaces."),
        ],
    )

    pdf2 = papers_dir / "paper_beta.pdf"
    create_mock_pdf(
        pdf2,
        title="Beta Residual Vision Networks",
        sections_and_text=[
            ("Abstract", "Residual networks ease the training of substantially deeper architectures."),
            ("1. Introduction", "Deep convolutional networks demonstrate strong visual representation."),
        ],
    )

    provider = SyntheticEmbeddingProvider()
    pipeline = IngestionPipeline(
        embedding_provider=provider,
        documents_dir=documents_dir,
        index_dir=index_dir,
    )

    result = pipeline.run(papers_dir=papers_dir)

    # 1. Verify telemetry counts
    assert len(result.papers) == 2
    assert result.total_pages == 2
    assert result.total_sections >= 4
    assert result.total_chunks >= 2
    assert len(result.failed_files) == 0

    # 2. Verify per-paper document metadata in data/documents/
    doc_alpha = documents_dir / "paper_alpha.json"
    doc_beta = documents_dir / "paper_beta.json"
    assert doc_alpha.exists()
    assert doc_beta.exists()

    alpha_data = json.loads(doc_alpha.read_text(encoding="utf-8"))
    assert alpha_data["paper_id"] == "paper_alpha"
    assert "Alpha Transformer Models" in alpha_data["paper"]["title"]
    assert alpha_data["num_chunks"] > 0
    assert len(alpha_data["chunk_ids"]) == alpha_data["num_chunks"]

    # 3. Verify master manifest.json
    manifest_path = documents_dir / "manifest.json"
    assert manifest_path.exists()
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_data["total_papers"] == 2
    assert manifest_data["total_indexed_vectors"] == 2 + result.total_chunks
    assert len(manifest_data["papers"]) == 2

    # 4. Verify FAISS index files
    assert (index_dir / "papers.index").exists()
    assert (index_dir / "papers.json").exists()
    assert (index_dir / "chunks.index").exists()
    assert (index_dir / "chunks.json").exists()
    assert (index_dir / "index_meta.json").exists()


def test_pipeline_fault_tolerance(tmp_path: Path):
    """Verify that a corrupted or bad PDF does not crash the pipeline."""
    papers_dir = tmp_path / "papers"
    documents_dir = tmp_path / "documents"
    index_dir = tmp_path / "index"
    papers_dir.mkdir()

    # 1. Valid PDF
    valid_pdf = papers_dir / "valid_paper.pdf"
    create_mock_pdf(
        valid_pdf,
        title="Valid Machine Learning Paper",
        sections_and_text=[("Abstract", "Clean valid abstract text for testing.")],
    )

    # 2. Corrupt / invalid PDF (plain text with .pdf extension)
    corrupt_pdf = papers_dir / "corrupted_file.pdf"
    corrupt_pdf.write_text("This is completely corrupt non-PDF plain text content.")

    provider = SyntheticEmbeddingProvider()
    pipeline = IngestionPipeline(
        embedding_provider=provider,
        documents_dir=documents_dir,
        index_dir=index_dir,
    )

    result = pipeline.run(papers_dir=papers_dir)

    # Valid paper was ingested
    assert len(result.papers) == 1
    assert result.papers[0].paper_id == "valid_paper"
    assert (documents_dir / "valid_paper.json").exists()

    # Corrupt paper was caught and reported
    assert len(result.failed_files) == 1
    assert "corrupted_file.pdf" in result.failed_files[0]["file"]
    assert result.failed_files[0]["error"] != ""

    # Manifest reflects the failure
    manifest_data = json.loads((documents_dir / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest_data["failed_files"]) == 1


def test_pipeline_repeatability_and_idempotency(tmp_path: Path):
    """Verify that running ingestion twice produces identical IDs without duplicate vectors."""
    papers_dir = tmp_path / "papers"
    documents_dir = tmp_path / "documents"
    index_dir = tmp_path / "index"
    papers_dir.mkdir()

    pdf_file = papers_dir / "repeatable_paper.pdf"
    create_mock_pdf(
        pdf_file,
        title="Repeatable Paper",
        sections_and_text=[
            ("Abstract", "Abstract text here."),
            ("1. Intro", "Intro text here."),
        ],
    )

    provider = SyntheticEmbeddingProvider()
    pipeline = IngestionPipeline(
        embedding_provider=provider,
        documents_dir=documents_dir,
        index_dir=index_dir,
    )

    # Run 1
    result1 = pipeline.run(papers_dir=papers_dir)

    # Run 2
    result2 = pipeline.run(papers_dir=papers_dir)

    assert len(result1.papers) == len(result2.papers) == 1
    assert result1.total_chunks == result2.total_chunks
    assert [p.paper_id for p in result1.papers] == [p.paper_id for p in result2.papers]
    assert [c.chunk_id for c in result1.chunks] == [c.chunk_id for c in result2.chunks]

    # Verify FAISS index meta shows exact same vector count
    index_meta = json.loads((index_dir / "index_meta.json").read_text(encoding="utf-8"))
    assert index_meta["num_papers"] == 1
    assert index_meta["num_chunks"] == result1.total_chunks
