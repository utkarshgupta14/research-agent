from pathlib import Path
import pytest

from app.ingestion.chunker import ChunkerConfig, TextChunker, chunk_document, count_tokens
from app.ingestion.structure import SectionBlock, StructuredDocument
from app.models.chunk import Chunk


@pytest.fixture
def sample_structured_doc() -> StructuredDocument:
    """Create a synthetic StructuredDocument with multi-page and multi-section content."""
    # Section 1: Short Introduction (approx. 50 tokens) on Page 1
    s1_text = (
        "1. Introduction\n"
        "Video understanding requires capturing dynamic spatiotemporal features.\n"
        "Traditional ConvNets process video uniformly across time and space."
    )
    s1 = SectionBlock(
        title="1. Introduction",
        page_start=1,
        page_end=1,
        line_start=1,
        pages=[1],
        text=s1_text,
        lines=[(1, line) for line in s1_text.splitlines()],
    )

    # Section 2: Long Methodology (spans Page 1 and Page 2, ~500 words)
    p1_lines = [
        f"Methodology part one sentence {i}. We establish the Slow pathway architecture."
        for i in range(1, 25)
    ]
    p2_lines = [
        f"Methodology part two sentence {i}. We describe the Fast pathway operating at high temporal rate."
        for i in range(1, 25)
    ]
    s2_lines = [(1, line) for line in p1_lines] + [(2, line) for line in p2_lines]
    s2_text = "\n\n".join(p1_lines) + "\n\n" + "\n\n".join(p2_lines)

    s2 = SectionBlock(
        title="2. Methodology",
        page_start=1,
        page_end=2,
        line_start=10,
        pages=[1, 2],
        text=s2_text,
        lines=s2_lines,
    )

    # Section 3: Experiments on Page 2
    s3_text = (
        "3. Experiments\n"
        "We evaluate our model on Kinetics-400 benchmark dataset.\n"
        "Top-1 accuracy achieves 79.8% outperforming prior methods."
    )
    s3 = SectionBlock(
        title="3. Experiments",
        page_start=2,
        page_end=2,
        line_start=40,
        pages=[2],
        text=s3_text,
        lines=[(2, line) for line in s3_text.splitlines()],
    )

    return StructuredDocument(
        paper_id="test_paper_001",
        source_path="mock/path.pdf",
        sections=[s1.to_section(), s2.to_section(), s3.to_section()],
        section_blocks=[s1, s2, s3],
    )


def test_deterministic_chunk_ids(sample_structured_doc: StructuredDocument):
    """Verify that chunking the same document twice produces identical chunk IDs and contents."""
    chunker = TextChunker(ChunkerConfig(chunk_size=150, chunk_overlap=30))
    chunks_run1 = chunker.chunk_document(sample_structured_doc)
    chunks_run2 = chunker.chunk_document(sample_structured_doc)

    assert len(chunks_run1) > 0
    assert len(chunks_run1) == len(chunks_run2)

    for c1, c2 in zip(chunks_run1, chunks_run2):
        assert c1.chunk_id == c2.chunk_id
        assert c1.paper_id == c2.paper_id
        assert c1.page == c2.page
        assert c1.section == c2.section
        assert c1.text == c2.text


def test_provenance_metadata_preserved(sample_structured_doc: StructuredDocument):
    """Verify that all chunks contain valid provenance: paper_id, page, section, text."""
    chunks = chunk_document(sample_structured_doc)

    assert len(chunks) >= 3
    for chunk in chunks:
        assert isinstance(chunk, Chunk)
        assert chunk.paper_id == "test_paper_001"
        assert chunk.chunk_id.startswith("test_paper_001_c")
        assert chunk.section in ["1. Introduction", "2. Methodology", "3. Experiments"]
        assert chunk.page in [1, 2]
        assert len(chunk.text) >= 40


def test_no_empty_or_tiny_chunks():
    """Verify that empty sections and tiny noise fragments are omitted."""
    doc = StructuredDocument(
        paper_id="empty_paper",
        source_path="empty.pdf",
        sections=[],
        section_blocks=[
            SectionBlock(
                title="Tiny Section",
                page_start=1,
                page_end=1,
                line_start=1,
                pages=[1],
                text="Too short",
                lines=[(1, "Too short")],
            ),
            SectionBlock(
                title="Empty Section",
                page_start=2,
                page_end=2,
                line_start=1,
                pages=[2],
                text="   \n\n  ",
                lines=[],
            ),
        ],
    )

    chunks = chunk_document(doc, ChunkerConfig(min_chunk_chars=40))
    assert len(chunks) == 0


def test_section_boundary_isolation(sample_structured_doc: StructuredDocument):
    """Verify that chunks never blend content across section boundaries."""
    chunks = chunk_document(sample_structured_doc)

    for chunk in chunks:
        if chunk.section == "1. Introduction":
            assert "Kinetics-400" not in chunk.text
            assert "Fast pathway" not in chunk.text
        elif chunk.section == "3. Experiments":
            assert "Traditional ConvNets" not in chunk.text
            assert "Slow pathway" not in chunk.text


def test_chunk_size_and_overlap_behavior():
    """Verify that text exceeding chunk_size generates overlapping consecutive chunks."""
    long_para1 = "Alpha paragraph discussing first aspect in deep detail. " * 20
    long_para2 = "Beta paragraph expanding on theoretical implications. " * 20
    long_para3 = "Gamma paragraph providing empirical conclusions. " * 20

    full_text = f"{long_para1}\n\n{long_para2}\n\n{long_para3}"
    block = SectionBlock(
        title="Long Section",
        page_start=1,
        page_end=1,
        line_start=1,
        pages=[1],
        text=full_text,
        lines=[(1, line) for line in full_text.splitlines()],
    )
    doc = StructuredDocument(
        paper_id="overlap_test",
        source_path="overlap.pdf",
        sections=[block.to_section()],
        section_blocks=[block],
    )

    config = ChunkerConfig(chunk_size=100, chunk_overlap=30, min_chunk_chars=20)
    chunks = chunk_document(doc, config)

    assert len(chunks) >= 2
    # Verify sequential IDs
    assert chunks[0].chunk_id == "overlap_test_c0001"
    assert chunks[1].chunk_id == "overlap_test_c0002"

    # Verify that overlap exists: tail words of chunk 0 appear in chunk 1
    chunk0_words = set(chunks[0].text.split()[-15:])
    chunk1_words = set(chunks[1].text.split()[:15])
    common_words = chunk0_words.intersection(chunk1_words)
    assert len(common_words) > 0


def test_multi_page_chunk_page_attribution(sample_structured_doc: StructuredDocument):
    """Verify that chunks originating from Page 2 are attributed to Page 2."""
    config = ChunkerConfig(chunk_size=80, chunk_overlap=15)
    chunks = chunk_document(sample_structured_doc, config)

    methodology_chunks = [c for c in chunks if c.section == "2. Methodology"]
    pages_attributed = {c.page for c in methodology_chunks}

    # Methodology spans page 1 and page 2; chunks should accurately reflect both pages
    assert 1 in pages_attributed
    assert 2 in pages_attributed


def test_count_tokens():
    """Verify token counting helper with normal text and edge cases."""
    assert count_tokens("") == 0
    assert count_tokens("   ") == 0
    toks = count_tokens("Transformer neural network architecture.")
    assert toks > 0
    assert toks < 10


def test_real_sample_paper_chunking():
    """Verify chunking a real paper PDF from data/papers produces valid chunks."""
    pdf_path = Path("data/papers/SlowFast_Networks_for_Video_Recognition.pdf")
    if not pdf_path.exists():
        pytest.skip("Sample PDF not found")

    from app.ingestion.parser import parse_pdf
    from app.ingestion.structure import extract_document_structure

    pages = parse_pdf(pdf_path)
    doc = extract_document_structure(pages)
    chunks = chunk_document(doc)

    assert len(chunks) > 10
    # Sequential IDs
    assert chunks[0].chunk_id == f"{doc.paper_id}_c0001"
    assert chunks[1].chunk_id == f"{doc.paper_id}_c0002"

    # Metadata integrity
    for c in chunks:
        assert c.paper_id == doc.paper_id
        assert 1 <= c.page <= len(pages)
        assert bool(c.section)
        assert len(c.text) >= 40
