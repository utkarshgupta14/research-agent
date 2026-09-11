from pathlib import Path
import pytest

from app.ingestion.parser import ParsedPage
from app.ingestion.structure import (
    DocumentStructureExtractor,
    extract_document_structure,
    extract_sections,
)


@pytest.fixture
def synthetic_pages() -> list[ParsedPage]:
    """Create a list of ParsedPage objects simulating a 6-page research paper."""
    return [
        ParsedPage(
            page_number=1,
            text="Paper Title\nAuthor One, Author Two\n\nAbstract\nThis paper explores multimodal video models.",
            source_path="mock/paper.pdf",
            paper_id="paper_123",
        ),
        ParsedPage(
            page_number=2,
            text="1. Introduction\nVideo understanding requires capturing both spatial and temporal signals.",
            source_path="mock/paper.pdf",
            paper_id="paper_123",
        ),
        ParsedPage(
            page_number=3,
            text="Recent advances in ConvNets and Transformers have demonstrated strong progress.\nWe outline our approach below.",
            source_path="mock/paper.pdf",
            paper_id="paper_123",
        ),
        ParsedPage(
            page_number=4,
            text="2. Methodology\nOur architecture consists of two primary pathways operating at different frame rates.",
            source_path="mock/paper.pdf",
            paper_id="paper_123",
        ),
        ParsedPage(
            page_number=5,
            text="3. Experiments\nWe benchmark on Kinetics-400 and Charades.\nResults show superior efficiency.",
            source_path="mock/paper.pdf",
            paper_id="paper_123",
        ),
        ParsedPage(
            page_number=6,
            text="References\n[1] Author A, 2020.\n[2] Author B, 2022.",
            source_path="mock/paper.pdf",
            paper_id="paper_123",
        ),
    ]


def test_extract_sections_basic(synthetic_pages: list[ParsedPage]):
    """Verify section titles and multi-page spans from synthetic pages."""
    sections = extract_sections(synthetic_pages)

    assert len(sections) == 5
    titles = [s.title for s in sections]
    assert titles == [
        "Abstract",
        "1. Introduction",
        "2. Methodology",
        "3. Experiments",
        "References",
    ]

    # Verify multi-page range for Introduction (spans page 2 to 3)
    intro_sec = next(s for s in sections if s.title == "1. Introduction")
    assert intro_sec.page_start == 2
    assert intro_sec.page_end == 3

    # Verify other page ranges
    abstract_sec = next(s for s in sections if s.title == "Abstract")
    assert abstract_sec.page_start == 1
    assert abstract_sec.page_end == 1

    method_sec = next(s for s in sections if s.title == "2. Methodology")
    assert method_sec.page_start == 4
    assert method_sec.page_end == 4

    exp_sec = next(s for s in sections if s.title == "3. Experiments")
    assert exp_sec.page_start == 5
    assert exp_sec.page_end == 5

    ref_sec = next(s for s in sections if s.title == "References")
    assert ref_sec.page_start == 6
    assert ref_sec.page_end == 6


def test_extract_document_structure_text_association(synthetic_pages: list[ParsedPage]):
    """Verify StructuredDocument associates the right pages and aggregated text with each section."""
    doc_struct = extract_document_structure(synthetic_pages)

    assert doc_struct.paper_id == "paper_123"
    assert doc_struct.source_path == "mock/paper.pdf"
    assert len(doc_struct.section_blocks) == 5

    intro_block = next(b for b in doc_struct.section_blocks if b.title == "1. Introduction")
    assert intro_block.page_start == 2
    assert intro_block.page_end == 3
    assert intro_block.pages == [2, 3]
    assert "Video understanding requires" in intro_block.text
    assert "Recent advances in ConvNets" in intro_block.text


def test_fallback_to_unknown_when_no_headings():
    """Verify that a paper without detected headings safely falls back to a single Unknown section."""
    pages = [
        ParsedPage(page_number=1, text="Just continuous narrative text with no section headings.", source_path="test.pdf", paper_id="p1"),
        ParsedPage(page_number=2, text="More text continuing the story on page 2.", source_path="test.pdf", paper_id="p1"),
    ]

    sections = extract_sections(pages)
    assert len(sections) == 1
    assert sections[0].title == "Unknown"
    assert sections[0].page_start == 1
    assert sections[0].page_end == 2


def test_empty_pages_list():
    """Verify that an empty list of pages returns empty structure without raising exceptions."""
    doc = DocumentStructureExtractor().extract([])
    assert doc.sections == []
    assert doc.section_blocks == []


def test_real_sample_paper_structure():
    """Verify structure extraction on a real sample paper if present."""
    sample_pdf = Path("data/papers/SlowFast_Networks_for_Video_Recognition.pdf")
    if not sample_pdf.exists():
        pytest.skip("Sample PDF not found")

    from app.ingestion.parser import parse_pdf
    pages = parse_pdf(sample_pdf)
    doc_struct = extract_document_structure(pages)

    assert len(doc_struct.sections) >= 4
    section_titles = [s.title.lower() for s in doc_struct.sections]

    # Verify core sections are detected
    assert any("abstract" in t for t in section_titles)
    assert any("introduction" in t for t in section_titles)
    assert any("related work" in t for t in section_titles)

    # Verify all section ranges are valid
    for s in doc_struct.sections:
        assert 1 <= s.page_start <= s.page_end <= len(pages)


def test_numbered_lists_and_long_lines_ignored():
    """Verify that numbered list sentences and long lines are not misclassified as headings."""
    pages = [
        ParsedPage(
            page_number=1,
            text=(
                "1. Introduction\n"
                "Here is our introductory text.\n"
                "1. We propose a new two-stream architecture for recognition\n"
                "2. Our findings show that temporal modeling is very important\n"
                "3. This is an unusually long numbered line with way more than ten words in the title string\n"
                "2. Methodology\n"
                "Description of methodology."
            ),
            source_path="mock/paper.pdf",
            paper_id="paper_lists",
        )
    ]

    sections = extract_sections(pages)
    titles = [s.title for s in sections]

    # Only actual section headings should be detected
    assert titles == ["1. Introduction", "2. Methodology"]


def test_disjoint_text_when_sections_share_page():
    """Verify that multiple sections sharing the same page have disjoint text without duplication."""
    page_text = (
        "1. Introduction\n"
        "Introduction paragraph one.\n"
        "2. Related Work\n"
        "Related work paragraph.\n"
        "3. Methodology\n"
        "Methodology paragraph."
    )
    pages = [
        ParsedPage(page_number=1, text=page_text, source_path="mock/paper.pdf", paper_id="p1")
    ]

    doc = extract_document_structure(pages)
    assert len(doc.section_blocks) == 3

    intro_b = doc.section_blocks[0]
    related_b = doc.section_blocks[1]
    method_b = doc.section_blocks[2]

    assert intro_b.title == "1. Introduction"
    assert "Introduction paragraph one." in intro_b.text
    assert "Related work paragraph." not in intro_b.text
    assert "Methodology paragraph." not in intro_b.text

    assert related_b.title == "2. Related Work"
    assert "Introduction paragraph one." not in related_b.text
    assert "Related work paragraph." in related_b.text
    assert "Methodology paragraph." not in related_b.text

    assert method_b.title == "3. Methodology"
    assert "Introduction paragraph one." not in method_b.text
    assert "Related work paragraph." not in method_b.text
    assert "Methodology paragraph." in method_b.text
