from pathlib import Path
import pytest

from app.ingestion.parser import ParsedPage
from app.ingestion.structure import (
    DocumentStructureExtractor,
    extract_document_structure,
    extract_paper,
    extract_sections,
)
from app.models.paper import Paper


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


def test_structured_document_to_paper_standard(synthetic_pages: list[ParsedPage]):
    """Verify that to_paper extracts title, authors, abstract, and sections from standard pages."""
    doc = extract_document_structure(synthetic_pages)
    paper = doc.to_paper()

    assert isinstance(paper, Paper)
    assert paper.paper_id == "paper_123"
    assert paper.title == "Paper Title"
    assert paper.authors == ["Author One", "Author Two"]
    assert paper.abstract == "This paper explores multimodal video models."
    assert len(paper.sections) == 5

    # Test convenience wrapper
    paper_direct = extract_paper(synthetic_pages)
    assert paper_direct == paper


def test_structured_document_to_paper_no_abstract_section():
    """Verify that when no Abstract section exists, to_paper falls back to the first section text."""
    pages = [
        ParsedPage(
            page_number=1,
            text=(
                "Paper Without Abstract Section\n"
                "Author One, Author Two\n"
                "1. Introduction\n"
                "This paper starts directly with an introduction paragraph."
            ),
            source_path="mock/direct_intro_2023.pdf",
            paper_id="direct_001",
        )
    ]
    paper = extract_paper(pages)

    assert paper.title == "Paper Without Abstract Section"
    assert "This paper starts directly with an introduction paragraph." in paper.abstract
    assert paper.authors == ["Author One", "Author Two"]
    assert paper.year == 2023


def test_structured_document_to_paper_fallback():
    """Verify graceful fallback for title and abstract when no headings exist."""
    pages = [
        ParsedPage(
            page_number=1,
            text="Some arbitrary unformatted plain text without any academic section markers.",
            source_path="mock/paper_with_underscores_2021.pdf",
            paper_id="raw_doc",
        )
    ]
    paper = extract_paper(pages)

    assert paper.paper_id == "raw_doc"
    assert paper.title == "Some arbitrary unformatted plain text without any academic section markers."
    assert paper.abstract == "Some arbitrary unformatted plain text without any academic section markers."
    assert paper.year == 2021


def test_abstract_pattern_across_formats():
    """Verify that ABSTRACT_PATTERN recognizes standard, spaced, and inline abstract headings."""
    from app.ingestion.structure import ABSTRACT_PATTERN

    assert ABSTRACT_PATTERN.match("Abstract")
    assert ABSTRACT_PATTERN.match("ABSTRACT")
    assert ABSTRACT_PATTERN.match("A B S T R A C T")
    assert ABSTRACT_PATTERN.match("Abstract:")
    assert ABSTRACT_PATTERN.match("Abstract—In this paper we present...")
    assert not ABSTRACT_PATTERN.match("1. Introduction")
    assert not ABSTRACT_PATTERN.match("Overview of Abstract Algebra")


def test_structured_document_to_paper_spaced_heading():
    """Verify to_paper extracts abstract when heading is space-separated (A B S T R A C T)."""
    pages = [
        ParsedPage(
            page_number=1,
            text=(
                "UniRTL: A Universal Benchmark for Tracking\n"
                "Lian Zhang, Lingxue Wang\n"
                "A B S T R A C T\n"
                "Solving tracking problems under low illumination.\n"
                "1. Introduction\n"
                "We introduce the dataset."
            ),
            source_path="mock/unirtl.pdf",
            paper_id="unirtl_001",
        )
    ]
    paper = extract_paper(pages)

    assert paper.title == "UniRTL: A Universal Benchmark for Tracking"
    assert paper.abstract == "Solving tracking problems under low illumination."
