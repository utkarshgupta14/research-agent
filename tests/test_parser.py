from pathlib import Path
import pytest
import pymupdf

from app.ingestion import PDFParser, ParsedPage, parse_pdf


@pytest.fixture
def sample_pdf_path(tmp_path: Path) -> Path:
    """Create a synthetic 3-page PDF fixture with text and an empty page."""
    pdf_file = tmp_path / "sample_synthetic_paper.pdf"
    doc = pymupdf.open()

    # Page 1: Title and Abstract
    p1 = doc.new_page()
    p1.insert_text((72, 72), "Research Paper Title\n\nAbstract\nThis is a sample abstract for testing.")

    # Page 2: Section 1
    p2 = doc.new_page()
    p2.insert_text((72, 72), "1. Introduction\nHere is the introductory background text.")

    # Page 3: Blank page (empty, no text)
    _p3 = doc.new_page()

    doc.save(str(pdf_file))
    doc.close()
    return pdf_file


def test_parse_pdf_with_fixture(sample_pdf_path: Path):
    """Test deterministic page-level parsing on a controlled PDF fixture."""
    pages = parse_pdf(sample_pdf_path, paper_id="synth_001")

    assert len(pages) == 3

    # Page 1 checks
    p1 = pages[0]
    assert p1.page_number == 1
    assert p1.page == 1
    assert p1.paper_id == "synth_001"
    assert p1.source_path == str(sample_pdf_path)
    assert "Research Paper Title" in p1.text
    assert "Abstract" in p1.text
    assert not p1.is_empty
    assert p1.char_count > 0

    # Page 2 checks
    p2 = pages[1]
    assert p2.page_number == 2
    assert p2.page == 2
    assert "1. Introduction" in p2.text
    assert not p2.is_empty

    # Page 3 (Empty page) checks: should handle gracefully without crashing
    p3 = pages[2]
    assert p3.page_number == 3
    assert p3.page == 3
    assert p3.text == ""
    assert p3.is_empty is True
    assert p3.char_count == 0


def test_default_paper_id_from_stem(sample_pdf_path: Path):
    """Test that paper_id defaults to filename stem when not explicitly provided."""
    pages = parse_pdf(sample_pdf_path)
    assert all(p.paper_id == "sample_synthetic_paper" for p in pages)


def test_parsed_page_serialization():
    """Test Pydantic serialization and round-trip deserialization."""
    page = ParsedPage(
        page_number=1,
        text="Sample page text content.",
        source_path="/data/sample.pdf",
        paper_id="sample_paper",
    )

    json_str = page.model_dump_json()
    reconstructed = ParsedPage.model_validate_json(json_str)

    assert reconstructed == page
    assert reconstructed.page_number == 1
    assert reconstructed.text == "Sample page text content."
    assert reconstructed.source_path == "/data/sample.pdf"
    assert reconstructed.paper_id == "sample_paper"


def test_parse_pdf_file_not_found():
    """Test that parsing a non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        parse_pdf("/non/existent/path/paper.pdf")


def test_parse_real_sample_paper():
    """Test parsing a real research paper from data/papers if present."""
    real_pdf = Path("data/papers/SlowFast_Networks_for_Video_Recognition.pdf")
    if not real_pdf.exists():
        pytest.skip(f"Real paper sample {real_pdf} not found")

    parser = PDFParser()
    pages = parser.parse(real_pdf)

    assert len(pages) == 10
    # Sequential 1-based page numbers
    assert [p.page_number for p in pages] == list(range(1, 11))

    # Check page 1 content
    assert "SlowFast Networks for Video Recognition" in pages[0].text
    assert pages[0].paper_id == "SlowFast_Networks_for_Video_Recognition"
    assert pages[0].source_path == str(real_pdf)
    assert pages[0].char_count > 1000
