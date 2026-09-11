"""PDF parsing module for ResearchPilot MVP 0 (Step 0.3).

Extracts deterministic page-level text and metadata from PDF research papers using PyMuPDF.
"""

from pathlib import Path
import pymupdf
from pydantic import BaseModel, ConfigDict, Field


class ParsedPage(BaseModel):
    """Represents the parsed text content and metadata of a single PDF page."""

    model_config = ConfigDict(populate_by_name=True)

    page_number: int = Field(description="1-based physical page number in the PDF document")
    text: str = Field(default="", description="Extracted text content for this page")
    source_path: str = Field(description="Path to the source PDF file")
    paper_id: str = Field(description="Identifier of the paper document")

    @property
    def page(self) -> int:
        """Alias for page_number to align with Chunk/Citation conventions."""
        return self.page_number

    @property
    def char_count(self) -> int:
        """Total character count of the extracted page text."""
        return len(self.text)

    @property
    def is_empty(self) -> bool:
        """Indicates whether the page has little or no extracted text."""
        return not bool(self.text.strip())


class PDFParser:
    """Deterministic PDF text parser using PyMuPDF."""

    def parse(self, pdf_path: str | Path, paper_id: str | None = None) -> list[ParsedPage]:
        """Parse a PDF file into a sequence of page-level representations.

        Args:
            pdf_path: File system path to the target PDF file.
            paper_id: Optional unique identifier for the paper. Defaults to filename stem.

        Returns:
            A list of ParsedPage objects, one for each page in document order.

        Raises:
            FileNotFoundError: If the provided path does not exist.
            ValueError: If the provided path is not a regular file.
        """
        path_obj = Path(pdf_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        if not path_obj.is_file():
            raise ValueError(f"Path is not a regular file: {pdf_path}")

        resolved_paper_id = paper_id or path_obj.stem
        source_str = str(path_obj)

        pages: list[ParsedPage] = []
        with pymupdf.open(str(path_obj)) as doc:
            for page_idx, page in enumerate(doc):
                raw_text = page.get_text("text") or ""
                cleaned_text = raw_text.strip() if isinstance(raw_text, str) else str(raw_text).strip()
                pages.append(
                    ParsedPage(
                        page_number=page_idx + 1,
                        text=cleaned_text,
                        source_path=source_str,
                        paper_id=resolved_paper_id,
                    )
                )

        return pages


def parse_pdf(pdf_path: str | Path, paper_id: str | None = None) -> list[ParsedPage]:
    """Convenience function to parse a PDF file into page representations."""
    return PDFParser().parse(pdf_path=pdf_path, paper_id=paper_id)
