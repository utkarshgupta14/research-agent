"""Document structure extraction module for ResearchPilot MVP 0 (Step 0.4).

Extracts basic, deterministic document sections from page-level parsed text
using lightweight heuristics without using an LLM.
"""

from __future__ import annotations

import re
from typing import Sequence
from pydantic import BaseModel, Field

from app.ingestion.parser import ParsedPage
from app.models.paper import Section


# Common standard unnumbered section titles in research papers (case-insensitive)
STANDARD_HEADINGS: set[str] = {
    "abstract",
    "introduction",
    "related work",
    "prior work",
    "background",
    "methodology",
    "method",
    "methods",
    "approach",
    "proposed method",
    "model",
    "experiments",
    "experiment",
    "experimental setup",
    "experimental results",
    "results",
    "results and discussion",
    "discussion",
    "conclusion",
    "conclusions",
    "conclusion and future work",
    "references",
    "bibliography",
    "acknowledgments",
    "acknowledgements",
    "appendix",
}

# Regex pattern for numbered headings:
# Matches: "1. Introduction", "1 Introduction", "1.1 Model", "II. Related Work", "A. Proof"
# Constrains section numbers to 1-2 digits (excluding 4-digit years or counts like 400)
# and ensures title starts with a capital letter.
NUMBERED_HEADING_PATTERN = re.compile(
    r"^(?:(?:[1-9][0-9]?(?:\.[0-9]{1,2})*|[IVXLCDM]{1,4})\.?|[A-Z]\.)\s+([A-Z][A-Za-z0-9\s\-:,]{2,60})$"
)


def _is_candidate_heading(line: str) -> bool:
    """Determine whether a single text line looks like an obvious section heading."""
    cleaned = line.strip()
    if not cleaned or len(cleaned) > 70:
        return False

    # Headings must be capitalized or title-cased in research papers
    if not (cleaned[0].isupper() or cleaned[0].isdigit()):
        return False

    # Check against standard unnumbered section titles
    lower_cleaned = cleaned.lower()
    if lower_cleaned in STANDARD_HEADINGS:
        # Avoid generic short words unless title-cased/uppercase
        if len(cleaned.split()) == 1 and not (cleaned.istitle() or cleaned.isupper()):
            return False
        return True

    # Check for numbered heading pattern
    match = NUMBERED_HEADING_PATTERN.match(cleaned)
    if match:
        # Ignore common non-heading lines like "Figure 1: ...", "Table 2: ..."
        if lower_cleaned.startswith(("figure", "fig.", "table", "tab.")):
            return False
        # Avoid lines ending with punctuation typically seen in body text
        if cleaned.endswith((".", ",", ";", ":")):
            return False

        # Exclude numbered list items or sentences (word count > 10 or starting with sentence pronouns)
        title_words = match.group(1).split()
        if len(title_words) > 10:
            return False
        if title_words and title_words[0] in {"We", "Our", "This", "These", "There", "Here", "It"}:
            return False

        return True

    return False


def _clean_heading_title(line: str) -> str:
    """Normalize a matched heading line into a clean title string."""
    return " ".join(line.strip().split())


class SectionBlock(BaseModel):
    """Represents a section with its page span and aggregated text content."""

    title: str
    page_start: int
    page_end: int
    pages: list[int] = Field(default_factory=list)
    text: str = ""

    def to_section(self) -> Section:
        """Convert this block to the core Section model."""
        return Section(
            title=self.title,
            page_start=self.page_start,
            page_end=self.page_end,
        )


class StructuredDocument(BaseModel):
    """Represents the structured representation of a parsed research paper."""

    paper_id: str
    source_path: str
    sections: list[Section] = Field(default_factory=list)
    section_blocks: list[SectionBlock] = Field(default_factory=list)


class DocumentStructureExtractor:
    """Extracts section boundaries and associates page text using simple heuristics."""

    def extract(self, pages: Sequence[ParsedPage]) -> StructuredDocument:
        """Extract sections and structured text from parsed PDF pages.

        Args:
            pages: Sequence of ParsedPage objects from the PDF parser.

        Returns:
            A StructuredDocument containing detected Sections and SectionBlocks.
        """
        if not pages:
            return StructuredDocument(
                paper_id="",
                source_path="",
                sections=[],
                section_blocks=[],
            )

        paper_id = pages[0].paper_id
        source_path = pages[0].source_path

        # Step 1: Detect heading locations across pages
        # Each entry: (page_number, heading_title)
        detected_headings: list[tuple[int, str]] = []

        for page in pages:
            if page.is_empty:
                continue

            for line in page.text.splitlines():
                if _is_candidate_heading(line):
                    title = _clean_heading_title(line)
                    # Don't add identical consecutive heading on the same page
                    if not detected_headings or detected_headings[-1] != (page.page_number, title):
                        detected_headings.append((page.page_number, title))

        total_pages = max(p.page_number for p in pages)

        # Fallback: if no headings are detected, create a single 'Unknown' section
        if not detected_headings:
            combined_text = "\n\n".join(p.text for p in pages if p.text.strip())
            all_page_nums = [p.page_number for p in pages]
            fallback_block = SectionBlock(
                title="Unknown",
                page_start=min(all_page_nums),
                page_end=max(all_page_nums),
                pages=all_page_nums,
                text=combined_text,
            )
            return StructuredDocument(
                paper_id=paper_id,
                source_path=source_path,
                sections=[fallback_block.to_section()],
                section_blocks=[fallback_block],
            )

        # Step 2: Build section boundaries
        # Check if the first detected heading starts after page 1 or after introductory content
        first_heading_page = detected_headings[0][0]
        section_ranges: list[tuple[str, int, int]] = []

        # If the first heading starts after page 1, mark the preamble as 'Abstract' or 'Header'
        if first_heading_page > 1:
            section_ranges.append(("Header", 1, first_heading_page - 1))

        # Build ranges for detected headings
        for idx, (page_num, title) in enumerate(detected_headings):
            if idx + 1 < len(detected_headings):
                next_page_num = detected_headings[idx + 1][0]
                # If next heading is on the same page, this section starts on page_num and ends on next_page_num
                end_page = max(page_num, next_page_num)
            else:
                end_page = total_pages

            section_ranges.append((title, page_num, end_page))

        # Adjust overlaps: a section's page_end should be at least page_start
        adjusted_ranges: list[tuple[str, int, int]] = []
        for idx, (title, p_start, p_end) in enumerate(section_ranges):
            if idx + 1 < len(section_ranges):
                next_start = section_ranges[idx + 1][1]
                p_end = max(p_start, next_start if next_start == p_start else next_start - 1)
            else:
                p_end = total_pages
            adjusted_ranges.append((title, p_start, max(p_start, p_end)))

        # Step 3: Associate pages and text with sections
        blocks: list[SectionBlock] = []
        page_map = {p.page_number: p for p in pages}

        for title, p_start, p_end in adjusted_ranges:
            sec_pages: list[int] = []
            sec_texts: list[str] = []

            for p_num in range(p_start, p_end + 1):
                if p_num in page_map:
                    sec_pages.append(p_num)
                    page_text = page_map[p_num].text.strip()
                    if page_text:
                        sec_texts.append(page_text)

            blocks.append(
                SectionBlock(
                    title=title,
                    page_start=p_start,
                    page_end=p_end,
                    pages=sec_pages,
                    text="\n\n".join(sec_texts),
                )
            )

        sections = [b.to_section() for b in blocks]

        return StructuredDocument(
            paper_id=paper_id,
            source_path=source_path,
            sections=sections,
            section_blocks=blocks,
        )


def extract_sections(pages: Sequence[ParsedPage]) -> list[Section]:
    """Convenience function to extract a list of Section models from parsed pages."""
    return DocumentStructureExtractor().extract(pages).sections


def extract_document_structure(pages: Sequence[ParsedPage]) -> StructuredDocument:
    """Convenience function to extract a StructuredDocument from parsed pages."""
    return DocumentStructureExtractor().extract(pages)
