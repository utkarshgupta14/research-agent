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

        # Step 1: Collect non-empty lines with their corresponding page number
        all_lines: list[tuple[int, str]] = []
        for page in pages:
            if page.is_empty:
                continue
            for line in page.text.splitlines():
                stripped = line.strip()
                if stripped:
                    all_lines.append((page.page_number, stripped))

        all_page_nums = [p.page_number for p in pages]
        min_page = min(all_page_nums)
        max_page = max(all_page_nums)

        if not all_lines:
            return StructuredDocument(
                paper_id=paper_id,
                source_path=source_path,
                sections=[],
                section_blocks=[],
            )

        # Step 2: Identify heading line positions in the sequential line stream
        # Each entry: (line_index, page_number, heading_title)
        heading_indices: list[tuple[int, int, str]] = []
        for idx, (p_num, line) in enumerate(all_lines):
            if _is_candidate_heading(line):
                title = _clean_heading_title(line)
                # Avoid consecutive duplicate heading lines
                if not heading_indices or heading_indices[-1][2] != title:
                    heading_indices.append((idx, p_num, title))

        # Fallback: if no headings are detected across the paper, return single 'Unknown' section
        if not heading_indices:
            combined_text = "\n".join(line for _, line in all_lines)
            fallback_block = SectionBlock(
                title="Unknown",
                page_start=min_page,
                page_end=max_page,
                pages=all_page_nums,
                text=combined_text,
            )
            return StructuredDocument(
                paper_id=paper_id,
                source_path=source_path,
                sections=[fallback_block.to_section()],
                section_blocks=[fallback_block],
            )

        # Step 3: Build disjoint section blocks using line slices
        blocks: list[SectionBlock] = []

        first_h_idx, first_h_page, _ = heading_indices[0]
        # If there is preamble before the first heading:
        # - If first heading is after page 1, emit a 'Header' block for preceding pages
        # - If first heading is on page 1, attach the preamble to that first section (e.g. Abstract)
        if first_h_idx > 0 and first_h_page > 1:
            preamble_lines = [line for _, line in all_lines[:first_h_idx]]
            preamble_pages = sorted(set(p for p, _ in all_lines[:first_h_idx]))
            blocks.append(
                SectionBlock(
                    title="Header",
                    page_start=min(preamble_pages),
                    page_end=max(preamble_pages),
                    pages=preamble_pages,
                    text="\n".join(preamble_lines),
                )
            )

        for i, (h_idx, h_page, title) in enumerate(heading_indices):
            # Line slice for this section up to the next heading
            if i + 1 < len(heading_indices):
                next_h_idx = heading_indices[i + 1][0]
                sec_line_tuples = all_lines[h_idx:next_h_idx]
            else:
                sec_line_tuples = all_lines[h_idx:]

            sec_lines = [line for _, line in sec_line_tuples]
            sec_pages = sorted(set(p for p, _ in sec_line_tuples))

            # Prepend page 1 preamble (e.g. title/authors) to the first heading on page 1
            if i == 0 and first_h_idx > 0 and first_h_page == 1:
                preamble_lines = [line for _, line in all_lines[:first_h_idx]]
                sec_lines = preamble_lines + sec_lines
                sec_pages = sorted(set(sec_pages) | set(p for p, _ in all_lines[:first_h_idx]))

            p_start = min(sec_pages) if sec_pages else h_page
            # The last section extends to document end
            p_end = max_page if i == len(heading_indices) - 1 else (max(sec_pages) if sec_pages else p_start)

            blocks.append(
                SectionBlock(
                    title=title,
                    page_start=p_start,
                    page_end=max(p_start, p_end),
                    pages=sec_pages,
                    text="\n".join(sec_lines),
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
