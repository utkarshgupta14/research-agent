"""Document structure extraction module for ResearchPilot MVP 0 (Step 0.4).

Extracts basic, deterministic document sections from page-level parsed text
using lightweight heuristics without using an LLM.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Sequence
from pydantic import BaseModel, Field

from app.ingestion.parser import ParsedPage
from app.models.paper import Paper, Section


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

# Regex pattern matching 'Abstract' across formats: standard, all-caps, space-separated letters
# (e.g. 'A B S T R A C T'), and with trailing punctuation/dashes (e.g. 'Abstract—...', 'Abstract:').
ABSTRACT_PATTERN = re.compile(
    r"^\s*a\s*b\s*s\s*t\s*r\s*a\s*c\s*t(?:\b|[—\-:])",
    re.IGNORECASE,
)


def _is_candidate_heading(line: str) -> bool:
    """Determine whether a single text line looks like an obvious section heading."""
    cleaned = line.strip()
    if not cleaned or len(cleaned) > 70:
        return False

    # Check for abstract heading (standard, spaced "A B S T R A C T", or with punctuation)
    if ABSTRACT_PATTERN.match(cleaned):
        return True

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
    cleaned = line.strip()
    if ABSTRACT_PATTERN.match(cleaned):
        return "Abstract"
    return " ".join(cleaned.split())


class SectionBlock(BaseModel):
    """Represents a section with its page span and aggregated text content."""

    title: str
    page_start: int
    page_end: int
    line_start: int = 1
    pages: list[int] = Field(default_factory=list)
    text: str = ""
    lines: list[tuple[int, str]] = Field(default_factory=list)

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
    preamble_lines: list[str] = Field(default_factory=list)

    def to_paper(self) -> Paper:
        """Construct the domain Paper model from extracted structure.

        Extracts:
        - title: from Page 1 preamble (lines before first heading) or fallback to sanitized filename.
        - abstract: from detected 'Abstract' section block (with heading/preamble stripped), or fallback.
        - authors: candidate author names from Page 1 preamble lines.
        - year: publication year from path, paper_id, or preamble lines.
        - sections: self.sections.
        """
        return Paper(
            paper_id=self.paper_id,
            title=self._extract_title(),
            authors=self._extract_authors(),
            year=self._extract_year(),
            abstract=self._extract_abstract(),
            source_path=self.source_path,
            sections=self.sections,
        )

    def _get_preamble_lines(self) -> list[str]:
        """Get preamble lines either from self.preamble_lines or from the first block."""
        if self.preamble_lines:
            return self.preamble_lines
        if not self.section_blocks:
            return []
        lines = [l.strip() for _, l in self.section_blocks[0].lines if l.strip()]
        cand: list[str] = []
        for l in lines:
            if ABSTRACT_PATTERN.match(l):
                break
            cand.append(l)
        return cand

    def _extract_abstract(self) -> str:
        """Extract clean abstract text.

        1. Look for the first section block matching ABSTRACT_PATTERN.
        2. Strip the heading label (and prefix if inline), returning clean body text.
        3. Fallback to the first section block's text if no Abstract section was found.
        """
        abs_block = next((b for b in self.section_blocks if ABSTRACT_PATTERN.match(b.title)), None)
        if abs_block:
            lines = [l.strip() for l in abs_block.text.splitlines() if l.strip()]
            for idx, l in enumerate(lines):
                if ABSTRACT_PATTERN.match(l):
                    cleaned_inline = ABSTRACT_PATTERN.sub("", l).strip()
                    if cleaned_inline:
                        body = [cleaned_inline] + lines[idx + 1:]
                    else:
                        body = lines[idx + 1:]
                    return " ".join(body).strip()
            return abs_block.text.strip()

        # Fallback to the first section block
        if self.section_blocks:
            return self.section_blocks[0].text.strip()
        return ""

    def _extract_title(self) -> str:
        """Extract paper title from preamble lines or fallback to sanitized filename."""
        preamble = self._get_preamble_lines()
        candidates: list[str] = []
        for line in preamble:
            stripped = line.strip()
            if not stripped:
                continue
            lower = stripped.lower()
            if any(k in lower for k in [
                "sciencedirect", "journal homepage", "contents lists available",
                "arxiv.org", "doi.org", "http://", "https://", "elsevier",
                "all rights reserved", "ieee", "cvpr", "iccv", "neurips"
            ]):
                continue
            if stripped.isdigit():
                continue
            candidates.append(stripped)

        if candidates:
            title = candidates[0]
            if len(candidates) > 1 and len(title) < 60 and not title.endswith((".", ":")):
                next_line = candidates[1]
                if not any(ch in next_line for ch in [",", "@", "†", "*"]) and not any(
                    w in next_line.lower() for w in ["university", "department", "lab", "institute", "center", "school"]
                ):
                    title = f"{title} {next_line}"
            return title

        if self.source_path:
            stem = Path(self.source_path).stem
            clean_stem = re.sub(r"^(?:arXiv[-_])?[0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?[-_]?", "", stem)
            clean_title = clean_stem.replace("_", " ").replace("-", " ").strip()
            if clean_title:
                return clean_title

        if self.paper_id:
            return self.paper_id.replace("_", " ").replace("-", " ").strip()

        return "Untitled Paper"

    def _extract_authors(self) -> list[str]:
        """Extract candidate author names from preamble lines."""
        preamble = self._get_preamble_lines()
        authors: list[str] = []
        for line in preamble[1:]:
            stripped = line.strip()
            lower = stripped.lower()
            if any(k in lower for k in [
                "@", "http", "university", "department", "lab", "institute",
                "center", "school", "college", "corporation", "fair", "google",
                "deepmind", "research", "keywords", "china", "usa"
            ]):
                continue
            if "," in stripped or " and " in stripped:
                cleaned_line = re.sub(r"[†*∗‡§\d]", "", stripped)
                cleaned_line = re.sub(r"\s+[a-f](?:,[a-f])*(?=[,\s]|$)", "", cleaned_line)
                parts = re.split(r"[,;]|\band\b", cleaned_line)
                for part in parts:
                    name = part.strip()
                    if name and 1 < len(name.split()) <= 4 and name[0].isupper():
                        authors.append(name)
                if authors:
                    break
        return authors

    def _extract_year(self) -> int | None:
        """Extract publication year from path, paper_id, or preamble."""
        path_str = f"{self.source_path} {self.paper_id}"
        m = re.search(r"(?:[^0-9]|^)(19[89][0-9]|20[0-2][0-9])(?:[^0-9]|$)", path_str)
        if m:
            return int(m.group(1))

        arxiv_m = re.search(r"(?:^|[^0-9])([0-2][0-9])(?:0[1-9]|1[0-2])\.[0-9]{4,5}", path_str)
        if arxiv_m:
            return 2000 + int(arxiv_m.group(1))

        for line in self._get_preamble_lines():
            ym = re.search(r"(?:[^0-9]|^)(19[89][0-9]|20[0-2][0-9])(?:[^0-9]|$)", line)
            if ym:
                return int(ym.group(1))

        return None


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

        # Step 1: Collect non-empty lines with their corresponding page and line number on that page
        # Each item: (page_number, line_on_page, line_text)
        all_lines: list[tuple[int, int, str]] = []
        for page in pages:
            if page.is_empty:
                continue
            for line_idx, line in enumerate(page.text.splitlines(), start=1):
                stripped = line.strip()
                if stripped:
                    all_lines.append((page.page_number, line_idx, stripped))

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
        # Each entry: (all_lines_index, page_number, line_on_page, heading_title)
        heading_indices: list[tuple[int, int, int, str]] = []
        for idx, (p_num, line_on_page, line) in enumerate(all_lines):
            if _is_candidate_heading(line):
                title = _clean_heading_title(line)
                # Avoid consecutive duplicate heading lines
                if not heading_indices or heading_indices[-1][3] != title:
                    heading_indices.append((idx, p_num, line_on_page, title))

        # Fallback: if no headings are detected across the paper, return single 'Unknown' section
        if not heading_indices:
            combined_text = "\n".join(line for _, _, line in all_lines)
            fallback_block = SectionBlock(
                title="Unknown",
                page_start=min_page,
                page_end=max_page,
                line_start=1,
                pages=all_page_nums,
                text=combined_text,
                lines=[(p, line) for p, _, line in all_lines],
            )
            return StructuredDocument(
                paper_id=paper_id,
                source_path=source_path,
                sections=[fallback_block.to_section()],
                section_blocks=[fallback_block],
            )

        # Step 3: Build disjoint section blocks using line slices
        blocks: list[SectionBlock] = []

        first_h_idx, first_h_page, _, _ = heading_indices[0]
        preamble_lines: list[str] = [line for _, _, line in all_lines[:first_h_idx]] if first_h_idx > 0 else []

        # If there is preamble before the first heading:
        # - If first heading is after page 1, emit a 'Header' block for preceding pages
        # - If first heading is on page 1, attach the preamble to that first section (e.g. Abstract)
        if first_h_idx > 0 and first_h_page > 1:
            preamble_pages = sorted(set(p for p, _, _ in all_lines[:first_h_idx]))
            blocks.append(
                SectionBlock(
                    title="Header",
                    page_start=min(preamble_pages),
                    page_end=max(preamble_pages),
                    line_start=1,
                    pages=preamble_pages,
                    text="\n".join(preamble_lines),
                    lines=[(p, line) for p, _, line in all_lines[:first_h_idx]],
                )
            )

        for i, (h_idx, h_page, h_line, title) in enumerate(heading_indices):
            # Line slice for this section up to the next heading
            if i + 1 < len(heading_indices):
                next_h_idx = heading_indices[i + 1][0]
                sec_line_tuples = all_lines[h_idx:next_h_idx]
            else:
                sec_line_tuples = all_lines[h_idx:]

            sec_lines = [line for _, _, line in sec_line_tuples]
            sec_pages = sorted(set(p for p, _, _ in sec_line_tuples))

            # Prepend page 1 preamble (e.g. title/authors) to the first heading on page 1
            if i == 0 and first_h_idx > 0 and first_h_page == 1:
                sec_lines = preamble_lines + sec_lines
                sec_pages = sorted(set(sec_pages) | set(p for p, _, _ in all_lines[:first_h_idx]))
                sec_all_tuples = all_lines[:first_h_idx] + sec_line_tuples
            else:
                sec_all_tuples = sec_line_tuples

            p_start = min(sec_pages) if sec_pages else h_page
            # The last section extends to document end
            p_end = max_page if i == len(heading_indices) - 1 else (max(sec_pages) if sec_pages else p_start)

            blocks.append(
                SectionBlock(
                    title=title,
                    page_start=p_start,
                    page_end=max(p_start, p_end),
                    line_start=h_line,
                    pages=sec_pages,
                    text="\n".join(sec_lines),
                    lines=[(p, line) for p, _, line in sec_all_tuples],
                )
            )

        sections = [b.to_section() for b in blocks]

        return StructuredDocument(
            paper_id=paper_id,
            source_path=source_path,
            sections=sections,
            section_blocks=blocks,
            preamble_lines=preamble_lines,
        )


def extract_sections(pages: Sequence[ParsedPage]) -> list[Section]:
    """Convenience function to extract a list of Section models from parsed pages."""
    return DocumentStructureExtractor().extract(pages).sections


def extract_document_structure(pages: Sequence[ParsedPage]) -> StructuredDocument:
    """Convenience function to extract a StructuredDocument from parsed pages."""
    return DocumentStructureExtractor().extract(pages)


def extract_paper(pages: Sequence[ParsedPage]) -> Paper:
    """Convenience function to extract a Paper model directly from parsed pages."""
    return DocumentStructureExtractor().extract(pages).to_paper()
