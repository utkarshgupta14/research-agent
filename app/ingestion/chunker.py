"""Document chunking module for ResearchPilot MVP 0 (Step 0.5).

Produces deterministic, searchable Chunk objects from structured document representations
while strictly preserving paper_id, page, and section provenance.
"""

from __future__ import annotations

import re
from typing import Sequence
from pydantic import BaseModel, Field

from app.ingestion.structure import SectionBlock, StructuredDocument
from app.models.chunk import Chunk


def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken (cl100k_base) with word-based heuristic fallback."""
    cleaned = text.strip()
    if not cleaned:
        return 0
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(cleaned, disallowed_special=()))
    except Exception:
        # Fallback estimation: 1 word ~ 1.3 tokens
        return max(1, int(len(cleaned.split()) * 1.3))


class ChunkerConfig(BaseModel):
    """Configuration options for text chunking."""

    chunk_size: int = Field(
        default=600,
        description="Target maximum chunk size in tokens (typical: 500-800)",
    )
    chunk_overlap: int = Field(
        default=100,
        description="Target overlap between consecutive chunks in tokens",
    )
    min_chunk_chars: int = Field(
        default=40,
        description="Minimum character count to emit a valid chunk (avoids empty noise)",
    )


class TextChunker:
    """Configurable, deterministic chunker for structured research papers."""

    def __init__(self, config: ChunkerConfig | None = None) -> None:
        self.config = config or ChunkerConfig()

    def chunk_document(self, doc: StructuredDocument) -> list[Chunk]:
        """Convert a StructuredDocument into a list of deterministic Chunk objects.

        Args:
            doc: StructuredDocument produced by DocumentStructureExtractor.

        Returns:
            A list of Chunk objects retaining paper, page, section, and text provenance.
        """
        if not doc.section_blocks:
            return []

        chunks: list[Chunk] = []
        chunk_idx = 1

        for block in doc.section_blocks:
            block_chunks = self._chunk_section_block(
                block=block,
                paper_id=doc.paper_id,
                start_index=chunk_idx,
            )
            chunks.extend(block_chunks)
            chunk_idx += len(block_chunks)

        return chunks

    def _chunk_section_block(
        self,
        block: SectionBlock,
        paper_id: str,
        start_index: int,
    ) -> list[Chunk]:
        """Slice a single SectionBlock into discrete overlapping chunks."""
        text = block.text.strip()
        if len(text) < self.config.min_chunk_chars:
            return []

        # Extract atomic units (paragraphs or line groups) annotated with page numbers
        units = self._extract_atomic_units(block)
        if not units:
            return []

        chunks: list[Chunk] = []
        current_idx = start_index

        current_units: list[tuple[int, str]] = []
        current_tokens = 0

        for page_num, unit_text in units:
            unit_tokens = count_tokens(unit_text)
            if not unit_tokens:
                continue

            # If adding unit_text exceeds chunk_size and we already have content:
            if current_units and (current_tokens + unit_tokens > self.config.chunk_size):
                # Emit current chunk
                chunk = self._create_chunk(
                    paper_id=paper_id,
                    index=current_idx,
                    section=block.title,
                    items=current_units,
                )
                if chunk is not None:
                    chunks.append(chunk)
                    current_idx += 1

                # Calculate overlap subset from trailing units
                current_units, current_tokens = self._calculate_overlap(current_units)

            current_units.append((page_num, unit_text))
            current_tokens += unit_tokens

        # Flush final chunk for this section
        if current_units:
            chunk = self._create_chunk(
                paper_id=paper_id,
                index=current_idx,
                section=block.title,
                items=current_units,
            )
            if chunk is not None:
                chunks.append(chunk)

        return chunks

    def _extract_atomic_units(self, block: SectionBlock) -> list[tuple[int, str]]:
        """Deconstruct a SectionBlock into atomic paragraphs/sentences with page numbers."""
        units: list[tuple[int, str]] = []

        if block.lines:
            # Group contiguous lines by page and paragraph breaks
            curr_page = block.lines[0][0]
            curr_lines: list[str] = []

            for p_num, line in block.lines:
                # If page changes or an empty-line paragraph boundary was detected
                if p_num != curr_page and curr_lines:
                    text_blob = "\n".join(curr_lines).strip()
                    if text_blob:
                        units.extend(self._split_into_paragraphs_or_sentences(curr_page, text_blob))
                    curr_lines = []
                    curr_page = p_num

                curr_lines.append(line)

            if curr_lines:
                text_blob = "\n".join(curr_lines).strip()
                if text_blob:
                    units.extend(self._split_into_paragraphs_or_sentences(curr_page, text_blob))
        else:
            # Fallback if line annotations were not supplied
            fallback_page = block.page_start
            units.extend(self._split_into_paragraphs_or_sentences(fallback_page, block.text))

        return units

    def _split_into_paragraphs_or_sentences(
        self,
        page_num: int,
        text: str,
    ) -> list[tuple[int, str]]:
        """Split text into sentence-level atomic units under chunk_size."""
        result: list[tuple[int, str]] = []
        raw_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        for para in raw_paragraphs:
            # Sub-split paragraph into individual sentences
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", para) if s.strip()]
            for sentence in sentences:
                sent_tokens = count_tokens(sentence)
                if sent_tokens <= self.config.chunk_size:
                    result.append((page_num, sentence))
                else:
                    # Sub-split oversized sentences by word chunks
                    words = sentence.split()
                    curr_words: list[str] = []
                    for word in words:
                        curr_words.append(word)
                        if count_tokens(" ".join(curr_words)) >= self.config.chunk_size:
                            result.append((page_num, " ".join(curr_words)))
                            curr_words = []
                    if curr_words:
                        result.append((page_num, " ".join(curr_words)))

        return result

    def _calculate_overlap(
        self,
        items: list[tuple[int, str]],
    ) -> tuple[list[tuple[int, str]], int]:
        """Compute the tail subset of units to carry over into the next chunk for overlap."""
        if self.config.chunk_overlap <= 0 or not items:
            return [], 0

        overlap_items: list[tuple[int, str]] = []
        accumulated_tokens = 0

        for page_num, text in reversed(items):
            toks = count_tokens(text)
            if accumulated_tokens + toks <= self.config.chunk_overlap or not overlap_items:
                overlap_items.insert(0, (page_num, text))
                accumulated_tokens += toks
            else:
                break

        return overlap_items, accumulated_tokens

    def _create_chunk(
        self,
        paper_id: str,
        index: int,
        section: str,
        items: list[tuple[int, str]],
    ) -> Chunk | None:
        """Create a Chunk object with deterministic ID and provenance metadata."""
        if not items:
            return None

        combined_text = " ".join(t for _, t in items).strip()
        if len(combined_text) < self.config.min_chunk_chars:
            return None

        # Chunk page attribution is the starting page of the chunk's content
        chunk_page = items[0][0]
        chunk_id = f"{paper_id}_c{index:04d}"

        return Chunk(
            chunk_id=chunk_id,
            paper_id=paper_id,
            page=chunk_page,
            section=section,
            text=combined_text,
        )


def chunk_document(
    doc: StructuredDocument,
    config: ChunkerConfig | None = None,
) -> list[Chunk]:
    """Convenience function to chunk a StructuredDocument using default or custom config."""
    return TextChunker(config=config).chunk_document(doc)
