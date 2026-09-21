"""Base retriever abstraction for ResearchPilot (Step 0.7).

Defines the core interface/contract for paper indexing, chunk indexing,
paper search, and chunk evidence retrieval before implementing specific vector backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from app.models.chunk import Chunk, Evidence
from app.models.paper import Paper, ScoredPaper


class Retriever(ABC):
    """Abstract base class defining the retrieval contract for ResearchPilot."""

    @abstractmethod
    def index_papers(self, papers: Sequence[Paper]) -> None:
        """Index a collection of research papers for paper-level discovery.

        Args:
            papers: Sequence of Paper domain models containing metadata, title, and abstract.
        """
        ...

    @abstractmethod
    def index_chunks(self, chunks: Sequence[Chunk]) -> None:
        """Index a collection of text chunks for evidence retrieval.

        Args:
            chunks: Sequence of Chunk models with text, section, and page provenance.
        """
        ...

    @abstractmethod
    def search_papers(self, query: str, k: int = 5) -> list[ScoredPaper]:
        """Search the indexed papers by semantic similarity to a query.

        Args:
            query: Search query or technical research question.
            k: Maximum number of top relevant papers to return.

        Returns:
            List of ScoredPaper models ranked by relevance score in descending order.
        """
        ...

    @abstractmethod
    def retrieve_evidence(
        self,
        query: str,
        paper_ids: Sequence[str] | None = None,
        k: int = 5,
    ) -> list[Evidence]:
        """Retrieve fine-grained evidence chunks matching a query.

        Args:
            query: Search query or factual question.
            paper_ids: Optional collection of paper IDs to restrict search to.
                If None or empty, searches across all indexed chunks.
            k: Maximum number of evidence items to return.

        Returns:
            List of Evidence models ranked by relevance score in descending order.
        """
        ...

    @abstractmethod
    def get_paper(self, paper_id: str) -> Paper | None:
        """Retrieve paper metadata and structural sections by paper ID.

        Args:
            paper_id: Unique paper identifier.

        Returns:
            The Paper domain model if found, else None.
        """
        ...
