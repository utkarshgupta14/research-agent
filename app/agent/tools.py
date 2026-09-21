"""ResearchPilot MVP 0 agent tools (Step 0.12).

Provides the three research tools:
1. search_papers: Paper-level discovery with metadata, summaries, and scores.
2. retrieve_evidence: Fine-grained evidence retrieval with stable source IDs.
3. get_paper: Detailed inspection of paper metadata, abstract, and sections.

All tools interact strictly with the Retriever abstraction, decoupling the agent
from the underlying FAISS or vector store implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from langchain_core.tools import BaseTool, tool

from app.config import INDEX_DIR
from app.retrieval.base import Retriever
from app.retrieval.faiss_retriever import FAISSRetriever


_default_retriever: Retriever | None = None


def get_default_retriever() -> Retriever:
    """Get the active default retriever instance.

    If not explicitly set via set_default_retriever, loads the FAISSRetriever
    from the default INDEX_DIR if present, or an unindexed FAISSRetriever.
    """
    global _default_retriever
    if _default_retriever is None:
        retriever = FAISSRetriever(index_dir=INDEX_DIR)
        if Path(INDEX_DIR).exists():
            try:
                retriever.load(INDEX_DIR)
            except Exception:
                pass
        _default_retriever = retriever
    return _default_retriever


def set_default_retriever(retriever: Retriever) -> None:
    """Set the default retriever instance for research tools."""
    global _default_retriever
    _default_retriever = retriever


def reset_default_retriever() -> None:
    """Reset the default retriever instance to None (useful for testing)."""
    global _default_retriever
    _default_retriever = None


# ==============================================================================
# Tool Implementations
# ==============================================================================

@tool
def search_papers(query: str) -> list[dict[str, Any]]:
    """Find relevant research papers for a technical query or question.

    Args:
        query: Search query or technical topic.

    Returns:
        List of compact paper summaries including paper_id, title, authors, year,
        abstract or short summary, and relevance score.
    """
    retriever = get_default_retriever()
    scored_papers = retriever.search_papers(query=query)

    results: list[dict[str, Any]] = []
    for p in scored_papers:
        # Keep abstract compact for LLM context efficiency
        abstract_snippet = (
            p.abstract[:350] + "..." if len(p.abstract) > 350 else p.abstract
        )
        results.append(
            {
                "paper_id": p.paper_id,
                "title": p.title,
                "authors": p.authors,
                "year": p.year,
                "abstract": abstract_snippet,
                "score": p.score,
            }
        )
    return results


@tool
def retrieve_evidence(
    query: str,
    paper_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Retrieve fine-grained evidence chunks matching a query, optionally filtered by paper IDs.

    Args:
        query: Search query or factual question to find supporting evidence for.
        paper_ids: Optional list of paper IDs to restrict candidate chunks to.

    Returns:
        List of evidence items with stable source IDs (e.g. 'S1', 'S2'), paper_id,
        page number, section title, text excerpt, and relevance score.
    """
    retriever = get_default_retriever()
    evidence_items = retriever.retrieve_evidence(query=query, paper_ids=paper_ids)

    results: list[dict[str, Any]] = []
    for i, e in enumerate(evidence_items):
        source_id = e.source_id if e.source_id else f"S{i + 1}"
        results.append(
            {
                "source_id": source_id,
                "paper_id": e.paper_id,
                "page": e.page,
                "section": e.section,
                "text": e.text,
                "score": e.score,
            }
        )
    return results


@tool
def get_paper(paper_id: str) -> dict[str, Any]:
    """Get metadata, full abstract, and section outline for a specific paper by its ID.

    Args:
        paper_id: Unique identifier of the paper.

    Returns:
        Dictionary containing paper metadata, abstract, and basic section information,
        or an error message if the paper is not found.
    """
    retriever = get_default_retriever()
    paper = retriever.get_paper(paper_id=paper_id)
    if paper is None:
        return {"error": f"Paper '{paper_id}' not found."}

    return {
        "paper_id": paper.paper_id,
        "title": paper.title,
        "authors": paper.authors,
        "year": paper.year,
        "abstract": paper.abstract,
        "sections": [
            {
                "title": s.title,
                "page_start": s.page_start,
                "page_end": s.page_end,
            }
            for s in paper.sections
        ],
    }


# ==============================================================================
# Tool Factory
# ==============================================================================

def create_research_tools(retriever: Retriever | None = None) -> list[BaseTool]:
    """Create research tools bound to a specific or default retriever instance.

    Args:
        retriever: Optional Retriever instance. If provided, creates tools bound
            to this retriever. Otherwise, uses the default retriever.

    Returns:
        List of the three research tools [search_papers, retrieve_evidence, get_paper].
    """
    if retriever is None:
        return [search_papers, retrieve_evidence, get_paper]

    target = retriever

    @tool
    def search_papers_bound(query: str) -> list[dict[str, Any]]:
        """Find relevant research papers for a technical query or question."""
        scored_papers = target.search_papers(query=query)
        results: list[dict[str, Any]] = []
        for p in scored_papers:
            abstract_snippet = (
                p.abstract[:350] + "..." if len(p.abstract) > 350 else p.abstract
            )
            results.append(
                {
                    "paper_id": p.paper_id,
                    "title": p.title,
                    "authors": p.authors,
                    "year": p.year,
                    "abstract": abstract_snippet,
                    "score": p.score,
                }
            )
        return results

    search_papers_bound.name = "search_papers"

    @tool
    def retrieve_evidence_bound(
        query: str,
        paper_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve fine-grained evidence chunks matching a query, optionally filtered by paper IDs."""
        evidence_items = target.retrieve_evidence(query=query, paper_ids=paper_ids)
        results: list[dict[str, Any]] = []
        for i, e in enumerate(evidence_items):
            source_id = e.source_id if e.source_id else f"S{i + 1}"
            results.append(
                {
                    "source_id": source_id,
                    "paper_id": e.paper_id,
                    "page": e.page,
                    "section": e.section,
                    "text": e.text,
                    "score": e.score,
                }
            )
        return results

    retrieve_evidence_bound.name = "retrieve_evidence"

    @tool
    def get_paper_bound(paper_id: str) -> dict[str, Any]:
        """Get metadata, full abstract, and section outline for a specific paper by its ID."""
        paper = target.get_paper(paper_id=paper_id)
        if paper is None:
            return {"error": f"Paper '{paper_id}' not found."}

        return {
            "paper_id": paper.paper_id,
            "title": paper.title,
            "authors": paper.authors,
            "year": paper.year,
            "abstract": paper.abstract,
            "sections": [
                {
                    "title": s.title,
                    "page_start": s.page_start,
                    "page_end": s.page_end,
                }
                for s in paper.sections
            ],
        }

    get_paper_bound.name = "get_paper"

    return [search_papers_bound, retrieve_evidence_bound, get_paper_bound]
