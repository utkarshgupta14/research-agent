"""Tests for the Retriever abstraction in ResearchPilot (Step 0.7).

Verifies the interface contract using an in-memory FakeRetriever implementation,
ensuring domain model decoupling without depending on FAISS.
"""

from __future__ import annotations

from typing import Sequence
import pytest

from app.models.chunk import Chunk, Evidence
from app.models.paper import Paper, ScoredPaper
from app.retrieval.base import Retriever


class FakeRetriever(Retriever):
    """In-memory mock retriever used to verify the Retriever interface contract."""

    def __init__(self) -> None:
        self._papers: dict[str, Paper] = {}
        self._chunks: list[Chunk] = []

    def index_papers(self, papers: Sequence[Paper]) -> None:
        for paper in papers:
            self._papers[paper.paper_id] = paper

    def index_chunks(self, chunks: Sequence[Chunk]) -> None:
        self._chunks.extend(chunks)

    def search_papers(self, query: str, k: int = 5) -> list[ScoredPaper]:
        if not query.strip() or k <= 0 or not self._papers:
            return []

        query_terms = set(query.lower().split())
        scored: list[ScoredPaper] = []

        for paper in self._papers.values():
            corpus_text = f"{paper.title} {paper.abstract}".lower()
            matches = sum(1 for term in query_terms if term in corpus_text)
            if matches > 0:
                score = matches / len(query_terms)
                scored.append(
                    ScoredPaper(
                        paper_id=paper.paper_id,
                        title=paper.title,
                        authors=paper.authors,
                        year=paper.year,
                        abstract=paper.abstract,
                        source_path=paper.source_path,
                        sections=paper.sections,
                        score=round(score, 4),
                    )
                )

        scored.sort(key=lambda p: p.score or 0.0, reverse=True)
        return scored[:k]

    def retrieve_evidence(
        self,
        query: str,
        paper_ids: Sequence[str] | None = None,
        k: int = 5,
    ) -> list[Evidence]:
        if not query.strip() or k <= 0 or not self._chunks:
            return []

        allowed_papers = set(paper_ids) if paper_ids is not None else None
        query_terms = set(query.lower().split())
        matches: list[tuple[float, Chunk]] = []

        for chunk in self._chunks:
            if allowed_papers is not None and chunk.paper_id not in allowed_papers:
                continue

            chunk_text = chunk.text.lower()
            term_matches = sum(1 for term in query_terms if term in chunk_text)
            if term_matches > 0:
                score = term_matches / len(query_terms)
                matches.append((score, chunk))

        matches.sort(key=lambda item: item[0], reverse=True)
        top_matches = matches[:k]

        results: list[Evidence] = []
        for i, (score, chunk) in enumerate(top_matches):
            results.append(
                Evidence(
                    source_id=f"S{i+1}",
                    chunk_id=chunk.chunk_id,
                    paper_id=chunk.paper_id,
                    page=chunk.page,
                    section=chunk.section,
                    text=chunk.text,
                    score=round(score, 4),
                )
            )
        return results

    def get_paper(self, paper_id: str) -> Paper | None:
        return self._papers.get(paper_id)


# ==============================================================================
# Unit Tests
# ==============================================================================

def test_cannot_instantiate_abstract_retriever():
    """Verify that Retriever is an abstract class and cannot be directly instantiated."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        Retriever()  # type: ignore[abstract]


def test_fake_retriever_index_and_search_papers():
    """Verify indexing papers and searching returns ranked ScoredPaper instances."""
    retriever = FakeRetriever()
    papers = [
        Paper(
            paper_id="paper_001",
            title="Attention Is All You Need",
            authors=["Vaswani et al."],
            year=2017,
            abstract="We propose the Transformer architecture based on self-attention.",
        ),
        Paper(
            paper_id="paper_002",
            title="Deep Residual Learning for Image Recognition",
            authors=["He et al."],
            year=2016,
            abstract="Residual networks ease the training of substantially deeper neural networks.",
        ),
    ]

    retriever.index_papers(papers)

    results = retriever.search_papers("transformer self-attention", k=5)

    assert len(results) == 1
    top_paper = results[0]

    # Verify model types and attributes
    assert isinstance(top_paper, ScoredPaper)
    assert isinstance(top_paper, Paper)
    assert top_paper.paper_id == "paper_001"
    assert top_paper.title == "Attention Is All You Need"
    assert top_paper.score is not None
    assert top_paper.score > 0.0


def test_fake_retriever_index_and_retrieve_evidence():
    """Verify indexing chunks and retrieving evidence returns ranked Evidence instances."""
    retriever = FakeRetriever()
    chunks = [
        Chunk(
            chunk_id="chunk_001",
            paper_id="paper_001",
            page=3,
            section="3.2 Attention",
            text="An attention function can be described as mapping a query and a set of key-value pairs.",
        ),
        Chunk(
            chunk_id="chunk_002",
            paper_id="paper_001",
            page=5,
            section="4.1 Multi-Head Attention",
            text="Multi-head attention allows the model to jointly attend to information from different representation subspaces.",
        ),
        Chunk(
            chunk_id="chunk_003",
            paper_id="paper_002",
            page=2,
            section="3 Deep Residual Learning",
            text="We introduce residual mapping formulations for very deep CNNs.",
        ),
    ]

    retriever.index_chunks(chunks)

    evidence_list = retriever.retrieve_evidence("attention function query", k=2)

    assert len(evidence_list) == 2
    for i, evidence in enumerate(evidence_list):
        assert isinstance(evidence, Evidence)
        assert isinstance(evidence, Chunk)
        assert evidence.source_id == f"S{i+1}"
        assert evidence.score is not None
        assert evidence.score > 0.0
        assert evidence.paper_id == "paper_001"

    assert evidence_list[0].chunk_id == "chunk_001"


def test_fake_retriever_evidence_filtering_by_paper_ids():
    """Verify that retrieve_evidence respects paper_ids filter."""
    retriever = FakeRetriever()
    chunks = [
        Chunk(
            chunk_id="chunk_a",
            paper_id="paper_001",
            page=1,
            section="Intro",
            text="Deep networks perform well on vision benchmarks.",
        ),
        Chunk(
            chunk_id="chunk_b",
            paper_id="paper_002",
            page=1,
            section="Intro",
            text="Deep networks with residual shortcuts solve degradation.",
        ),
    ]

    retriever.index_chunks(chunks)

    # Search query matches both, but restricted to paper_002
    results = retriever.retrieve_evidence("deep networks", paper_ids=["paper_002"], k=5)

    assert len(results) == 1
    assert results[0].paper_id == "paper_002"
    assert results[0].chunk_id == "chunk_b"


def test_fake_retriever_empty_handling():
    """Verify graceful handling of empty queries and empty indexes."""
    retriever = FakeRetriever()

    # Empty index
    assert retriever.search_papers("anything") == []
    assert retriever.retrieve_evidence("anything") == []

    # Non-empty index, empty/whitespace query
    retriever.index_papers([Paper(paper_id="p1", title="Title", abstract="Abstract")])
    retriever.index_chunks([Chunk(chunk_id="c1", paper_id="p1", page=1, section="S", text="T")])

    assert retriever.search_papers("") == []
    assert retriever.search_papers("   ") == []
    assert retriever.retrieve_evidence("") == []
    assert retriever.retrieve_evidence("   ") == []

    # k <= 0
    assert retriever.search_papers("title", k=0) == []
    assert retriever.retrieve_evidence("text", k=0) == []


def test_fake_retriever_get_paper():
    """Verify get_paper returns the paper model or None if missing."""
    retriever = FakeRetriever()
    paper = Paper(
        paper_id="paper_001",
        title="Attention Is All You Need",
        authors=["Vaswani et al."],
        year=2017,
        abstract="The dominant sequence transduction models...",
    )
    retriever.index_papers([paper])

    assert retriever.get_paper("paper_001") == paper
    assert retriever.get_paper("non_existent") is None
