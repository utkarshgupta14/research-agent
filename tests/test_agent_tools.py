"""Unit tests for ResearchPilot MVP 0 agent tools (Step 0.12).

Verifies:
- search_papers tool: paper-level discovery, compact summaries, relevance scores.
- retrieve_evidence tool: chunk-level evidence retrieval, stable source IDs, paper_ids filter.
- get_paper tool: metadata, full abstract, and section outline; error handling for missing paper.
- Decoupling: tools interact strictly with Retriever abstraction without exposing internal FAISS objects.
- create_research_tools factory with custom and default retrievers.
"""

from __future__ import annotations

from typing import Sequence
import pytest

from app.agent.tools import (
    create_research_tools,
    get_paper,
    reset_default_retriever,
    retrieve_evidence,
    search_papers,
    set_default_retriever,
)
from app.models.chunk import Chunk, Evidence
from app.models.paper import Paper, ScoredPaper, Section
from app.retrieval.base import Retriever


class FakeRetriever(Retriever):
    """In-memory mock retriever implementing the Retriever abstraction without FAISS."""

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
            text = f"{paper.title} {paper.abstract}".lower()
            matches = sum(1 for t in query_terms if t in text)
            if matches > 0:
                scored.append(
                    ScoredPaper(
                        **paper.model_dump(),
                        score=round(matches / len(query_terms), 4),
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

        allowed = set(paper_ids) if paper_ids is not None else None
        query_terms = set(query.lower().split())
        matches: list[tuple[float, Chunk]] = []

        for chunk in self._chunks:
            if allowed is not None and chunk.paper_id not in allowed:
                continue
            text = chunk.text.lower()
            m = sum(1 for t in query_terms if t in text)
            if m > 0:
                matches.append((m / len(query_terms), chunk))

        matches.sort(key=lambda item: item[0], reverse=True)
        results: list[Evidence] = []
        for i, (score, chunk) in enumerate(matches[:k]):
            results.append(
                Evidence(
                    source_id=f"S{i + 1}",
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


@pytest.fixture
def populated_retriever() -> FakeRetriever:
    """Fixture providing a FakeRetriever populated with sample papers and chunks."""
    retriever = FakeRetriever()

    papers = [
        Paper(
            paper_id="videomae_2022",
            title="VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training",
            authors=["Zhan Tong", "Yibo Liming", "Kunchang Li", "Jiashuo Wang"],
            year=2022,
            abstract=(
                "We show that video masked autoencoders (VideoMAE) are data-efficient learners "
                "for self-supervised video pre-training. Inspired by the recent ImageMAE, we propose "
                "custom tube masking with an extremely high masking ratio (90% to 95%). This simple "
                "design makes video reconstruction challenging and forces the model to capture "
                "spatiotemporal structure without inductive bias."
            ),
            sections=[
                Section(title="Abstract", page_start=1, page_end=1),
                Section(title="Introduction", page_start=1, page_end=2),
                Section(title="Methodology", page_start=2, page_end=5),
                Section(title="Experiments", page_start=5, page_end=8),
            ],
        ),
        Paper(
            paper_id="slowfast_2019",
            title="SlowFast Networks for Video Recognition",
            authors=["Christoph Feichtenhofer", "Haoqi Fan", "Jitendra Malik", "Kaiming He"],
            year=2019,
            abstract=(
                "We present SlowFast networks for video recognition. Our model involves (i) a Slow pathway, "
                "operating at low frame rate, to capture spatial semantics, and (ii) a Fast pathway, "
                "operating at high frame rate, to capture motion at fine temporal resolution."
            ),
            sections=[
                Section(title="Introduction", page_start=1, page_end=2),
                Section(title="SlowFast Networks", page_start=2, page_end=5),
            ],
        ),
    ]

    chunks = [
        Chunk(
            chunk_id="vmae_c01",
            paper_id="videomae_2022",
            page=2,
            section="Methodology",
            text="VideoMAE uses a high masking ratio of 90% to 95% with tube masking.",
        ),
        Chunk(
            chunk_id="vmae_c02",
            paper_id="videomae_2022",
            page=3,
            section="Methodology",
            text="The encoder operates only on visible patches, yielding high computation efficiency.",
        ),
        Chunk(
            chunk_id="sf_c01",
            paper_id="slowfast_2019",
            page=2,
            section="SlowFast Networks",
            text="The Fast pathway has a small channel capacity but high temporal resolution.",
        ),
        Chunk(
            chunk_id="sf_c02",
            paper_id="slowfast_2019",
            page=3,
            section="SlowFast Networks",
            text="Lateral connections fuse information from the Fast pathway into the Slow pathway.",
        ),
    ]

    retriever.index_papers(papers)
    retriever.index_chunks(chunks)
    return retriever


@pytest.fixture(autouse=True)
def setup_and_teardown(populated_retriever: FakeRetriever):
    """Ensure the populated FakeRetriever is set as default for each test and reset afterward."""
    set_default_retriever(populated_retriever)
    yield
    reset_default_retriever()


# ==============================================================================
# search_papers Tests
# ==============================================================================

def test_search_papers_returns_expected_fields():
    """Verify search_papers returns paper_id, title, authors, year, abstract, and score."""
    results = search_papers.invoke({"query": "VideoMAE masked autoencoders"})

    assert len(results) >= 1
    top = results[0]

    assert top["paper_id"] == "videomae_2022"
    assert "VideoMAE" in top["title"]
    assert isinstance(top["authors"], list)
    assert top["year"] == 2022
    assert isinstance(top["abstract"], str)
    assert top["score"] is not None
    assert top["score"] > 0


def test_search_papers_compact_abstract():
    """Verify that search_papers does not output gigantic abstracts that flood LLM context."""
    results = search_papers.invoke({"query": "VideoMAE"})

    for paper in results:
        # Abstracts in search_papers should be concise (capped at ~355 chars)
        assert len(paper["abstract"]) <= 355


def test_search_papers_empty_or_no_match():
    """Verify search_papers returns an empty list for queries that yield no matches."""
    results = search_papers.invoke({"query": "quantum gravity black holes"})
    assert results == []


# ==============================================================================
# retrieve_evidence Tests
# ==============================================================================

def test_retrieve_evidence_returns_expected_fields():
    """Verify retrieve_evidence returns source_id, paper_id, page, section, text, and score."""
    results = retrieve_evidence.invoke({"query": "masking ratio tube masking"})

    assert len(results) >= 1
    item = results[0]

    assert item["source_id"] == "S1"
    assert item["paper_id"] == "videomae_2022"
    assert item["page"] == 2
    assert item["section"] == "Methodology"
    assert "90% to 95%" in item["text"]
    assert item["score"] is not None


def test_retrieve_evidence_stable_source_ids():
    """Verify that evidence items are assigned stable sequential source IDs (S1, S2, ...)."""
    results = retrieve_evidence.invoke({"query": "pathway resolution"})

    assert len(results) >= 2
    source_ids = [item["source_id"] for item in results]
    assert source_ids == ["S1", "S2"]


def test_retrieve_evidence_filter_by_paper_ids():
    """Verify that retrieve_evidence respects paper_ids filter."""
    # Query matches both VideoMAE and SlowFast concepts, but restricted to slowfast_2019
    results = retrieve_evidence.invoke(
        {
            "query": "pathway temporal",
            "paper_ids": ["slowfast_2019"],
        }
    )

    assert len(results) >= 1
    for item in results:
        assert item["paper_id"] == "slowfast_2019"


def test_retrieve_evidence_no_match():
    """Verify retrieve_evidence returns an empty list when no chunks match."""
    results = retrieve_evidence.invoke({"query": "unrelated non-existent query terms"})
    assert results == []


# ==============================================================================
# get_paper Tests
# ==============================================================================

def test_get_paper_success():
    """Verify get_paper returns full metadata, abstract, and section list."""
    result = get_paper.invoke({"paper_id": "videomae_2022"})

    assert result["paper_id"] == "videomae_2022"
    assert "VideoMAE" in result["title"]
    assert result["year"] == 2022
    assert len(result["abstract"]) > 0
    assert "sections" in result
    assert len(result["sections"]) == 4

    section_titles = [s["title"] for s in result["sections"]]
    assert "Abstract" in section_titles
    assert "Methodology" in section_titles
    assert result["sections"][2]["page_start"] == 2
    assert result["sections"][2]["page_end"] == 5


def test_get_paper_not_found():
    """Verify get_paper returns an informative error dictionary when paper_id is not found."""
    result = get_paper.invoke({"paper_id": "non_existent_paper"})

    assert "error" in result
    assert "non_existent_paper" in result["error"]


# ==============================================================================
# Decoupling & Factory Tests
# ==============================================================================

def test_tools_do_not_expose_faiss_objects():
    """Verify tools do not leak any FAISS or raw vector objects."""
    search_res = search_papers.invoke({"query": "SlowFast"})
    for p in search_res:
        for k in p:
            assert not k.startswith("_")
            assert "faiss" not in k.lower()

    evidence_res = retrieve_evidence.invoke({"query": "pathway"})
    for e in evidence_res:
        for k in e:
            assert not k.startswith("_")
            assert "faiss" not in k.lower()

    paper_res = get_paper.invoke({"paper_id": "slowfast_2019"})
    for k in paper_res:
        assert not k.startswith("_")
        assert "faiss" not in k.lower()


def test_create_research_tools_bound():
    """Verify create_research_tools produces tools bound to a specific retriever."""
    custom_retriever = FakeRetriever()
    custom_paper = Paper(
        paper_id="custom_p1",
        title="Custom Architecture",
        authors=["Alice"],
        year=2024,
        abstract="Custom paper abstract.",
        sections=[Section(title="Intro", page_start=1, page_end=1)],
    )
    custom_retriever.index_papers([custom_paper])

    tools = create_research_tools(custom_retriever)
    assert len(tools) == 3

    tool_map = {t.name: t for t in tools}
    assert "search_papers" in tool_map
    assert "retrieve_evidence" in tool_map
    assert "get_paper" in tool_map

    res = tool_map["get_paper"].invoke({"paper_id": "custom_p1"})
    assert res["paper_id"] == "custom_p1"
    assert res["title"] == "Custom Architecture"
