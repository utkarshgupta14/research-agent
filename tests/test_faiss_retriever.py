"""Tests for the FAISSRetriever component in ResearchPilot (Step 0.8).

Includes:
- Offline unit tests with a deterministic synthetic EmbeddingProvider (zero token cost).
- Verifications for paper search, chunk evidence retrieval, paper_ids filtering,
  and save/load persistence round-trip.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence
import pytest

from app.models.chunk import Chunk, Evidence
from app.models.paper import Paper, ScoredPaper
from app.retrieval.base import Retriever
from app.retrieval.embeddings import EmbeddingProvider, _l2_normalize
from app.retrieval.faiss_retriever import FAISSRetriever


class SyntheticEmbeddingProvider(EmbeddingProvider):
    """Deterministic, offline embedding provider for fast unit tests.

    Maps vocabulary words to predefined orthogonal dimensions.
    Dimensionality: 6.
    Dim 0: 'transformer', 'attention'
    Dim 1: 'residual', 'resnet', 'cnn'
    Dim 2: 'video', 'action', 'spatiotemporal'
    Dim 3: 'tracking', 'benchmark'
    Dim 4: 'cookie', 'baking', 'recipe'
    Dim 5: bias / general
    """

    VOCAB: dict[str, int] = {
        "transformer": 0,
        "attention": 0,
        "residual": 1,
        "resnet": 1,
        "cnn": 1,
        "video": 2,
        "action": 2,
        "spatiotemporal": 2,
        "tracking": 3,
        "benchmark": 3,
        "cookie": 4,
        "baking": 4,
        "recipe": 4,
    }

    @property
    def dimension(self) -> int:
        return 6

    def _embed_text(self, text: str) -> list[float]:
        vec = [0.0] * self.dimension
        words = text.lower().split()
        for w in words:
            clean_w = w.strip(".,;:!?()[]\"'")
            if clean_w in self.VOCAB:
                vec[self.VOCAB[clean_w]] += 1.0
            else:
                vec[5] += 0.1

        # Fallback if no words matched
        if sum(vec) == 0.0:
            vec[5] = 1.0

        return _l2_normalize(vec)

    def embed_query(self, text: str) -> list[float]:
        return self._embed_text(text)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_text(t) for t in texts]


# ==============================================================================
# Unit Tests
# ==============================================================================

@pytest.fixture
def synthetic_provider() -> SyntheticEmbeddingProvider:
    return SyntheticEmbeddingProvider()


@pytest.fixture
def sample_papers() -> list[Paper]:
    return [
        Paper(
            paper_id="paper_transformer",
            title="Attention Is All You Need",
            authors=["Ashish Vaswani", "Noam Shazeer"],
            year=2017,
            abstract="The Transformer is an architecture based solely on attention mechanisms.",
        ),
        Paper(
            paper_id="paper_resnet",
            title="Deep Residual Learning for Image Recognition",
            authors=["Kaiming He", "Xiangyu Zhang"],
            year=2016,
            abstract="Residual neural network layers ease the training of very deep networks.",
        ),
        Paper(
            paper_id="paper_video",
            title="SlowFast Networks for Video Recognition",
            authors=["Christoph Feichtenhofer", "Haoqi Fan"],
            year=2019,
            abstract="We present SlowFast networks for video action recognition.",
        ),
    ]


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id="chunk_trans_1",
            paper_id="paper_transformer",
            page=3,
            section="3.2 Attention",
            text="An attention function can be described as mapping a query and keys to values.",
        ),
        Chunk(
            chunk_id="chunk_trans_2",
            paper_id="paper_transformer",
            page=4,
            section="3.3 Multi-Head Attention",
            text="Multi-head attention allows the model to attend to different representation subspaces.",
        ),
        Chunk(
            chunk_id="chunk_res_1",
            paper_id="paper_resnet",
            page=2,
            section="3 Residual Learning",
            text="We introduce residual mapping formulations using shortcut connections.",
        ),
        Chunk(
            chunk_id="chunk_vid_1",
            paper_id="paper_video",
            page=1,
            section="1 Introduction",
            text="Video action recognition requires processing spatial and temporal dimensions.",
        ),
    ]


def test_interface_compliance(synthetic_provider: SyntheticEmbeddingProvider):
    """Verify that FAISSRetriever implements the Retriever interface."""
    assert issubclass(FAISSRetriever, Retriever)
    retriever = FAISSRetriever(embedding_provider=synthetic_provider)
    assert isinstance(retriever, Retriever)


def test_index_and_search_papers(
    synthetic_provider: SyntheticEmbeddingProvider,
    sample_papers: list[Paper],
):
    """Verify paper-level indexing and semantic search ranking."""
    retriever = FAISSRetriever(embedding_provider=synthetic_provider)
    retriever.index_papers(sample_papers)

    results = retriever.search_papers("transformer attention mechanism", k=2)

    assert len(results) == 2
    top = results[0]

    assert isinstance(top, ScoredPaper)
    assert isinstance(top, Paper)
    assert top.paper_id == "paper_transformer"
    assert top.title == "Attention Is All You Need"
    assert top.score is not None
    assert top.score > 0.5


def test_index_and_retrieve_evidence(
    synthetic_provider: SyntheticEmbeddingProvider,
    sample_chunks: list[Chunk],
):
    """Verify chunk indexing and evidence retrieval with sequential source IDs."""
    retriever = FAISSRetriever(embedding_provider=synthetic_provider)
    retriever.index_chunks(sample_chunks)

    evidence_list = retriever.retrieve_evidence("attention query", k=2)

    assert len(evidence_list) == 2
    for i, evidence in enumerate(evidence_list):
        assert isinstance(evidence, Evidence)
        assert isinstance(evidence, Chunk)
        assert evidence.source_id == f"S{i+1}"
        assert evidence.score is not None
        assert evidence.score > 0.0
        assert evidence.paper_id == "paper_transformer"

    returned_chunk_ids = {ev.chunk_id for ev in evidence_list}
    assert returned_chunk_ids == {"chunk_trans_1", "chunk_trans_2"}


def test_retrieve_evidence_filtered_by_paper_ids(
    synthetic_provider: SyntheticEmbeddingProvider,
    sample_chunks: list[Chunk],
):
    """Verify evidence retrieval restricts candidates when paper_ids is provided."""
    retriever = FAISSRetriever(embedding_provider=synthetic_provider)
    retriever.index_chunks(sample_chunks)

    # Search for video action text, but restrict to paper_transformer
    results = retriever.retrieve_evidence(
        "video action",
        paper_ids=["paper_transformer"],
        k=5,
    )

    # Only chunks from paper_transformer should be returned
    for ev in results:
        assert ev.paper_id == "paper_transformer"

    # Restrict to paper_video
    vid_results = retriever.retrieve_evidence(
        "video action recognition",
        paper_ids=["paper_video"],
        k=5,
    )
    assert len(vid_results) == 1
    assert vid_results[0].paper_id == "paper_video"
    assert vid_results[0].chunk_id == "chunk_vid_1"

    # Empty paper_ids list returns empty
    assert retriever.retrieve_evidence("attention", paper_ids=[], k=5) == []


def test_persistence_save_and_load_roundtrip(
    synthetic_provider: SyntheticEmbeddingProvider,
    sample_papers: list[Paper],
    sample_chunks: list[Chunk],
    tmp_path: Path,
):
    """Verify saving indexes and metadata to disk and restoring them identically."""
    retriever = FAISSRetriever(embedding_provider=synthetic_provider, index_dir=tmp_path)
    retriever.index_papers(sample_papers)
    retriever.index_chunks(sample_chunks)

    # Save to tmp_path
    retriever.save(tmp_path)

    assert (tmp_path / "papers.index").exists()
    assert (tmp_path / "papers.json").exists()
    assert (tmp_path / "chunks.index").exists()
    assert (tmp_path / "chunks.json").exists()
    assert (tmp_path / "index_meta.json").exists()

    # Load into a fresh retriever via from_saved
    restored = FAISSRetriever.from_saved(tmp_path, embedding_provider=synthetic_provider)

    # Verify paper search parity
    original_papers = retriever.search_papers("residual shortcuts", k=2)
    restored_papers = restored.search_papers("residual shortcuts", k=2)

    assert len(original_papers) == len(restored_papers)
    for orig, rest in zip(original_papers, restored_papers):
        assert orig.paper_id == rest.paper_id
        assert orig.title == rest.title
        assert orig.score == rest.score

    # Verify chunk evidence parity
    original_evidence = retriever.retrieve_evidence("attention", k=2)
    restored_evidence = restored.retrieve_evidence("attention", k=2)

    assert len(original_evidence) == len(restored_evidence)
    for orig, rest in zip(original_evidence, restored_evidence):
        assert orig.chunk_id == rest.chunk_id
        assert orig.source_id == rest.source_id
        assert orig.score == rest.score


def test_empty_edge_cases(synthetic_provider: SyntheticEmbeddingProvider):
    """Verify empty queries, empty indexes, and boundary k values."""
    retriever = FAISSRetriever(embedding_provider=synthetic_provider)

    # Empty index
    assert retriever.search_papers("transformer") == []
    assert retriever.retrieve_evidence("attention") == []

    # Non-empty index, empty/whitespace queries
    retriever.index_papers([Paper(paper_id="p1", title="Title", abstract="Abstract")])
    retriever.index_chunks([Chunk(chunk_id="c1", paper_id="p1", page=1, section="S", text="Text")])

    assert retriever.search_papers("") == []
    assert retriever.search_papers("   ") == []
    assert retriever.retrieve_evidence("") == []
    assert retriever.retrieve_evidence("   ") == []

    # k <= 0
    assert retriever.search_papers("Title", k=0) == []
    assert retriever.search_papers("Title", k=-1) == []
    assert retriever.retrieve_evidence("Text", k=0) == []
    assert retriever.retrieve_evidence("Text", k=-1) == []

    # Empty lists in indexing
    retriever.index_papers([])
    assert retriever.search_papers("Title") == []
    retriever.index_chunks([])
    assert retriever.retrieve_evidence("Text") == []
