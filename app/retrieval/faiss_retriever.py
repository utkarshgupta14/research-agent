"""FAISS-based dense vector retriever implementation for ResearchPilot (Step 0.8).

Maintains two separate FAISS IndexFlatIP indexes:
1. Paper index: for discovery and semantic matching of whole papers (title + abstract).
2. Chunk index: for fine-grained retrieval of passage evidence with section/page provenance.

Persists FAISS binary index files and structured metadata (JSON) locally to disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence
import numpy as np
import faiss

from app.config import INDEX_DIR
from app.models.chunk import Chunk, Evidence
from app.models.paper import Paper, ScoredPaper
from app.retrieval.base import Retriever
from app.retrieval.embeddings import EmbeddingProvider, get_embedding_provider


class FAISSRetriever(Retriever):
    """FAISS-backed retriever implementing the Retriever interface.

    Uses inner-product flat indexing (IndexFlatIP) over L2-normalized embeddings,
    giving exact cosine similarity scores in the [-1.0, 1.0] range.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider | None = None,
        index_dir: Path | str | None = None,
    ) -> None:
        """Initialize FAISSRetriever.

        Args:
            embedding_provider: EmbeddingProvider instance. Defaults to configured provider.
            index_dir: Directory where FAISS binary indexes and metadata JSON files are saved/loaded.
        """
        self.embedding_provider = embedding_provider or get_embedding_provider()
        self.index_dir = Path(index_dir or INDEX_DIR)

        self._paper_index: faiss.Index | None = None
        self._papers: list[Paper] = []

        self._chunk_index: faiss.Index | None = None
        self._chunks: list[Chunk] = []

    @property
    def dimension(self) -> int:
        """Dimensionality of the dense vectors produced by the embedding provider."""
        return self.embedding_provider.dimension

    # ==========================================================================
    # Indexing
    # ==========================================================================

    def index_papers(self, papers: Sequence[Paper]) -> None:
        """Index a collection of research papers for paper-level discovery.

        Each paper is represented by its title and abstract.

        Args:
            papers: Sequence of Paper domain models.
        """
        if not papers:
            self._paper_index = faiss.IndexFlatIP(self.dimension)
            self._papers = []
            return

        texts = [
            f"{p.title}\n\n{p.abstract}".strip() if p.abstract else p.title.strip()
            for p in papers
        ]

        embeddings = self.embedding_provider.embed_documents(texts)
        arr = np.asarray(embeddings, dtype=np.float32)
        faiss.normalize_L2(arr)

        index = faiss.IndexFlatIP(self.dimension)
        index.add(arr)

        self._paper_index = index
        self._papers = list(papers)

    def index_chunks(self, chunks: Sequence[Chunk]) -> None:
        """Index a collection of text chunks for fine-grained evidence retrieval.

        Args:
            chunks: Sequence of Chunk models.
        """
        if not chunks:
            self._chunk_index = faiss.IndexFlatIP(self.dimension)
            self._chunks = []
            return

        texts = [c.text for c in chunks]

        embeddings = self.embedding_provider.embed_documents(texts)
        arr = np.asarray(embeddings, dtype=np.float32)
        faiss.normalize_L2(arr)

        index = faiss.IndexFlatIP(self.dimension)
        index.add(arr)

        self._chunk_index = index
        self._chunks = list(chunks)

    # ==========================================================================
    # Search & Retrieval
    # ==========================================================================

    def search_papers(self, query: str, k: int = 5) -> list[ScoredPaper]:
        """Search the indexed papers by semantic similarity to a query.

        Args:
            query: Technical research question or search keywords.
            k: Maximum number of top papers to return.

        Returns:
            List of ScoredPaper models ordered by descending cosine similarity.
        """
        if not query.strip() or k <= 0:
            return []
        if self._paper_index is None or self._paper_index.ntotal == 0 or not self._papers:
            return []

        query_vec = self.embedding_provider.embed_query(query)
        q_arr = np.asarray([query_vec], dtype=np.float32)
        faiss.normalize_L2(q_arr)

        actual_k = min(k, self._paper_index.ntotal)
        distances, indices = self._paper_index.search(q_arr, actual_k)

        results: list[ScoredPaper] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._papers):
                continue
            paper = self._papers[idx]
            scored = ScoredPaper(
                **paper.model_dump(),
                score=round(float(dist), 4),
            )
            results.append(scored)

        return results

    def retrieve_evidence(
        self,
        query: str,
        paper_ids: Sequence[str] | None = None,
        k: int = 5,
    ) -> list[Evidence]:
        """Retrieve fine-grained evidence chunks matching a query.

        Args:
            query: Search query or factual technical question.
            paper_ids: Optional list of paper IDs to restrict candidate chunks to.
            k: Maximum number of evidence items to return.

        Returns:
            List of Evidence models with sequential source IDs (S1, S2, ...)
            ordered by descending similarity score.
        """
        if not query.strip() or k <= 0:
            return []
        if self._chunk_index is None or self._chunk_index.ntotal == 0 or not self._chunks:
            return []

        query_vec = self.embedding_provider.embed_query(query)
        q_arr = np.asarray([query_vec], dtype=np.float32)
        faiss.normalize_L2(q_arr)

        # When paper_ids filter is applied, search the full candidate set to guarantee top-k recall
        allowed_papers = set(paper_ids) if paper_ids is not None else None
        fetch_k = self._chunk_index.ntotal if allowed_papers is not None else min(k, self._chunk_index.ntotal)

        distances, indices = self._chunk_index.search(q_arr, fetch_k)

        results: list[Evidence] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._chunks):
                continue
            chunk = self._chunks[idx]

            if allowed_papers is not None and chunk.paper_id not in allowed_papers:
                continue

            evidence = Evidence(
                source_id=f"S{len(results) + 1}",
                chunk_id=chunk.chunk_id,
                paper_id=chunk.paper_id,
                page=chunk.page,
                section=chunk.section,
                text=chunk.text,
                score=round(float(dist), 4),
            )
            results.append(evidence)

            if len(results) == k:
                break

        return results

    def get_paper(self, paper_id: str) -> Paper | None:
        """Retrieve paper metadata and structural sections by paper ID.

        Args:
            paper_id: Unique paper identifier.

        Returns:
            The Paper model if found, else None.
        """
        for paper in self._papers:
            if paper.paper_id == paper_id:
                return paper
        return None

    # ==========================================================================
    # Persistence (Save & Load)
    # ==========================================================================

    def save(self, directory: Path | str | None = None) -> Path:
        """Persist FAISS binary indexes and metadata to disk.

        Args:
            directory: Target directory. Defaults to self.index_dir.

        Returns:
            The Path where indexes were saved.
        """
        target_dir = Path(directory or self.index_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        if self._paper_index is not None and self._papers:
            faiss.write_index(self._paper_index, str(target_dir / "papers.index"))
            papers_data = [p.model_dump() for p in self._papers]
            (target_dir / "papers.json").write_text(
                json.dumps(papers_data, indent=2), encoding="utf-8"
            )

        if self._chunk_index is not None and self._chunks:
            faiss.write_index(self._chunk_index, str(target_dir / "chunks.index"))
            chunks_data = [c.model_dump() for c in self._chunks]
            (target_dir / "chunks.json").write_text(
                json.dumps(chunks_data, indent=2), encoding="utf-8"
            )

        metadata = {
            "dimension": self.dimension,
            "num_papers": len(self._papers),
            "num_chunks": len(self._chunks),
        }
        (target_dir / "index_meta.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

        return target_dir

    def load(self, directory: Path | str | None = None) -> None:
        """Load FAISS binary indexes and metadata from disk.

        Args:
            directory: Directory containing index and json files. Defaults to self.index_dir.
        """
        target_dir = Path(directory or self.index_dir)

        papers_idx_path = target_dir / "papers.index"
        papers_json_path = target_dir / "papers.json"
        if papers_idx_path.exists() and papers_json_path.exists():
            self._paper_index = faiss.read_index(str(papers_idx_path))
            raw_papers = json.loads(papers_json_path.read_text(encoding="utf-8"))
            self._papers = [Paper(**p) for p in raw_papers]

        chunks_idx_path = target_dir / "chunks.index"
        chunks_json_path = target_dir / "chunks.json"
        if chunks_idx_path.exists() and chunks_json_path.exists():
            self._chunk_index = faiss.read_index(str(chunks_idx_path))
            raw_chunks = json.loads(chunks_json_path.read_text(encoding="utf-8"))
            self._chunks = [Chunk(**c) for c in raw_chunks]

    @classmethod
    def from_saved(
        cls,
        directory: Path | str,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> FAISSRetriever:
        """Factory method to instantiate and load a retriever from an existing index directory.

        Args:
            directory: Path to directory with index files.
            embedding_provider: Optional embedding provider.

        Returns:
            A ready-to-query FAISSRetriever.
        """
        retriever = cls(embedding_provider=embedding_provider, index_dir=directory)
        retriever.load(directory)
        return retriever
