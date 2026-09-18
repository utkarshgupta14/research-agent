"""Retrieval module for ResearchPilot."""

from app.retrieval.base import Retriever
from app.retrieval.embeddings import (
    EmbeddingProvider,
    OpenAIEmbeddingProvider,
    get_embedding_provider,
)
from app.retrieval.faiss_retriever import FAISSRetriever

__all__ = [
    "Retriever",
    "FAISSRetriever",
    "EmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "get_embedding_provider",
]

