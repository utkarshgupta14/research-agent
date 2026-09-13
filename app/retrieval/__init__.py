"""Retrieval module for ResearchPilot."""

from app.retrieval.embeddings import (
    EmbeddingProvider,
    OpenAIEmbeddingProvider,
    get_embedding_provider,
)

__all__ = [
    "EmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "get_embedding_provider",
]
