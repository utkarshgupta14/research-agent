"""Embedding abstractions and provider implementations for ResearchPilot (Step 0.6).

Isolates embedding models behind an abstract interface to ensure the retrieval layer
and FAISS indexes remain decoupled from provider-specific APIs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence, Any
import numpy as np

from app.config import EMBEDDING_MODEL, MODEL_PROVIDER, OPENAI_API_KEY


class EmbeddingProvider(ABC):
    """Abstract interface for dense vector embedding providers."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of the dense vectors produced by this provider."""
        ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string into a dense vector."""
        ...

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a sequence of text documents or chunks into dense vectors."""
        ...


def _l2_normalize(vector: list[float] | np.ndarray) -> list[float]:
    """Normalize a vector to unit L2 norm so inner products equal cosine similarity."""
    arr = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm > 0.0:
        arr = arr / norm
    return arr.tolist()


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider using the official OpenAI client."""

    # Default dimensions lookup for well-known OpenAI models
    KNOWN_DIMENSIONS: dict[str, int] = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        dimensions: int | None = None,
        batch_size: int = 64,
        normalize: bool = True,
    ) -> None:
        """Initialize the OpenAI embedding provider.

        Args:
            api_key: OpenAI API key. If omitted, reads from OPENAI_API_KEY (environment or conda).
            model: Model name. If omitted, reads from EMBEDDING_MODEL config.
            dimensions: Optional vector dimensionality override (for models supporting shortening).
            batch_size: Maximum number of texts to embed in a single API call.
            normalize: Whether to ensure vectors are strictly L2 unit normalized.
        """
        import os

        resolved_key = (api_key or OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", "")).strip()
        if not resolved_key:
            raise ValueError(
                "OpenAI API key is missing. Please set OPENAI_API_KEY in your environment, "
                "conda environment variables (`conda env config vars set OPENAI_API_KEY=...`), "
                "or pass it explicitly."
            )

        self.model = model or EMBEDDING_MODEL or "text-embedding-3-small"
        self._dimensions_override = dimensions
        self.batch_size = max(1, batch_size)
        self.normalize = normalize

        from openai import OpenAI

        self.client = OpenAI(api_key=resolved_key)

    @property
    def dimension(self) -> int:
        """Dimensionality of the dense vectors."""
        if self._dimensions_override is not None:
            return self._dimensions_override
        if self.model in self.KNOWN_DIMENSIONS:
            return self.KNOWN_DIMENSIONS[self.model]
        # Default fallback for standard embedding models
        return 1536

    def embed_query(self, text: str) -> list[float]:
        """Embed a single search query into a dense vector.

        Args:
            text: Query string.

        Returns:
            A list of floats representing the embedding vector.
        """
        cleaned = text.strip()
        if not cleaned:
            # Fallback zero-vector with correct dimensionality
            return [0.0] * self.dimension

        kwargs: dict[str, Any] = {"model": self.model, "input": cleaned}
        if self._dimensions_override is not None:
            kwargs["dimensions"] = self._dimensions_override

        response = self.client.embeddings.create(**kwargs)
        vector = response.data[0].embedding

        if self.normalize:
            vector = _l2_normalize(vector)
        return vector

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a sequence of documents/chunks into dense vectors in batches.

        Args:
            texts: List or sequence of text strings.

        Returns:
            A list of embedding vectors corresponding to the input texts.
        """
        if not texts:
            return []

        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = list(texts[i : i + self.batch_size])
            # OpenAI requires non-empty inputs; replace empty strings with a single space placeholder
            sanitized_batch = [t if t.strip() else " " for t in batch]

            kwargs: dict[str, Any] = {"model": self.model, "input": sanitized_batch}
            if self._dimensions_override is not None:
                kwargs["dimensions"] = self._dimensions_override

            response = self.client.embeddings.create(**kwargs)

            # Sort by index to ensure ordering is strictly preserved
            sorted_data = sorted(response.data, key=lambda item: item.index)
            for item in sorted_data:
                vec = item.embedding
                if self.normalize:
                    vec = _l2_normalize(vec)
                all_embeddings.append(vec)

        return all_embeddings


def get_embedding_provider(
    provider: str | None = None,
    **kwargs,
) -> EmbeddingProvider:
    """Factory to create an EmbeddingProvider based on configuration.

    Args:
        provider: Provider name (e.g., 'openai'). If None, uses MODEL_PROVIDER config.
        **kwargs: Provider-specific options forwarded to the constructor.

    Returns:
        An instance conforming to EmbeddingProvider.
    """
    resolved_provider = (provider or MODEL_PROVIDER or "openai").lower()

    if resolved_provider == "openai":
        return OpenAIEmbeddingProvider(**kwargs)

    raise ValueError(
        f"Unsupported embedding provider: '{resolved_provider}'. "
        f"Supported providers in MVP 0: 'openai'."
    )
