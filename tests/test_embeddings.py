"""Tests for the embedding component in ResearchPilot (Step 0.6).

Includes:
- Unit tests verifying interface compliance, normalization, batching, and error handling.
- Live smoke test embedding one query and a small batch of texts with semantic checks.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch
import pytest

from app.retrieval.embeddings import (
    EmbeddingProvider,
    OpenAIEmbeddingProvider,
    _l2_normalize,
    get_embedding_provider,
)


# ==============================================================================
# 1. Unit Tests (Offline / Mocked - Zero Credits)
# ==============================================================================

def test_interface_compliance():
    """Verify that OpenAIEmbeddingProvider implements EmbeddingProvider."""
    assert issubclass(OpenAIEmbeddingProvider, EmbeddingProvider)


def test_missing_api_key_raises_value_error(monkeypatch):
    """Verify ValueError is raised with clear messaging when API key is missing."""
    monkeypatch.setenv("OPENAI_API_KEY", "")
    with patch("app.retrieval.embeddings.OPENAI_API_KEY", ""):
        with pytest.raises(ValueError, match="OpenAI API key is missing"):
            OpenAIEmbeddingProvider(api_key="")


def test_l2_normalization_math():
    """Verify that _l2_normalize produces a unit vector."""
    raw_vector = [3.0, 4.0]
    norm_vector = _l2_normalize(raw_vector)
    norm = math.sqrt(sum(x ** 2 for x in norm_vector))
    assert math.isclose(norm, 1.0, rel_tol=1e-5)
    assert math.isclose(norm_vector[0], 0.6, rel_tol=1e-5)
    assert math.isclose(norm_vector[1], 0.8, rel_tol=1e-5)


def test_l2_normalization_zero_vector():
    """Verify that zero vectors are handled gracefully without ZeroDivisionError."""
    raw_vector = [0.0, 0.0, 0.0]
    norm_vector = _l2_normalize(raw_vector)
    assert norm_vector == [0.0, 0.0, 0.0]


def test_batching_logic_with_mock():
    """Verify that embed_documents correctly splits requests exceeding batch_size."""
    mock_client = MagicMock()

    def mock_create(model, input, **kwargs):
        resp = MagicMock()
        items = []
        for idx, text in enumerate(input):
            item = MagicMock()
            item.index = idx
            # Mock 4-dimensional vector
            item.embedding = [1.0, 0.0, 0.0, 0.0]
            items.append(item)
        resp.data = items
        return resp

    mock_client.embeddings.create.side_effect = mock_create

    with patch("openai.OpenAI", return_value=mock_client):
        provider = OpenAIEmbeddingProvider(
            api_key="sk-mock-key-for-testing",
            batch_size=2,
            normalize=True,
        )

        texts = ["Text 1", "Text 2", "Text 3", "Text 4", "Text 5"]
        results = provider.embed_documents(texts)

        # 5 texts with batch_size=2 should require ceil(5/2) = 3 calls
        assert mock_client.embeddings.create.call_count == 3
        assert len(results) == 5
        assert len(results[0]) == 4


def test_empty_documents_returns_empty_list():
    """Verify that passing an empty list returns an empty list without calling API."""
    mock_client = MagicMock()
    with patch("openai.OpenAI", return_value=mock_client):
        provider = OpenAIEmbeddingProvider(api_key="sk-mock-key-for-testing")
        assert provider.embed_documents([]) == []
        mock_client.embeddings.create.assert_not_called()


def test_factory_function():
    """Verify get_embedding_provider returns configured provider."""
    with patch("openai.OpenAI"):
        provider = get_embedding_provider(
            provider="openai",
            api_key="sk-mock-key-for-testing",
        )
        assert isinstance(provider, OpenAIEmbeddingProvider)

    with pytest.raises(ValueError, match="Unsupported embedding provider"):
        get_embedding_provider(provider="unsupported_dummy")


# ==============================================================================
# 2. Live Smoke Test (Real API Call - Minimal Token Usage)
# ==============================================================================

def test_live_embedding_smoke_test():
    """Live smoke test embedding 1 query and a small batch of 3 texts.

    Verifies:
    1. Query embedding returns a vector of dimension 1536.
    2. Document batch embedding returns 3 vectors of dimension 1536.
    3. All vectors are unit L2-normalized (length == 1.0).
    4. Semantic sanity: query about attention is more similar to transformer text
       than to an unrelated distractor (chocolate cookie recipe).
    """
    try:
        provider = get_embedding_provider()
    except ValueError as err:
        pytest.skip(f"Live test skipped: {err}")

    query = "attention mechanisms in transformer architectures"
    documents = [
        "Self-attention mechanisms allow sequence models to attend to all positions simultaneously.",
        "Convolutional neural networks apply localized kernels across spatial feature maps.",
        "To bake chocolate chip cookies, preheat the oven to 375 degrees and combine butter with sugar.",
    ]

    # 1. Embed query
    query_vec = provider.embed_query(query)
    assert len(query_vec) == provider.dimension, (
        f"Expected query vector dimension {provider.dimension}, got {len(query_vec)}"
    )

    # 2. Embed batch of documents
    doc_vecs = provider.embed_documents(documents)
    assert len(doc_vecs) == len(documents), (
        f"Expected {len(documents)} document vectors, got {len(doc_vecs)}"
    )
    for i, vec in enumerate(doc_vecs):
        assert len(vec) == provider.dimension, (
            f"Doc {i} vector has dimension {len(vec)}, expected {provider.dimension}"
        )

    # 3. Verify L2 normalization
    q_norm = math.sqrt(sum(x ** 2 for x in query_vec))
    assert math.isclose(q_norm, 1.0, rel_tol=1e-3), f"Query vector norm was {q_norm}"

    for i, vec in enumerate(doc_vecs):
        d_norm = math.sqrt(sum(x ** 2 for x in vec))
        assert math.isclose(d_norm, 1.0, rel_tol=1e-3), f"Doc {i} vector norm was {d_norm}"

    # 4. Semantic similarity check (dot product of L2 normalized vectors == cosine similarity)
    cos_sim_relevant = sum(q * d for q, d in zip(query_vec, doc_vecs[0]))
    cos_sim_domain = sum(q * d for q, d in zip(query_vec, doc_vecs[1]))
    cos_sim_distractor = sum(q * d for q, d in zip(query_vec, doc_vecs[2]))

    print(f"\n[Smoke Test Results]")
    print(f"Query: '{query}'")
    print(f"Dimension: {provider.dimension}")
    print(f"Similarity -> Transformer chunk: {cos_sim_relevant:.4f}")
    print(f"Similarity -> CNN chunk:         {cos_sim_domain:.4f}")
    print(f"Similarity -> Cookie recipe:     {cos_sim_distractor:.4f}")

    # The transformer text must be significantly more similar to the query than the cookie recipe
    assert cos_sim_relevant > cos_sim_distractor, (
        f"Semantic check failed: relevant ({cos_sim_relevant:.4f}) <= distractor ({cos_sim_distractor:.4f})"
    )
    assert cos_sim_relevant > cos_sim_domain, (
        f"Transformer chunk ({cos_sim_relevant:.4f}) should score higher than CNN chunk ({cos_sim_domain:.4f})"
    )


if __name__ == "__main__":
    print("Running embedding smoke test...")
    test_interface_compliance()
    test_l2_normalization_math()
    test_l2_normalization_zero_vector()
    test_batching_logic_with_mock()
    test_empty_documents_returns_empty_list()
    test_factory_function()
    test_live_embedding_smoke_test()
    print("\n[SUCCESS] All embedding unit tests and live smoke test passed!")
