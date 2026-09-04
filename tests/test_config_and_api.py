"""Test script to verify OPENAI_API_KEY environment variable loading and optional API connectivity.

Zero credit usage:
- Environment variable check: 0 calls
- OpenAI models list endpoint: $0.00 / 0 credits consumed
"""

import pytest
from app.config import OPENAI_API_KEY, MODEL_PROVIDER, LLM_MODEL


def test_openai_key_loaded():
    """Verify that OPENAI_API_KEY is loaded from environment variables."""
    assert OPENAI_API_KEY, "OPENAI_API_KEY is not set or empty!"
    assert OPENAI_API_KEY.startswith("sk-"), "OPENAI_API_KEY does not start with 'sk-'"

    # Mask key for privacy
    masked_key = f"{OPENAI_API_KEY[:7]}...{OPENAI_API_KEY[-4:]}"
    print(f"\n[SUCCESS] OPENAI_API_KEY correctly loaded: {masked_key}")
    print(f"MODEL_PROVIDER: {MODEL_PROVIDER}")
    print(f"LLM_MODEL: {LLM_MODEL}")


def test_openai_api_connectivity():
    """Verify API key authentication with OpenAI using models.list ($0.00 cost / 0 token credits)."""
    if not OPENAI_API_KEY:
        pytest.skip("OPENAI_API_KEY not set")

    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    # models.list() tests key validity without spending any tokens/credits
    response = client.models.list()
    assert response.data, "No models returned from OpenAI API"

    model_ids = [m.id for m in response.data]
    assert "gpt-4o-mini" in model_ids or "gpt-4o" in model_ids or len(model_ids) > 0

    print(f"\n[SUCCESS] Successfully authenticated with OpenAI API! (Available models: {len(model_ids)})")


if __name__ == "__main__":
    test_openai_key_loaded()
    test_openai_api_connectivity()
