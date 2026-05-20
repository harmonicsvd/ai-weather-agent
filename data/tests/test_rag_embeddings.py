"""Tests for Gemini embedding provider (construction + shape behavior)."""

import pytest

from apps.rag.embeddings import GeminiEmbeddingProvider


def test_provider_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(ValueError):
        GeminiEmbeddingProvider(google_api_key=None)
