from __future__ import annotations

"""Embedding provider implementation for RAG retrieval."""

import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings


class GeminiEmbeddingProvider:
    """
    Real embedding provider backed by Google Generative AI embeddings.

    This class matches the `EmbeddingProvider` protocol used in retriever.py.
    """

    def __init__(
        self,
        model: str = "models/gemini-embedding-2",
        google_api_key: str | None = None,
    ) -> None:
        key = google_api_key or os.getenv("GOOGLE_API_KEY", "")
        if not key:
            raise ValueError("GOOGLE_API_KEY is not configured for embeddings.")

        self._emb = GoogleGenerativeAIEmbeddings(
            model=model,
            google_api_key=key,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._emb.embed_documents(texts)
        # normalize to plain list[list[float]]
        return [list(map(float, v)) for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        if not text.strip():
            return []
        vec = self._emb.embed_query(text)
        return list(map(float, vec))
