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

        # Keep order + size identical to input list.
        normalized = [t.strip() for t in texts]

        # First try batch embedding (fast path).
        try:
            batch_vectors = self._emb.embed_documents(normalized)
        except Exception:
            batch_vectors = []

        # If provider returns exact count, use it.
        if len(batch_vectors) == len(normalized):
            return [list(map(float, v)) for v in batch_vectors]

        # Fallback: embed one-by-one so count always matches.
        vectors: list[list[float]] = []
        for text in normalized:
            if not text:
                # Shouldn't happen with your chunker, but keep safe.
                vectors.append([])
                continue
            vec = self._emb.embed_query(text)
            vectors.append(list(map(float, vec)))

        return vectors

    def embed_query(self, text: str) -> list[float]:
        if not text.strip():
            return []
        vec = self._emb.embed_query(text)
        return list(map(float, vec))
