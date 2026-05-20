from __future__ import annotations

"""Simple in-memory vector index for RAG chunks."""

from dataclasses import dataclass

from typing import Any
import numpy as np

from apps.rag.chunker import RAGChunk


@dataclass(frozen=True)
class IndexedChunk:
    chunk: RAGChunk
    vector: list[float]


class InMemoryVectorIndex:
    """
    Minimal in-memory vector index.

    - Stores vectors + original chunks
    - Supports metadata filtering by user_sub
    - Returns top-k by cosine similarity
    """

    def __init__(self) -> None:
        self._items: list[IndexedChunk] = []

    def add(self, chunk: RAGChunk, vector: list[float]) -> None:
        if not vector:
            raise ValueError("vector must not be empty")
        self._items.append(IndexedChunk(chunk=chunk, vector=vector))

    def search(
        self,
        query_vector: list[float],
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[RAGChunk, float]]:
        if not query_vector:
            raise ValueError("query_vector must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be > 0")

        candidates = self._items
        if filters:
            candidates = [
                item for item in candidates
                if all(item.chunk.metadata.get(k) == v for k, v in filters.items())
            ]

        scored: list[tuple[RAGChunk, float]] = []
        for item in candidates:
            if len(item.vector) != len(query_vector):
                continue
            score = _cosine_similarity(query_vector, item.vector)
            scored.append((item.chunk, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]




def _cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.asarray(a, dtype=float)
    vb = np.asarray(b, dtype=float)

    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)

