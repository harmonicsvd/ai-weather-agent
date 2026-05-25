from __future__ import annotations

"""RAG retriever pipeline: load -> chunk -> embed -> index -> search."""

from dataclasses import dataclass
from typing import Protocol

from apps.rag.chunker import RAGChunk, chunk_documents
from apps.rag.index import InMemoryVectorIndex
from apps.rag.loader import load_user_knowledge_documents


class EmbeddingProvider(Protocol):
    """Protocol so we can swap embedding backends (Gemini/OpenAI/local)."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


@dataclass(frozen=True)
class RetrievedContext:
    chunk: RAGChunk
    score: float


def retrieve_user_context(
    *,
    user_sub: str,
    query_text: str,
    embedding_provider: EmbeddingProvider,
    top_k: int = 3,
) -> list[RetrievedContext]:
    """
    End-to-end retrieval for one user.

    Returns top-k chunk matches with scores.
    """
    if not query_text.strip():
        return []

    docs = load_user_knowledge_documents(user_sub=user_sub)
    if not docs:
        return []

    chunks = chunk_documents(docs, chunk_size=700, overlap=120)
    if not chunks:
        return []

    chunk_texts = [c.text for c in chunks]
    chunk_vectors = embedding_provider.embed_documents(chunk_texts)

    if len(chunk_vectors) != len(chunks):
        raise ValueError(
            f"Embedding provider returned mismatched document vector count: "
            f"{len(chunk_vectors)} vs {len(chunks)}"
        )

    # Skip bad/empty vectors instead of crashing retrieval.
    pairs = [
        (chunk, vec)
        for chunk, vec in zip(chunks, chunk_vectors)
        if isinstance(vec, list) and len(vec) > 0
    ]
    if not pairs:
        return []

    index = InMemoryVectorIndex()
    for chunk, vec in pairs:
        index.add(chunk, vec)
    query_vector = embedding_provider.embed_query(query_text)
    hits = index.search(query_vector, top_k=top_k, filters={"user_sub": user_sub})

    return [RetrievedContext(chunk=chunk, score=score) for chunk, score in hits]
