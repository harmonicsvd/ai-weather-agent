from __future__ import annotations

"""Chunking utilities for RAG document preparation."""

from dataclasses import dataclass
from typing import Any

from apps.rag.loader import RAGDocument


@dataclass(frozen=True)
class RAGChunk:
    """One chunk ready for embedding/indexing."""

    chunk_id: str
    text: str
    metadata: dict[str, Any]


def _build_chunk_id(user_sub: str, source_file: str, chunk_index: int) -> str:
    """Stable chunk id for deterministic tests and traceability."""
    safe_source = source_file.replace(" ", "_")
    return f"{user_sub}:{safe_source}:{chunk_index}"


def chunk_documents(
    documents: list[RAGDocument],
    chunk_size: int = 700,
    overlap: int = 120,
) -> list[RAGChunk]:
    """
    Split document content into overlapping character chunks.

    Notes:
    - Character-based first version for simplicity.
    - Keeps metadata for downstream filtering and citations.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[RAGChunk] = []

    for doc in documents:
        text = (doc.content or "").strip()
        if not text:
            continue

        user_sub = str(doc.metadata.get("user_sub", "unknown"))
        source_file = str(doc.metadata.get("source_file", "unknown"))

        start = 0
        idx = 0
        step = chunk_size - overlap

        while start < len(text):
            end = min(start + chunk_size, len(text))
            piece = text[start:end].strip()
            if piece:
                chunk_meta = dict(doc.metadata)
                chunk_meta["chunk_index"] = idx
                chunk_meta["char_start"] = start
                chunk_meta["char_end"] = end

                chunks.append(
                    RAGChunk(
                        chunk_id=_build_chunk_id(user_sub, source_file, idx),
                        text=piece,
                        metadata=chunk_meta,
                    )
                )
            idx += 1
            start += step

    return chunks
