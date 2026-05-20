"""Tests for RAG chunking behavior."""

import pytest

from apps.rag.chunker import chunk_documents
from apps.rag.loader import RAGDocument


def test_chunk_documents_splits_long_text_and_preserves_metadata() -> None:
    docs = [
        RAGDocument(
            content="A" * 1000,
            metadata={"user_sub": "u1", "source_file": "notes.md", "role": "contractor"},
        )
    ]

    chunks = chunk_documents(docs, chunk_size=400, overlap=100)

    assert len(chunks) >= 3
    assert chunks[0].metadata["user_sub"] == "u1"
    assert chunks[0].metadata["source_file"] == "notes.md"
    assert chunks[0].metadata["chunk_index"] == 0
    assert chunks[0].metadata["char_start"] == 0
    assert chunks[0].metadata["char_end"] == 400
    assert chunks[0].chunk_id.startswith("u1:notes.md:")


def test_chunk_documents_skips_empty_content() -> None:
    docs = [
        RAGDocument(content="", metadata={"user_sub": "u1", "source_file": "a.md"}),
        RAGDocument(content="   ", metadata={"user_sub": "u1", "source_file": "b.md"}),
    ]

    chunks = chunk_documents(docs)
    assert chunks == []


def test_chunk_documents_validates_parameters() -> None:
    docs = [RAGDocument(content="hello", metadata={"user_sub": "u1", "source_file": "a.md"})]

    with pytest.raises(ValueError):
        chunk_documents(docs, chunk_size=0)

    with pytest.raises(ValueError):
        chunk_documents(docs, chunk_size=100, overlap=-1)

    with pytest.raises(ValueError):
        chunk_documents(docs, chunk_size=100, overlap=100)
