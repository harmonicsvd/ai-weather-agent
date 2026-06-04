from __future__ import annotations

import os
from contextlib import contextmanager

import psycopg
from uuid import uuid4

from apps.rag.chunker import RAGChunk


def get_database_url() -> str:
    return os.getenv("DATABASE_URL", "").strip()


def get_embedding_dimension() -> int:
    return int(os.getenv("EMBEDDING_DIMENSION", "3072"))

@contextmanager
def get_vector_db():
    """Open a Postgres connection for Sham vector storage."""
    database_url = get_database_url()
    if not database_url:
        raise RuntimeError("DATABASE_URL is required for vector storage.")

    conn = psycopg.connect(database_url)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_vector_store() -> None:
    """Create pgvector extension and chunk-vector table if needed."""
    dimension = get_embedding_dimension()
    with get_vector_db() as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS knowledge_chunk_vectors (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                user_sub TEXT NOT NULL,
                source_file TEXT NOT NULL,
                source_path TEXT,
                chunk_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                chunk_text TEXT NOT NULL,
                embedding vector({dimension}) NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_knowledge_chunk_vectors_user_sub
            ON knowledge_chunk_vectors(user_sub)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_knowledge_chunk_vectors_document_id
            ON knowledge_chunk_vectors(document_id)
            """
        )
        
def _vector_literal(vector: list[float]) -> str:
    """Convert Python vector list into pgvector literal format."""
    return "[" + ",".join(str(float(value)) for value in vector) + "]"

def save_chunk_vectors(
    *,
    document_id: str,
    user_sub: str,
    chunks: list[RAGChunk],
    vectors: list[list[float]],
) -> int:
    """Persist chunk text + embeddings in pgvector."""
    if len(chunks) != len(vectors):
        raise ValueError("chunks and vectors must have the same length")

    rows = []

    for chunk, vector in zip(chunks, vectors):
        rows.append(
            (
                uuid4().hex,
                document_id,
                user_sub,
                chunk.metadata.get("source_file") or "unknown",
                chunk.metadata.get("source_path"),
                chunk.chunk_id,
                int(chunk.metadata.get("chunk_index", 0)),
                chunk.text,
                _vector_literal(vector),
            )
        )

    with get_vector_db() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO knowledge_chunk_vectors (
                    id,
                    document_id,
                    user_sub,
                    source_file,
                    source_path,
                    chunk_id,
                    chunk_index,
                    chunk_text,
                    embedding
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
                """,
                rows,
            )
    return len(rows)


def search_chunk_vectors(
    *,
    user_sub: str,
    query_vector: list[float],
    top_k: int = 3,
) -> list[dict]:
    """Search persisted Sham vectors by cosine similarity for one user."""
    if not user_sub.strip():
        return []
    if not query_vector:
        return []
    if top_k <= 0:
        raise ValueError("top_k must be > 0")

    query_literal = _vector_literal(query_vector)

    with get_vector_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id,
                    document_id,
                    user_sub,
                    source_file,
                    source_path,
                    chunk_id,
                    chunk_index,
                    chunk_text,
                    1 - (embedding <=> %s::vector) AS score
                FROM knowledge_chunk_vectors
                WHERE user_sub = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (query_literal, user_sub, query_literal, top_k),
            )
            rows = cur.fetchall()

    return [
        {
            "id": row[0],
            "document_id": row[1],
            "user_sub": row[2],
            "source_file": row[3],
            "source_path": row[4],
            "chunk_id": row[5],
            "chunk_index": row[6],
            "text": row[7],
            "score": float(row[8]),
        }
        for row in rows
    ]

