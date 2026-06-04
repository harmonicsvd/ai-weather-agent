from pathlib import Path

from apps.rag.chunker import RAGChunk, chunk_documents
from apps.rag.loader import RAGDocument


def chunk_uploaded_markdown(
    *,
    user_sub: str,
    markdown_path: str | Path,
    original_filename: str,
) -> list[RAGChunk]:
    """Load generated Markdown and reuse the existing RAG chunker."""
    path = Path(markdown_path)
    markdown_text = path.read_text(encoding="utf-8")

    document = RAGDocument(
        content=markdown_text,
        metadata={
            "user_sub": user_sub,
            "source_file": original_filename,
            "source_path": str(path),
        },
    )

    return chunk_documents(
        [document],
        chunk_size=700,
        overlap=120,
    )