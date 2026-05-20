"""Tests for end-to-end RAG retriever flow with fake embeddings."""

from pathlib import Path

from apps.rag.retriever import retrieve_user_context


class FakeEmbeddingProvider:
    """
    Very simple deterministic embedding:
    vector = [count_of_word("weather"), count_of_word("safety"), text_length]
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        low = text.lower()
        return [
            float(low.count("weather")),
            float(low.count("safety")),
            float(len(low)),
        ]


def test_retrieve_user_context_returns_ranked_hits(tmp_path: Path, monkeypatch) -> None:
    # Create temporary knowledge base
    root = tmp_path / "knowledge"
    user_sub = "u-1"
    user_dir = root / user_sub
    user_dir.mkdir(parents=True)

    (user_dir / "a.md").write_text("weather weather wind site", encoding="utf-8")
    (user_dir / "b.md").write_text("safety helmet checklist", encoding="utf-8")
    (user_dir / "c.md").write_text("random unrelated text", encoding="utf-8")

    # Patch loader root by patching function call path
    from apps.rag import loader as loader_module
    from apps.rag import retriever as retriever_module

    monkeypatch.setattr(loader_module, "_default_data_root", lambda: root)
    monkeypatch.setattr(retriever_module, "load_user_knowledge_documents", loader_module.load_user_knowledge_documents)

    provider = FakeEmbeddingProvider()
    hits = retrieve_user_context(
        user_sub=user_sub,
        query_text="weather forecast for site",
        embedding_provider=provider,
        top_k=2,
    )

    assert len(hits) == 2
    assert all(h.chunk.metadata["user_sub"] == user_sub for h in hits)
    # first hit should be weather-heavy doc
    assert hits[0].chunk.metadata["source_file"] == "a.md"


def test_retrieve_user_context_returns_empty_when_no_docs(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "knowledge"

    from apps.rag import loader as loader_module
    from apps.rag import retriever as retriever_module

    monkeypatch.setattr(loader_module, "_default_data_root", lambda: root)
    monkeypatch.setattr(retriever_module, "load_user_knowledge_documents", loader_module.load_user_knowledge_documents)

    provider = FakeEmbeddingProvider()
    hits = retrieve_user_context(
        user_sub="missing-user",
        query_text="weather risk",
        embedding_provider=provider,
        top_k=3,
    )

    assert hits == []


def test_retrieve_user_context_raises_on_vector_count_mismatch(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "knowledge"
    user_sub = "u-1"
    user_dir = root / user_sub
    user_dir.mkdir(parents=True)
    (user_dir / "a.md").write_text("weather safety", encoding="utf-8")

    from apps.rag import loader as loader_module
    from apps.rag import retriever as retriever_module

    monkeypatch.setattr(loader_module, "_default_data_root", lambda: root)
    monkeypatch.setattr(retriever_module, "load_user_knowledge_documents", loader_module.load_user_knowledge_documents)

    class BadProvider(FakeEmbeddingProvider):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return []  # wrong count on purpose

    provider = BadProvider()

    try:
        retrieve_user_context(
            user_sub=user_sub,
            query_text="weather",
            embedding_provider=provider,
            top_k=3,
        )
        assert False, "Expected ValueError due to embedding count mismatch"
    except ValueError as exc:
        assert "mismatched document vector count" in str(exc)
