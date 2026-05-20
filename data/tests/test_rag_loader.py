"""Tests for Phase 3 user-document loading used by the RAG pipeline."""

import json
from pathlib import Path

from apps.rag.loader import load_user_knowledge_documents


def _write(path: Path, text: str) -> None:
    """Write text file using UTF-8 for test fixtures."""
    path.write_text(text, encoding="utf-8")


def test_load_user_knowledge_documents_returns_scoped_files(tmp_path: Path) -> None:
    """Loader should only read files under the requested `user_sub` directory."""
    root = tmp_path / "knowledge"
    user_dir = root / "u-1"
    user_dir.mkdir(parents=True)
    other_dir = root / "u-2"
    other_dir.mkdir(parents=True)

    _write(user_dir / "site_notes.md", "# site\nline a")
    _write(user_dir / "travel_logistics.md", "line b")
    _write(other_dir / "private.md", "must not be loaded for u-1")

    docs = load_user_knowledge_documents("u-1", data_root=root)

    assert len(docs) == 2
    assert {d.metadata["source_file"] for d in docs} == {
        "site_notes.md",
        "travel_logistics.md",
    }
    assert all(d.metadata["user_sub"] == "u-1" for d in docs)


def test_loader_applies_manifest_role_and_sanitizes_identity_lines(tmp_path: Path) -> None:
    """Identity hints should be removed from text and stored as metadata instead."""
    root = tmp_path / "knowledge"
    user_dir = root / "u-1"
    user_dir.mkdir(parents=True)

    (user_dir / "knowledge_manifest.json").write_text(
        json.dumps({"user_sub": "u-1", "role": "contractor"}, ensure_ascii=True),
        encoding="utf-8",
    )
    _write(
        user_dir / "notes.md",
        "\n".join(
            [
                "user_sub: u-1",
                "role: contractor",
                "Actual note line",
            ]
        ),
    )

    docs = load_user_knowledge_documents("u-1", data_root=root)

    assert len(docs) == 1
    assert docs[0].metadata["role"] == "contractor"
    assert "user_sub:" not in docs[0].content.lower()
    assert "role:" not in docs[0].content.lower()
    assert "Actual note line" in docs[0].content


def test_loader_returns_empty_list_when_directory_missing(tmp_path: Path) -> None:
    """Unknown users should return no documents rather than raising errors."""
    docs = load_user_knowledge_documents("missing-user", data_root=tmp_path / "knowledge")
    assert docs == []

