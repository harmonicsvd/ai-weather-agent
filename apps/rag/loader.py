from __future__ import annotations

"""Document loader for user-scoped RAG knowledge files."""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SUPPORTED_TEXT_EXTENSIONS = {".md", ".txt"}


@dataclass(frozen=True)
class RAGDocument:
    """
    One raw document unit before chunking/embedding.

    - `content` keeps user-authored text (lightly sanitized)
    - `metadata` holds routing/filter keys such as user_sub and source file
    """

    content: str
    metadata: dict[str, Any]


def _default_data_root() -> Path:
    """Resolve the repository data root when caller does not pass one."""
    return Path(__file__).resolve().parents[2] / "data" / "knowledge"


def _sanitize_content(raw_text: str) -> str:
    """
    Normalize noisy text while preserving meaning for retrieval.

    We remove obvious identity headers when present to keep user identity as
    metadata responsibility rather than retrieval text responsibility.
    """
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    cleaned_lines: list[str] = []

    identity_patterns = [
        re.compile(r"^\s*user_sub\s*[:= ]", re.IGNORECASE),
        re.compile(r"^\s*for\s*:?\s*user_sub\b", re.IGNORECASE),
        re.compile(r"^\s*role\s*[:= ]", re.IGNORECASE),
    ]

    for line in lines:
        if any(pattern.search(line) for pattern in identity_patterns):
            continue
        cleaned_lines.append(line.rstrip())

    # Keep paragraph structure but collapse 3+ consecutive blank lines.
    cleaned = "\n".join(cleaned_lines).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


def _load_manifest(knowledge_dir: Path) -> dict[str, Any]:
    """Load optional `knowledge_manifest.json` from a user knowledge directory."""
    manifest_path = knowledge_dir / "knowledge_manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _iter_text_files(knowledge_dir: Path) -> list[Path]:
    """List supported text files in stable order for deterministic tests/runs."""
    files = []
    for path in knowledge_dir.glob("*"):
        if not path.is_file():
            continue
        if path.name == "knowledge_manifest.json":
            continue
        if path.suffix.lower() not in SUPPORTED_TEXT_EXTENSIONS:
            continue
        files.append(path)
    return sorted(files, key=lambda p: p.name.lower())


def load_user_knowledge_documents(
    user_sub: str,
    data_root: str | Path | None = None,
) -> list[RAGDocument]:
    """
    Load all knowledge documents for one user.

    Args:
    - user_sub: unique user identifier used for storage scoping
    - data_root: override root directory; defaults to `data/knowledge`

    Returns:
    - list of `RAGDocument` with text + metadata ready for chunking
    """
    normalized_user_sub = (user_sub or "").strip()
    if not normalized_user_sub:
        raise ValueError("user_sub is required to load knowledge documents.")

    root = Path(data_root) if data_root is not None else _default_data_root()
    knowledge_dir = root / normalized_user_sub
    if not knowledge_dir.exists():
        return []

    manifest = _load_manifest(knowledge_dir)
    manifest_role = (manifest.get("role") or "").strip() or None

    documents: list[RAGDocument] = []
    for file_path in _iter_text_files(knowledge_dir):
        try:
            raw_text = file_path.read_text(encoding="utf-8")
        except OSError:
            continue

        content = _sanitize_content(raw_text)
        if not content:
            continue

        documents.append(
            RAGDocument(
                content=content,
                metadata={
                    "user_sub": normalized_user_sub,
                    "source_file": file_path.name,
                    "source_path": str(file_path),
                    "role": manifest_role,
                },
            )
        )

    return documents

