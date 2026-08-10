"""API-level tests for weather-agent internal summary endpoint."""

from fastapi.testclient import TestClient
import pytest

from apps.api import main as api_main


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch):
    """Test client fixture with deterministic internal API key override."""
    monkeypatch.setattr(api_main, "WEATHER_INTERNAL_API_KEY", "test-weather-key")
    with TestClient(api_main.app) as c:
        yield c


def test_internal_meeting_weather_summary_requires_internal_key(client: TestClient) -> None:
    """Internal summary endpoint must reject calls without auth header."""
    response = client.get(
        "/internal/meeting-weather-summary",
        params={"user_sub": "u1", "date": "2026-04-15", "tz": "Europe/Berlin"},
    )
    assert response.status_code == 403
    assert response.json()["error"] == "forbidden"


def test_internal_meeting_weather_summary_returns_graph_derived_payload(
    monkeypatch: pytest.MonkeyPatch,
    client: TestClient,
) -> None:
    """Endpoint should map graph output into stable API summary contract."""
    class _FakeGraph:
        """Minimal graph stub so the API test can avoid real graph execution."""
        def invoke(self, _state):
            """Return a deterministic payload shaped like real graph output."""
            return {
                "events": [
                    {
                        "title": "Client Visit",
                        "time": "2026-04-15T10:00:00+02:00",
                        "location": "Berlin Office",
                        "meeting_mode": "in_person",
                        "is_virtual": False,
                        "city": "Berlin",
                    },
                    {
                        "title": "Online Sync",
                        "time": "2026-04-15T14:00:00+02:00",
                        "location": None,
                        "meeting_mode": "online",
                        "is_virtual": True,
                        "city": None,
                    },
                ],
                "in_person_events": [
                    {
                        "title": "Client Visit",
                        "time": "2026-04-15T10:00:00+02:00",
                        "location": "Berlin Office",
                        "meeting_mode": "in_person",
                        "is_virtual": False,
                        "city": "Berlin",
                    }
                ],
                "risk_summary": [{"event_title": "Client Visit", "risk": "low"}],
                "recommendations": ["Client Visit (Berlin): low weather risk."],
            }

    monkeypatch.setattr(api_main, "MEETING_PREVIEW_APP", _FakeGraph())

    response = client.get(
        "/internal/meeting-weather-summary",
        params={"user_sub": "u1", "date": "2026-04-15", "tz": "Europe/Berlin"},
        headers={"X-Internal-API-Key": "test-weather-key"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["counts"] == {"total": 2, "in_person": 1, "online": 1}
    assert "you have 2 meetings: 1 in-person and 1 online" in body["summary_text"].lower()
    assert body["risk_summary"][0]["risk"] == "low"


def test_app_starts_when_vector_store_init_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing/suspended Postgres vector DB should not fail API startup."""
    monkeypatch.setattr(api_main, "WEATHER_INTERNAL_API_KEY", "test-weather-key")
    monkeypatch.setattr(
        api_main,
        "init_vector_store",
        lambda: (_ for _ in ()).throw(OSError("failed to resolve host")),
    )

    with TestClient(api_main.app) as test_client:
        response = test_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert api_main.VECTOR_STORE_AVAILABLE is False


def test_knowledge_upload_falls_back_when_vector_save_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """PDF ingestion should keep file/markdown fallback when pgvector save is unavailable."""
    class _FakeEmbeddingProvider:
        def embed_documents(self, texts):
            return [[1.0, 0.0] for _text in texts]

    class _Chunk:
        text = "site safety notes"

    monkeypatch.setattr(api_main, "WEATHER_INTERNAL_API_KEY", "test-weather-key")
    monkeypatch.setattr(api_main, "init_vector_store", lambda: None)
    monkeypatch.setattr(api_main, "extract_pdf_to_markdown", lambda *_args: "# Notes")
    monkeypatch.setattr(api_main, "chunk_uploaded_markdown", lambda **_kwargs: [_Chunk()])
    monkeypatch.setattr(api_main, "GeminiEmbeddingProvider", _FakeEmbeddingProvider)
    monkeypatch.setattr(
        api_main,
        "save_chunk_vectors",
        lambda **_kwargs: (_ for _ in ()).throw(OSError("failed to resolve host")),
    )

    with TestClient(api_main.app) as test_client:
        response = test_client.post(
            "/internal/knowledge/upload",
            data={"user_sub": "u1"},
            files={"file": ("notes.pdf", b"%PDF-1.4", "application/pdf")},
            headers={"X-Internal-API-Key": "test-weather-key"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["saved_vector_count"] == 0
    assert body["vector_store_available"] is False
