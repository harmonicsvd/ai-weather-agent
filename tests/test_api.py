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
        def invoke(self, _state):
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
