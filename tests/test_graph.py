import pytest
from apps.graph.nodes import score_event_weather_risk

from langgraph.checkpoint.sqlite import SqliteSaver
from apps.graph import workflows


def _make_weather(weather_code: int, wind_speed_kmh: float, temperature_c: float) -> dict:
    return {
        "location": {
            "name": "Test City",
            "latitude": 0.0,
            "longitude": 0.0,
            "country": "Test",
        },
        "current_weather": {
            "temperature_c": temperature_c,
            "apparent_temperature_c": temperature_c,
            "humidity_percent": 50,
            "wind_speed_kmh": wind_speed_kmh,
            "weather_code": weather_code,
            "observation_time": "2026-03-29T10:00:00",
        },
    }



def test_score_event_weather_risk_high_moderate_low() -> None:
    state = {
        "event_weather": [
            {
                "event": {"title": "Storm Review", "city": "Hamburg"},
                "weather": _make_weather(weather_code=82, wind_speed_kmh=10.0, temperature_c=8.0),
            },
            {
                "event": {"title": "Client Sync", "city": "Berlin"},
                "weather": _make_weather(weather_code=3, wind_speed_kmh=24.0, temperature_c=10.0),
            },
            {
                "event": {"title": "Team Catchup", "city": "Munich"},
                "weather": _make_weather(weather_code=2, wind_speed_kmh=8.0, temperature_c=15.0),
            },
        ]
    }

    result = score_event_weather_risk(state)

    risks = [item["risk"] for item in result["risk_summary"]]
    assert risks == ["high", "moderate", "low"]
    assert len(result["recommendations"]) == 3
    assert "high weather risk" in result["recommendations"][0]
    assert "moderate weather risk" in result["recommendations"][1]
    assert "low weather risk" in result["recommendations"][2]


def test_score_event_weather_risk_handles_unavailable_weather() -> None:
    state = {
        "event_weather": [
            {"event": {"title": "Unknown Event", "city": "Nowhere"}, "weather": None}
        ]
    }

    result = score_event_weather_risk(state)

    assert result["risk_summary"][0]["risk"] == "unknown"
    assert "Could not fetch weather" in result["recommendations"][0]
    
    
def _fake_fetch_weather_for_events(state):
    events = state.get("in_person_events") or []
    event_weather = []
    for event in events:
        event_weather.append(
            {
                "event": event,
                "weather": {
                    "location": {"name": event.get("city"), "latitude": 0, "longitude": 0, "country": "Test"},
                    "current_weather": {
                        "temperature_c": 10.0,
                        "apparent_temperature_c": 9.0,
                        "humidity_percent": 50,
                        "wind_speed_kmh": 10.0,
                        "weather_code": 3,
                        "observation_time": "2026-03-29T10:00:00",
                    },
                },
            }
        )
    return {"event_weather": event_weather}


def test_meeting_graph_checkpoint_history_grows(tmp_path, monkeypatch):
    monkeypatch.setattr(workflows, "fetch_weather_for_events", _fake_fetch_weather_for_events)

    db_path = tmp_path / "checkpoints.sqlite"
    with SqliteSaver.from_conn_string(str(db_path)) as checkpointer:
        app = workflows.build_meeting_preview_graph(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "meeting-test"}}
        inputs = {"user_query": "What are my meetings today?", "user_id": "1"}

        app.invoke(inputs, config=config)
        first = len(list(app.get_state_history(config)))

        app.invoke(inputs, config=config)
        second = len(list(app.get_state_history(config)))

    assert first > 0
    assert second > first