"""Local CLI runner for quickly testing the meeting preview graph."""

from langgraph.checkpoint.sqlite import SqliteSaver
from pathlib import Path
import sys


# When this script is executed directly from apps/graph/,
# Python would not automatically include project root on import path.
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # weather-agent/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
DB_PATH = Path(PROJECT_ROOT) / "data" / "checkpoints.sqlite"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

from apps.graph.workflows import build_meeting_preview_graph


def run_case(user_query: str, user_id: str, label: str) -> None:
    """
    Execute one graph run and print intermediate fields for debugging.

    Helpful for validating routing decisions, profile/city behavior, and
    recommendation output before testing through HTTP endpoints.
    """
    # Local debug runner: executes one graph call and prints key state outputs.
    db_path = PROJECT_ROOT / "data" / "checkpoints.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with SqliteSaver.from_conn_string(str(db_path)) as checkpointer:
        app = build_meeting_preview_graph(checkpointer=checkpointer)

        config = {"configurable": {"thread_id": f"meeting-{user_id}"}}
        result = app.invoke(
            
            {
                "user_query": user_query,
                "user_id": user_id,
                "user_sub": "104659023322141767006"  # Add this for profile lookup
            },
            config=config,
        )

        print(f"\n[{label}]")
        print("thread_id:", config["configurable"]["thread_id"])
        print("error:", result.get("error"))

        events = result.get("in_person_events") or []
        print(f"in_person_events_count: {len(events)}")
        for e in events:
            print(f"  - {e.get('title')} | {e.get('time')} | {e.get('city')} ({e.get('city_source')})")

        risk = result.get("risk_summary") or []
        print(f"risk_summary_count: {len(risk)}")
        for r in risk:
            print(
                f"  - {r.get('event_title')}: {r.get('risk')} | city={r.get('city')} | "
                f"code={r.get('weather_code')} | wind={r.get('wind_speed_kmh')} | temp={r.get('temperature_c')}"
            )

        recs = result.get("recommendations") or []
        print(f"recommendations_count: {len(recs)}")
        for i, rec in enumerate(recs, 1):
            print(f"  {i}. {rec}")

        retrieved_by_event = result.get("retrieved_context_by_event") or {}
        if retrieved_by_event:
            print("context_sources_by_event:")
            for event_title, items in retrieved_by_event.items():
                unique_sources = sorted({(it.get("source_file") or "unknown") for it in (items or [])})
                print(f"  - {event_title}: {', '.join(unique_sources) if unique_sources else 'none'}")

        print("final_response:")
        print(result.get("final_response"))

        latest = app.get_state(config)
        history = list(app.get_state_history(config))
        print("latest_checkpoint_id:", latest.config["configurable"].get("checkpoint_id"))
        print("checkpoint_count:", len(history))



if __name__ == "__main__":
    run_case(
        user_query="What are my meetings tomorrow?",
        user_id="1", # Add this for profile lookup
        label="meeting-preview-test"
    )
    
