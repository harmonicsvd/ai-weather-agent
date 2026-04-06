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
    db_path = PROJECT_ROOT / "data" / "checkpoints.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with SqliteSaver.from_conn_string(str(db_path)) as checkpointer:
        app = build_meeting_preview_graph(checkpointer=checkpointer)

        config = {"configurable": {"thread_id": f"meeting-{user_id}"}}
        result = app.invoke(
            {
                "user_query": user_query,
                "user_id": user_id,
            },
            config=config,
        )

        print(f"\n[{label}]")
        print("thread_id:", config["configurable"]["thread_id"])
        print("in_person_events:", result.get("in_person_events"))
        print("risk_summary:", result.get("risk_summary"))
        print("recommendations:", result.get("recommendations"))
        print("final_response:", result.get("final_response"))

        latest = app.get_state(config)
        history = list(app.get_state_history(config))
        print("latest_checkpoint_id:", latest.config["configurable"].get("checkpoint_id"))
        print("checkpoint_count:", len(history))



if __name__ == "__main__":
    run_case("What are my meetings today?", "2", "meeting-preview")
