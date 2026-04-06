# AI Weather Agent

An AI weather assistant built with LangChain and LangGraph.

## Overview
This project supports two flows:
- Single-city weather assistant (tool-calling agent)
- Meeting weather-risk preview (graph orchestration + calendar events)

## Project Evolution
### Phase 1: LangChain Foundation
- Implemented a tool-calling weather assistant in `main.py`
- Added a resilient Open-Meteo client with retry and timeout handling
- Introduced Pydantic schemas for typed weather responses
- Added tests for weather client behavior and error handling

### Phase 2: LangGraph Orchestration
- Built a stateful meeting-weather workflow using LangGraph
- Added graph nodes for:
  - calendar event loading
  - in-person event filtering
  - user default-city fallback (mock JSON profile store)
  - per-event weather fetch
  - weather risk scoring (`low` / `moderate` / `high`, plus `blocked`/`unknown`)
  - recommendation formatting
- Enabled SQLite checkpointing for thread-level state history
- Added graph tests for risk logic, filtering, and checkpoint growth

## Current Working Status
- Calendar events are loaded from a backend API (`/events`)
- In-person meetings are identified using explicit `meeting_mode`
- If event city is missing, graph can use user default city from `data/user_profiles.json`
- Weather risk recommendations are generated per meeting with robust fallback handling

## Model
- Provider: Google GenAI
- Model: `gemini-2.5-flash`
- Integration: `init_chat_model("google_genai:gemini-2.5-flash")`

## Repository Structure
- `main.py`: LangChain weather agent entrypoint
- `apps/tools/weather_client.py`: Open-Meteo weather client
- `apps/tools/calendar_client.py`: calendar events API client
- `apps/tools/schemas.py`: Pydantic schemas
- `apps/graph/state.py`: shared graph state contract
- `apps/graph/nodes.py`: graph node implementations
- `apps/graph/workflows.py`: graph definitions
- `apps/graph/run_graph.py`: local graph runner
- `tests/test_weather_client.py`: weather client tests
- `tests/test_graph.py`: graph/risk/checkpoint tests

## Requirements
- Python 3.11+
- Dependencies:
  - `langchain`
  - `langgraph`
  - `python-dotenv`
  - `httpx`
  - `tenacity`
  - `pydantic`
  - `pytest`

## Environment
Create `.env` in project root with:

```env
GOOGLE_API_KEY=your_google_api_key_here
CALENDAR_API_BASE_URL=http://127.0.0.1:8000
```

`CALENDAR_API_BASE_URL` should point to your running calendar backend (`voice-scheduling-agent`) that exposes `GET /events`.

## Run
Weather agent:

```bash
python main.py
```

Meeting preview graph:

```bash
python apps/graph/run_graph.py
```

## Test
```bash
python -m pytest -q
```

## Next Progress
- Replace mock profile JSON with real persisted user profile storage
- Add dedicated tests for calendar loading + profile fallback integration
- Extend risk scoring to event-time forecast instead of only current weather
- Add API/graph observability metrics for production monitoring
