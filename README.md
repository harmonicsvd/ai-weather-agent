# AI Weather Agent

An AI weather assistant built with LangChain and LangGraph.

## Overview
This project supports two flows:
- Single-city weather assistant (tool-calling agent)
- Meeting weather-risk preview (graph orchestration + calendar events)

## Current Working Status
- LangChain weather tool flow is live in `main.py`
- Open-Meteo client is implemented with retry + timeout handling
- Pydantic schemas validate weather and calendar payloads
- LangGraph meeting preview flow is implemented with:
  - calendar event loading
  - in-person meeting filtering
  - per-event weather fetch
  - weather risk scoring (`low` / `moderate` / `high`)
  - final recommendation formatting
- SQLite checkpointing is enabled for graph run history
- Test suites are passing for weather client and graph logic

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
- Improve meeting location-to-city resolution (handle `location=None` and free-text addresses)
- Add dedicated tests for calendar event loading node
- Add graceful fallbacks when calendar data is incomplete
- Extend risk scoring with event time + forecast window instead of only current weather
