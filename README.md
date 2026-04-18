# AI Weather Agent

LangGraph-based weather decisioning service for meeting workflows.

## What It Does
- Reads meetings from a calendar backend.
- Filters in-person events.
- Resolves weather city (event city or profile fallback).
- Fetches event-time hourly weather from Open-Meteo.
- Scores commute/weather risk per event.
- Rewrites risk recommendations via structured LLM output with deterministic fallback.
- Returns recommendations and summary payloads via internal API.

## Scope
This repository owns:
- LangGraph workflow orchestration
- Weather retrieval + normalization
- Rule-based risk scoring
- Internal summary API for downstream services

It is designed to work with a companion backend (`voice-scheduling-agent`) that provides:
- Calendar read endpoints
- Profile endpoints
- Voice assistant integration

## Architecture
### Workflows
- `build_weather_graph` (`apps/graph/workflows.py`)
- `build_meeting_preview_graph` (`apps/graph/workflows.py`)

### Main Nodes
- `load_calendar_events`
- `filter_in_person_events`
- `apply_user_default_city`
- `fetch_weather_for_events`
- `score_event_weather_risk`
- `add_high_risk_actions`
- `llm_recommendation_rewrite`
- `format_meeting_recommendations`

Node implementations live in `apps/graph/nodes.py`.

### State
Graph state contract: `apps/graph/state.py`.

### Clients
- Weather: `apps/tools/weather_client.py`
- Calendar backend: `apps/tools/calendar_client.py`
- Profile backend: `apps/tools/profile_client.py`
- Schemas: `apps/tools/schemas.py`

## Risk Scoring
Core scoring is deterministic (rule-based):
- `high`
- `moderate`
- `low`
- `blocked` (missing location/city context)
- `unknown` (weather unavailable)

Recommendation text layer uses a structured LLM rewrite:
- model: `google_genai:gemini-2.5-flash`
- output contract: `LLMRecommendationsResponseSchema`
- fallback: if LLM/validation fails, keep deterministic recommendations

## Internal API
Defined in `apps/api/main.py`.

### Endpoints
- `GET /health`
- `GET /internal/meeting-weather-summary`

### `GET /internal/meeting-weather-summary`
Query params:
- `user_sub` (required)
- `date` (`YYYY-MM-DD`, optional)
- `tz` (IANA timezone, optional, default `Europe/Berlin`)

Header:
- `X-Internal-API-Key`

Response contains:
- event counts
- event list
- risk summary
- recommendations
- `summary_text`

## Integration Contract
This service calls the companion backend:
- `GET /internal/events`
- `GET /internal/profile/{sub}`

The companion backend calls this service:
- `GET /internal/meeting-weather-summary`

All internal routes are key-protected.

## Configuration
Copy `.env.example` to `.env` and set values.

Required variables:
```env
GOOGLE_API_KEY=
CALENDAR_API_BASE_URL=
CALENDAR_INTERNAL_API_KEY=
PROFILE_API_BASE_URL=
PROFILE_INTERNAL_API_KEY=
WEATHER_INTERNAL_API_KEY=
```

## Run
### Weather assistant entrypoint
```bash
python main.py
```

### Graph runner
```bash
python apps/graph/run_graph.py
```

### Internal API
```bash
python -m uvicorn apps.api.main:app --reload
```

## Tests
```bash
python -m pytest -q
```

## Current Status
- Event-time hourly forecast scoring is enabled (`get_weather_by_city_at_iso` path).
- High-risk conditional branch adds travel mitigation guidance.
- Structured LLM recommendation rewrite is active with schema validation and fallback.
