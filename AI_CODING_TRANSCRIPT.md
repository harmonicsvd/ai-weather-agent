# AI Coding Transcript - Voice Scheduling + Weather Intelligence

## Candidate
- Name: Varad Kulkarni
- Role Applied: Full-Stack AI Engineer

## Project Scope
This work is part of a two-service system:

1. `voice-scheduling-agent` (FastAPI + UI + OAuth + VAPI tool webhooks)  
2. `weather-agent` (FastAPI internal API + LangGraph workflow + weather/LLM recommendation pipeline)

Core goal:
- Users can schedule meetings by voice
- Users can ask "what are my meetings today?"
- System returns meeting summary + weather-aware guidance for in-person events

---

## Stack Used
- Python, FastAPI
- LangGraph, LangChain
- Gemini 2.5 Flash
- VAPI tool-calling
- Google Calendar API
- Open-Meteo API
- SQLite
- pytest
- Render deployment

---

## Architecture Summary

### Service Responsibilities
- `voice-scheduling-agent`:
  - Google OAuth and session handling
  - Create-event tool endpoint
  - Internal events/profile endpoints
  - Meetings summary endpoint that delegates to weather service

- `weather-agent`:
  - Internal `/internal/meeting-weather-summary` endpoint
  - LangGraph flow:
    - load events
    - filter in-person meetings
    - resolve city fallback
    - fetch event-time weather
    - score risk deterministically
    - rewrite recommendations with structured LLM output + fallback

### Internal Security Model
- Server-to-server routes protected by `X-Internal-API-Key`
- Separate internal keys for voice and weather services
- `user_sub` propagated across tool calls for user-level scoping

---

## AI-Assisted Development Workflow (Condensed)

### Phase 1 - Scheduling + Tool Contracts
- Implemented strict request parsing for tool payloads (`toolCalls -> function.arguments`)
- Added validated schema for meeting creation fields
- Added `meeting_mode`, `city`, and `user_sub` handling
- Added robust extraction for `user_sub` from nested payload variants

### Phase 2 - Meetings Weather Summary
- Added `POST /meetings-weather-summary` in voice service
- Delegated weather reasoning to weather service internal API
- Added `/internal/meetings-weather-summary` pass-through route
- Added API-key checks and fallback error handling

### Phase 3 - Weather Graph Evolution
- Added event-time weather fetch (hourly selection nearest event timestamp)
- Added deterministic risk scoring (`low/moderate/high/blocked/unknown`)
- Added high-risk mitigation branch
- Added structured LLM rewrite layer for recommendations
- Preserved deterministic fallback on model/provider failure

### Phase 4 - Reliability Hardening
- Added retry handling for transient LLM failures (503/timeouts)
- Added deadline-based skip path for rewrite when latency budget is low
- Fixed sync-in-async bottleneck by switching weather-summary fetch in voice service to async `httpx.AsyncClient`
- Stabilized internal timeout behavior across voice <-> weather callback chain

### Phase 5 - Auth + Setup UX
- Added setup/onboarding flow after OAuth
- Redirect logic:
  - incomplete profile -> `/setup`
  - complete profile -> `/assistant`
- Added fields for profile personalization:
  - role, commute_mode, risk_tolerance, ppe_required

### Phase 6 - RAG Foundation (Current)
- Added user knowledge seed corpus (messy, realistic-style notes)
- Implemented:
  - `apps/rag/loader.py`
  - `apps/rag/chunker.py`
  - `apps/rag/index.py`
  - `apps/rag/retriever.py`
- Added tests for loader/chunker/retriever behaviors
- Current state: retrieval foundation complete; embedding backend + graph wiring next

---

## Important Technical Decisions

1. **Deterministic risk classification + LLM explanation**
- Risk label must stay testable and stable in code
- LLM used for explanation and action phrasing, not risk label authority

2. **City source priority**
- Explicit event city first
- Profile default city fallback second
- If both missing -> blocked/unknown path

3. **Service boundary clarity**
- Voice handles auth/tool webhooks/calendar
- Weather handles graph reasoning and weather semantics

4. **Operational resiliency**
- Retry transient failures
- Timeout budget aware rewrite skip
- Keep deterministic fallback paths for every external dependency

---

## Test Evidence

Typical commands used:

```bash
python -m pytest -q
python -m pytest -q tests/test_graph.py
python -m pytest -q tests/test_api.py
python -m pytest -q tests/test_rag_loader.py
python -m pytest -q tests/test_rag_chunker.py
python -m pytest -q tests/test_rag_retriever.py
```

Results during latest iteration:
- Weather service tests passing end-to-end
- New RAG foundation tests passing

---

## Deployment Verification Pattern

Used `curl` checks to validate each layer:

1. Health checks
```bash
curl https://<voice-service>/health
curl https://<weather-service>/health
```

2. Internal weather summary
```bash
curl -H "X-Internal-API-Key: <WEATHER_INTERNAL_API_KEY>" \
  "https://<weather-service>/internal/meeting-weather-summary?user_sub=<SUB>&date=<YYYY-MM-DD>&tz=Europe/Berlin"
```

3. Voice meetings summary tool endpoint
```bash
curl -X POST "https://<voice-service>/meetings-weather-summary" \
  -H "Content-Type: application/json" \
  -H "X-Internal-API-Key: <VOICE_INTERNAL_API_KEY>" \
  -d '{"message":{"toolCalls":[{"id":"tc-summary","function":{"arguments":{"user_sub":"<SUB>","date":"<YYYY-MM-DD>","timezone":"Europe/Berlin"}}}]}}'
```

---

## What I Learned / Engineering Approach
- Keep API contracts explicit and version-stable
- Keep domain-critical logic deterministic first
- Add LLM where personalization adds value, with fallback guarantees
- Design internal endpoints as product-grade integration surfaces
- Use logs + direct `curl` probes to isolate failures layer-by-layer quickly

---

## Links 
- Voice repo: https://github.com/harmonicsvd/voice-scheduling-agent
- Weather repo: https://github.com/harmonicsvd/ai-weather-agent
- Optional demo/log artifact: https://github.com/harmonicsvd/ai-weather-agent/blob/main/AI_CODING_TRANSCRIPT.md
