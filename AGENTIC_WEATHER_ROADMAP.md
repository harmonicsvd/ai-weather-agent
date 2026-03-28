# Agentic Weather Project Roadmap (Reference)

## Goal
Extend the current basic weather agent into a real-world learning project that teaches:
- LangChain agent foundations
- LangGraph orchestration and durable workflows
- RAG pipelines
- MCP integration
- Multi-agent system design
- Evaluation and production readiness

## Current Baseline
Current implementation in `weather-agent/main.py` includes:
- Basic `create_agent(...)` setup
- Two simple tools (`get_weather_for_location`, `get_user_location`)
- Context passing with `ToolRuntime`
- Structured output with `ToolStrategy`
- In-memory checkpointing via `InMemorySaver`

## Expansion Roadmap
1. LangChain Foundation
- Replace mock weather with real provider APIs.
- Add typed tool schemas, retries, and robust error handling.
- Standardize prompt contracts and output schema.

2. LangGraph Orchestration
- Move from linear `invoke()` calls to a graph-based workflow.
- Add nodes such as: intent detection, location resolution, retrieval decision, tool execution, response synthesis.
- Add conditional edges and persistent checkpointer.

3. RAG Layer
- Build knowledge ingestion for weather docs (safety guidance, advisories, policy docs).
- Chunk, embed, and store in a vector store.
- Expose retrieval as a tool and enforce citation-style grounded responses.

4. MCP Integration
- Build/consume MCP tools for external capabilities.
- Integrate LangChain agent with MCP client patterns.
- Add tool-approval and safe execution boundaries.

5. Agentic System Design
- Split responsibilities into specialized agents (planner, analyst, risk explainer, responder).
- Coordinate via LangGraph supervisor/worker patterns.
- Add fallback and escalation rules across agents.

6. Evaluation + Observability
- Enable LangSmith tracing.
- Build dataset-driven evaluations for tool usage, retrieval quality, and final response quality.
- Add regression tests to prevent behavior drift.

7. Production Shape
- Serve through an API layer.
- Add authentication, caching, background jobs, and deployment config.
- Move from in-memory to persistent stores for checkpoints and retrieval data.

## Suggested Repo Structure
```
weather-agent/
  app/
    graph/      # state, nodes, edges, orchestration
    tools/      # weather tools, mcp adapters, typed tool I/O
    rag/        # ingestion, embeddings, vector store, retriever
    api/        # FastAPI endpoints, request/response contracts
  tests/        # unit, integration, eval harness tests
  main.py       # local runner / entrypoint
```

## Official Docs (Primary References)
1. LangGraph Graph API  
   https://docs.langchain.com/oss/python/langgraph/graph-api
2. LangGraph Persistence  
   https://docs.langchain.com/oss/python/langgraph/persistence
3. LangGraph Durable Execution  
   https://docs.langchain.com/oss/python/langgraph/durable-execution
4. LangChain `create_agent`  
   https://docs.langchain.com/oss/python/releases/langchain-v1#create_agent
5. LangChain Structured Output  
   https://docs.langchain.com/oss/python/langchain/structured-output
6. LangChain RAG Agent  
   https://docs.langchain.com/oss/python/langchain/rag
7. LangGraph Agentic RAG  
   https://docs.langchain.com/oss/python/langgraph/agentic-rag
8. LangChain MCP  
   https://docs.langchain.com/oss/python/langchain/mcp
9. LangSmith Evaluation  
   https://docs.langchain.com/langsmith/evaluation

## Milestones (Execution Order)
1. Real weather tools + strong schemas (Phase 1)
2. First LangGraph workflow with persistent state (Phase 2)
3. RAG ingestion + retriever tool (Phase 3)
4. MCP server/client integration (Phase 4)
5. Multi-agent orchestration (Phase 5)
6. Evals + observability + regression tests (Phase 6)
7. API + deployment hardening (Phase 7)

## Tracker Update Protocol
- We work in guidance mode first: you implement, I guide/review unless you explicitly ask me to code.
- I will propose tracker updates after each milestone or sub-task.
- I will update this tracker only after your explicit approval: `yes do it`.
- Without `yes do it`, status remains unchanged.

## Progress Tracker
| Phase | Task | Owner | Status | Evidence | Date |
|---|---|---|---|---|---|
| Phase 1: LangChain Foundation | Replace mock weather tool with real weather provider integration | You | Completed | Open-Meteo client integrated and used by agent tools | 2026-03-28 |
| Phase 1: LangChain Foundation | Add typed request/response schemas for tools | You | Completed | `LocationSchema`, `CurrentWeatherSchema`, `WeatherByCityResponseSchema` added and used | 2026-03-28 |
| Phase 1: LangChain Foundation | Add retries, timeout handling, and user-safe error messages | You | Completed | Tenacity retry + provider error mapping + city-not-found handling implemented | 2026-03-28 |
| Phase 1: LangChain Foundation | Define stable prompt + structured output contract | You | Completed | `SYSTEM_PROMPT` + `ResponseFormat` stable; manual runs and tests verified | 2026-03-28 |
| Phase 2: LangGraph Orchestration | Build first graph with state + conditional edges | You | Planned | Pending | Pending |
| Phase 2: LangGraph Orchestration | Add persistent checkpointer (non-memory) | You | Planned | Pending | Pending |
| Phase 3: RAG Layer | Build ingestion pipeline (load/chunk/embed/index) | You | Planned | Pending | Pending |
| Phase 3: RAG Layer | Add retriever tool + grounded response behavior | You | Planned | Pending | Pending |
| Phase 4: MCP Integration | Integrate at least one MCP tool/server | You | Planned | Pending | Pending |
| Phase 5: Agentic System Design | Add multi-agent coordination with clear roles | You | Planned | Pending | Pending |
| Phase 6: Evaluation + Observability | Enable LangSmith tracing + baseline eval dataset | You | Planned | Pending | Pending |
| Phase 7: Production Shape | Expose API + basic deployment-ready config | You | Planned | Pending | Pending |

Status legend:
- Planned: task identified, not started
- In Progress: actively being worked on
- Completed: done and verified
- Blocked: waiting on dependency/decision

## Change Log
- 2026-03-28: Added initial roadmap and learning references.
- 2026-03-28: Added confirmation-gated tracker template (`yes do it` required before updates).
- 2026-03-28: Phase 1 foundation marked completed with verification (`python -m pytest -q` -> `6 passed in 0.12s`).

## Note For Next Session
Start with Phase 1 implementation and refactor the current file layout toward `app/graph`, `app/tools`, and `tests`.
