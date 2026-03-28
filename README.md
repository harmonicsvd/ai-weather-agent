# AI Weather Agent

An AI weather assistant powered by LangChain tool-calling and live weather data from Open-Meteo.

## Overview
This project provides a weather-focused agent that:
- Interprets natural language weather requests
- Resolves location context when needed
- Fetches real-time weather via geocoding + forecast APIs
- Returns structured, user-friendly responses
- Handles invalid locations and provider failures safely

## Key Features
- LangChain agent with tool integration
- Open-Meteo API client with retries and timeouts
- Pydantic schemas for typed response contracts
- Error handling for city lookup and provider availability
- Automated tests for core weather client flows

## Model
- Provider: Google GenAI
- Model: `gemini-2.5-flash`
- Integration: LangChain `init_chat_model("google_genai:gemini-2.5-flash")`

## Repository Structure
- `main.py`: agent setup, tools, and local demo execution
- `apps/tools/weather_client.py`: geocoding and weather client
- `apps/tools/schemas.py`: Pydantic models for typed outputs
- `tests/test_weather_client.py`: unit tests for client behavior

## Requirements
- Python 3.11+
- Installed dependencies for:
  - `langchain`
  - `langgraph`
  - `python-dotenv`
  - `httpx`
  - `tenacity`
  - `pydantic`
  - `pytest`

## Setup
1. Move into the project directory:
   ```bash
   cd weather-agent
   ```
2. Configure environment variables in `.env` (model provider credentials).

## Run
```bash
python main.py
```

## Test
```bash
python -m pytest -q
```
