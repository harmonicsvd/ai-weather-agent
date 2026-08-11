from __future__ import annotations

"""LangGraph node implementations for weather and meeting workflows."""

import re
import time
import math
import logging
import httpx
import os
from dotenv import load_dotenv
from pathlib import Path

logger = logging.getLogger(__name__)

# Node layer:
# - Each function reads a subset of GraphState and returns only keys it updates.
# - Keep network calls inside dedicated node/tool helpers to make tests easier.
from apps.graph.state import GraphState
from apps.tools.weather_client import (
    CityNotFoundError,
    OpenMeteoClient,
    WeatherProviderError,
)
from pydantic import ValidationError
from apps.tools.schemas import LLMRecommendationsResponseSchema


from datetime import datetime, timedelta, timezone

load_dotenv(dotenv_path=Path(__file__).resolve().parents[3] / ".env", override=False)

CALENDAR_API_BASE_URL = os.getenv("CALENDAR_API_BASE_URL", "http://127.0.0.1:8000")
CALENDAR_INTERNAL_API_KEY = os.getenv("CALENDAR_INTERNAL_API_KEY", "")

import json
from apps.tools.profile_client import ProfileClient, ProfileProviderError

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

from apps.rag.embeddings import GeminiEmbeddingProvider
from apps.rag.retriever import retrieve_user_context

LLM_REWRITE_MODEL = init_chat_model(
    "google_genai:gemini-2.5-flash",
    temperature=0,
).with_structured_output(LLMRecommendationsResponseSchema)


# Keep retries small so API latency stays within webhook limits.
LLM_REWRITE_MAX_ATTEMPTS = 2
LLM_REWRITE_BASE_SLEEP_SECONDS = 0.7


def _is_retryable_llm_error(exc: Exception) -> bool:
    """Decide whether an LLM failure is transient enough to try again."""
    msg = str(exc).lower()
    retry_markers = (
        "503",
        "unavailable",
        "resource_exhausted",
        "timed out",
        "timeout",
        "deadline exceeded",
    )
    return any(marker in msg for marker in retry_markers)


def _invoke_rewrite_model_with_retry(messages):
    """Call the rewrite model with small bounded retries for transient failures."""
    last_exc: Exception | None = None

    for attempt in range(1, LLM_REWRITE_MAX_ATTEMPTS + 1):
        try:
            return LLM_REWRITE_MODEL.invoke(messages)
        except Exception as exc:
            last_exc = exc
            should_retry = _is_retryable_llm_error(exc)
            is_last = attempt == LLM_REWRITE_MAX_ATTEMPTS

            if (not should_retry) or is_last:
                raise

            wait_s = LLM_REWRITE_BASE_SLEEP_SECONDS * (2 ** (attempt - 1))
            print(
                f"LLM rewrite transient failure (attempt {attempt}/{LLM_REWRITE_MAX_ATTEMPTS}); "
                f"retrying in {wait_s:.1f}s. error={repr(exc)}"
            )
            time.sleep(wait_s)

    raise last_exc or RuntimeError("Unknown LLM rewrite failure")



def _build_system_prompt(user_profile: dict | None) -> str:
    """
    Build LLM system instructions for recommendation rewriting.

    We keep risk classification deterministic (already scored) and ask the model
    only for explanation + actions, optionally personalized with profile fields.
    """
    base = (
        "You are an intelligent weather risk explainer. Your job is to take deterministic "
        "weather risk scores and rewrite them into personalized, actionable advice.\n\n"
        "SOURCE OF TRUTH: Risk levels (low/moderate/high/blocked/unknown) are final—do not change them.\n"
        "YOUR ROLE: Explain WHY each risk matters and suggest PRACTICAL ACTIONS.\n\n"
        "Output Rules:\n"
        "- Return strictly structured output matching the schema.\n"
        "- Keep risk labels exactly as: low, moderate, high, blocked, unknown.\n"
        "- Do not invent events or weather data.\n"
        "- Keep reasons short, specific, and actionable."
        "\n\nIf retrieved_context is provided, use it to make recommendations more specific.\n"
        "Do not hallucinate beyond retrieved snippets.\n"
        "If retrieved_context is empty, continue with weather + profile only."
    )

    if not user_profile:
        return base

    profile_context = (
        "\n\nUser Context (use this to personalize recommendations):\n"
    )
    
    role = user_profile.get("role")
    if role:
        profile_context += f"- Job/Role: {role}\n"
    
    commute_mode = user_profile.get("commute_mode")
    if commute_mode:
        profile_context += f"- How they commute: {commute_mode}\n"
    
    risk_tolerance = user_profile.get("risk_tolerance")
    if risk_tolerance:
        profile_context += f"- Risk tolerance: {risk_tolerance}\n"
    
    ppe_required = user_profile.get("ppe_required")
    if ppe_required:
        profile_context += f"- Requires protective equipment: Yes\n"

    profile_context += (
        "\nThink: How would weather impact THIS person given their job, commute, and risk preferences? "
        "What actions would make sense for them specifically?"
    )
    
    return base + profile_context

def _build_llm_rewrite_messages(
    risk_summary: list[dict],
    fallback: list[str],
    user_profile: dict | None = None,
    retrieved_context: list[dict] | None = None,
) -> list:

    """Package deterministic state into chat messages for structured LLM output."""
    system = SystemMessage(content=_build_system_prompt(user_profile))

    human = HumanMessage(
        content=json.dumps(
            {
                "risk_summary": risk_summary,
                "existing_recommendations": fallback,
                "user_profile": user_profile or {},
                "retrieved_context": retrieved_context or [],

            },
            ensure_ascii=False,
        )
    )
    return [system, human]



def apply_user_default_city(state: GraphState) -> GraphState:
    """
    Ensure every in-person event has a weather city when possible.

    Priority:
    1) user_profile.work_environment (for field work/construction sites)
    2) leave missing so downstream can mark event as blocked
    """
    events = state.get("in_person_events") or []

    user_profile = state.get("user_profile") or {}
    # Use work_environment as city hint for field work
    profile_city = None
    if user_profile.get("work_environment") in ["Field work", "Construction site"]:
        profile_city = "unknown"  # Will be handled by user input

    updated = []
    for event in events:
        if event.get("city"):
            updated.append(event)
            continue

        city = None
        city_source = "missing"

        if profile_city:
            city = profile_city
            city_source = "profile_api"

        updated.append({**event, "city": city, "city_source": city_source})

    return {"in_person_events": updated}



# Minimal pattern for phrases like: "weather in Hamburg?"
CITY_PATTERN = re.compile(r"\bin\s+([A-Za-z\s\-']+)\??$", re.IGNORECASE)


def route_intent(state: GraphState) -> GraphState:
    """
    Classify whether the query is weather-related and extract city when explicit.
    """
    # state.get(...) keeps this robust even if a key is missing from initial state.
    query = (state.get("user_query") or "").strip()
    q = query.lower()

    # any(...) returns True if at least one weather keyword appears in the query.
    is_weather = any(word in q for word in ["weather", "temperature", "forecast", "rain", "wind"])
    city = None

    # If user already provided city text, capture it here to skip fallback lookup.
    match = CITY_PATTERN.search(query)
    if match:
        city = match.group(1).strip()

    return {
        "intent": "weather" if is_weather else "other",
        "city": city,
        "error": None,
    }


def resolve_location(state: GraphState) -> GraphState:
    """
    Fill city only when missing.
    This mirrors our Phase 1 behavior: user_id-driven fallback location.
    """
    # Returning {} means "no state change" for this node.
    if state.get("city"):
        return {}

    user_id = state.get("user_id")
    fallback_city = "Florida" if user_id == "1" else "San Francisco"
    return {"city": fallback_city}


def fetch_weather(state: GraphState) -> GraphState:
    """
    Execute the external weather call and normalize failures into state.error.
    """
    if state.get("intent") != "weather":
        return {"error": "This workflow currently supports weather queries only."}

    city = state.get("city")
    if not city:
        return {"error": "No location available to fetch weather."}

    try:
        # Context manager guarantees connection cleanup after the call.
        with OpenMeteoClient() as client:
            weather = client.get_current_weather_by_city(city)
        # Return only the fields this node owns/updates.
        return {"weather": weather, "error": None}
    except CityNotFoundError:
        return {"error": f"City '{city}' was not found."}
    except WeatherProviderError:
        return {"error": "Weather provider is currently unavailable."}


def format_response(state: GraphState) -> GraphState:
    """
    Build the user-facing final response from either error or weather data.
    """
    # Error branch has highest priority so user gets a deterministic failure message.
    if state.get("error"):
        return {"final_response": state["error"]}

    # Meeting-preview branch should run before weather-specific branch.
    in_person_events = state.get("in_person_events")
    if in_person_events is not None:
        return {
            "final_response": f"Found {len(in_person_events)} in-person meetings to evaluate for weather."
        }

    weather = state.get("weather")
    if weather is None:
        return {"final_response": "No weather data available."}

    current = weather.current_weather
    location = weather.location
    location_label = f"{location.name}, {location.country}" if location.country else location.name

    # Build final response from typed schema attributes (not raw dict access).
    return {
        "final_response": (
            f"Current weather in {location_label}: "
            f"{current.temperature_c}°C (feels like {current.apparent_temperature_c}°C), "
            f"humidity {current.humidity_percent}%, wind {current.wind_speed_kmh} km/h."
        )
    }

def load_calendar_events(state: GraphState) -> GraphState:
    """
    Load meetings from Google Calendar API for today + next day.
    We intentionally start at day-begin (UTC) so "today" runs still include
    meetings that already happened earlier in the same day.
    """
    logger.info("🔍 load_calendar_events: starting")
    from_iso = state.get("from_iso")
    to_iso = state.get("to_iso")
    user_sub = state.get("user_sub")

    if not from_iso or not to_iso:
        now_utc = datetime.now(timezone.utc)
        day_start_utc = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
        from_iso = day_start_utc.isoformat().replace("+00:00", "Z")
        to_iso = (day_start_utc + timedelta(days=2)).isoformat().replace("+00:00", "Z")

    if not user_sub:
        logger.error("🔍 load_calendar_events: user_sub not provided")
        return {"events": [], "error": "User ID not provided."}

    # Fetch user's profile to get Google refresh token
    from apps.tools.profile_client import ProfileClient, ProfileProviderError

    try:
        with ProfileClient() as profile_client:
            profile = profile_client.get_profile_by_sub(user_sub)
            if not profile:
                logger.error(f"🔍 load_calendar_events: User profile not found for {user_sub}")
                return {"events": [], "error": "User profile not found. Please authenticate with Google Calendar first."}
            refresh_token = profile.google_refresh_token
            if not refresh_token:
                logger.error(f"🔍 load_calendar_events: Google Calendar not authenticated for {user_sub}")
                return {"events": [], "error": "Google Calendar not authenticated. Please authenticate with Google Calendar first."}
    except ProfileProviderError as e:
        logger.error(f"🔍 load_calendar_events: Failed to fetch user profile: {e}")
        return {"events": [], "error": f"Failed to fetch user profile: {str(e)}"}

    # Build Google Calendar service with user's OAuth credentials
    try:
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
        from google.auth.transport.requests import Request

        credentials = Credentials(
            token=None,
            refresh_token=refresh_token,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=GOOGLE_OAUTH_CLIENT_ID,
            client_secret=GOOGLE_OAUTH_CLIENT_SECRET,
            scopes=SCOPES,
        )
        # Refresh the token to get a valid access token
        credentials.refresh(Request())

        service = build("calendar", "v3", credentials=credentials)

        # Fetch events from Google Calendar API
        events_result = service.events().list(
            calendarId='primary',
            timeMin=from_iso,
            timeMax=to_iso,
            singleEvents=True,
            orderBy='startTime'
        ).execute()

        events = events_result.get('items', [])
        logger.info(f"🔍 load_calendar_events: loaded {len(events)} events from Google Calendar API")

    except Exception as e:
        logger.error(f"🔍 load_calendar_events: Failed to fetch events from Google Calendar API: {e}")
        return {"events": [], "error": f"Failed to fetch calendar events: {str(e)}"}

    # Normalize Google Calendar events to match expected format
    normalized = []
    for event in events:
        start = event.get("start", {}).get("dateTime", "")
        end = event.get("end", {}).get("dateTime", "")
        location = event.get("location", "")

        # Determine meeting mode based on location
        meeting_mode = "online"
        is_virtual = True
        if location and location.lower() not in ["", "online", "zoom", "meet", "teams"]:
            meeting_mode = "in_person"
            is_virtual = False

        normalized.append(
            {
                "title": event.get("summary", "No title"),
                "time": start,
                "end": end,  # Include end time
                "location": location,
                "is_virtual": is_virtual,
                "meeting_mode": meeting_mode,
                "city": None,  # Will be filled by apply_user_default_city
                "city_source": None,
                "user_sub": user_sub,
            }
        )

    logger.info(f"🔍 load_calendar_events: returning {len(normalized)} events")
    return {"events": normalized, "error": None}

def filter_in_person_events(state: GraphState) -> GraphState:
    """Keep only in-person events for weather evaluation."""
    logger.info("🔍 filter_in_person_events: starting")
    events = state.get("events") or []
    in_person_events = []

    for event in events:
        mode = (event.get("meeting_mode") or "unknown").strip().lower()

        if mode == "in_person":
            in_person_events.append(event)
            continue

        if mode == "online":
            continue

        # Fallback for older events where meeting_mode may be missing.
        if not bool(event.get("is_virtual")):
            in_person_events.append(event)

    logger.info(f"🔍 filter_in_person_events: filtered to {len(in_person_events)} in-person events")
    return {"in_person_events": in_person_events}




def fetch_weather_for_events(state: GraphState) -> GraphState:
    """
    For each in-person event:
    - fetch weather near event time if city/time exist
    - otherwise attach explicit reason so scoring can classify blocked/unknown
    """
    logger.info("🔍 fetch_weather_for_events: starting")
    events = state.get("in_person_events") or []
    if not events:
        logger.info("🔍 fetch_weather_for_events: no in-person events, returning empty")
        return {"event_weather": []}

    results = []
    logger.info(f"🔍 fetch_weather_for_events: fetching weather for {len(events)} events")
    try:
        with OpenMeteoClient() as client:
            for event in events:
                city = event.get("city")
                if not city:
                    results.append({"event": event, "weather": None, "reason": "missing location"})
                    continue

                event_time = event.get("time")
                if not event_time:
                    results.append({"event": event, "weather": None, "reason": "missing event time"})
                    continue

                try:
                    weather = client.get_weather_by_city_at_iso(city, event_time)
                    results.append({"event": event, "weather": weather.model_dump()})
                except ValueError as e:
                    logger.warning(f"🔍 fetch_weather_for_events: ValueError for event: {e}")
                    results.append({"event": event, "weather": None, "reason": "invalid event time"})
                except (CityNotFoundError, WeatherProviderError) as e:
                    logger.warning(f"🔍 fetch_weather_for_events: Weather error for event: {e}")
                    results.append({"event": event, "weather": None, "reason": "weather unavailable"})
    except Exception as e:
        logger.error(f"🔍 fetch_weather_for_events: Unexpected error: {e}")
        return {"event_weather": [], "error": str(e)}

    logger.info(f"🔍 fetch_weather_for_events: completed with {len(results)} results")
    return {"event_weather": results}


def score_event_weather_risk(state: GraphState) -> GraphState:
    """
    Score weather risk for each event.
    - blocked: missing meeting location
    - unknown: weather unavailable for other reasons
    - low/moderate/high: based on weather code + wind speed
    """
    logger.info("🔍 score_event_weather_risk: starting")
    event_weather = state.get("event_weather") or []
    risk_summary = []
    recommendations = []

    for item in event_weather:
        event = item["event"]
        weather = item["weather"]

        if weather is None:
            reason = item.get("reason") or "weather unavailable"

            if reason in {"missing location", "missing event time", "invalid event time"}:
                risk_summary.append(
                    {
                        "event_title": event.get("title"),
                        "city": event.get("city"),
                        "risk": "blocked",
                        "reason": reason,
                    }
                )

                if reason == "missing location":
                    recommendations.append(
                        f"{event.get('title')}: Add meeting location/city to evaluate weather risk."
                    )
                elif reason == "missing event time":
                    recommendations.append(
                        f"{event.get('title')}: Meeting time is missing; cannot evaluate event-time weather risk."
                    )
                else:  # invalid event time
                    recommendations.append(
                        f"{event.get('title')}: Meeting time format is invalid; cannot evaluate event-time weather risk."
                    )
            else:
                risk_summary.append(
                    {
                        "event_title": event.get("title"),
                        "city": event.get("city"),
                        "risk": "unknown",
                        "reason": reason,
                    }
                )
                recommendations.append(
                    f"Could not fetch weather for event '{event.get('title')}' in {event.get('city')}."
                )
            continue


        current = weather.get("current_weather", {})
        weather_code = current.get("weather_code") or 0
        wind_speed = current.get("wind_speed_kmh") or 0.0
        temperature = current.get("temperature_c")

        if weather_code >= 80 or wind_speed >= 35:
            risk = "high"
        elif weather_code >= 60 or wind_speed >= 20:
            risk = "moderate"
        else:
            risk = "low"

        risk_summary.append(
            {
                "event_title": event.get("title"),
                "city": event.get("city"),
                "risk": risk,
                "weather_code": weather_code,
                "wind_speed_kmh": wind_speed,
                "temperature_c": temperature,
            }
        )

        if risk == "high":
            recommendations.append(
                f"{event['title']} ({event.get('city')}): high weather risk. Consider rescheduling or leaving early."
            )
        elif risk == "moderate":
            recommendations.append(
                f"{event['title']} ({event.get('city')}): moderate weather risk. Plan extra commute buffer."
            )
        else:
            recommendations.append(
                f"{event['title']} ({event.get('city')}): low weather risk."
            )

    return {"risk_summary": risk_summary, "recommendations": recommendations}

def format_meeting_recommendations(state: GraphState) -> GraphState:
    """
    Format the final response to include weather risk recommendations for meetings.
    """
    recommendations = state.get("recommendations") or []
    retrieved_context = state.get("retrieved_context") or []

    if not recommendations:
        return {"final_response": "No in-person meetings found or no weather data available."}

    formatted_response = "Weather Risk Recommendations for Your Meetings:\n" + "\n".join(recommendations)

    if retrieved_context:
        seen = set()
        source_lines = []
        for item in retrieved_context:
            source = (item.get("source_file") or "unknown").strip()
            if source in seen:
                continue
            seen.add(source)
            source_lines.append(f"- {source}")

        if source_lines:
            formatted_response += "\n\nContext Sources:\n" + "\n".join(source_lines)

    return {"final_response": formatted_response}


def format_all_meetings(state: GraphState) -> GraphState:
    """
    Format all meetings (online and in-person) with weather recommendations only for in-person.
    Weather recommendations are added as internal smartness - only shown when relevant.
    """
    all_events = state.get("events") or []
    recommendations = state.get("recommendations") or []
    retrieved_context = state.get("retrieved_context") or []
    error = state.get("error")

    # Check for calendar errors first
    if error:
        return {"final_response": f"Unable to access your calendar: {error}"}

    if not all_events:
        return {"final_response": "No meetings found for this date."}

    # Build meeting list
    meeting_lines = []
    for event in all_events:
        title = event.get("summary", "No title")
        time = event.get("start", {}).get("dateTime", "No time")
        meeting_mode = event.get("extendedProperties", {}).get("private", {}).get("meeting_mode", "unknown")
        
        # Format time nicely
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(time.replace("Z", "+00:00"))
            time_str = dt.strftime("%I:%M %p")
        except:
            time_str = time

        meeting_lines.append(f"- {title} at {time_str} ({meeting_mode})")

    formatted_response = "Your meetings today:\n" + "\n".join(meeting_lines)

    # Add weather recommendations only if there are in-person meetings with recommendations
    if recommendations:
        in_person_count = sum(1 for e in all_events if e.get("extendedProperties", {}).get("private", {}).get("meeting_mode") == "in_person")
        if in_person_count > 0:
            formatted_response += "\n\nWeather recommendations for in-person meetings:\n" + "\n".join(recommendations)

    if retrieved_context:
        seen = set()
        source_lines = []
        for item in retrieved_context:
            source = (item.get("source_file") or "unknown").strip()
            if source in seen:
                continue
            seen.add(source)
            source_lines.append(f"- {source}")

        if source_lines:
            formatted_response += "\n\nContext Sources:\n" + "\n".join(source_lines)

    return {"final_response": formatted_response}



def add_high_risk_actions(state: GraphState) -> GraphState:
    """
    Add specific action recommendations for high-risk events.
    """
    risk_summary = state.get("risk_summary") or []
    recommendations= list(state.get("recommendations") or [])
    
    
    high_risk_items=[item for item in risk_summary if item.get("risk")=="high"]
    if not high_risk_items:
        return{"recommendations": recommendations}    
    
    recommendations.append(
        "High-risk travel guidance: leave at least 30 minutes early, check transit disruptions, and carry weather protection."
    )
    
    return {"recommendations": recommendations}



def llm_recommendation_rewrite(state: GraphState) -> GraphState:
    """
    Explanation layer:
    use LLM to rewrite deterministic risk_summary into concise recommendations.
    Falls back to existing recommendations on any failure.
    """
    logger.info("🔍 llm_recommendation_rewrite: starting")
    risk_summary = state.get("risk_summary") or []
    fallback = list(state.get("recommendations") or [])
    user_profile = state.get("user_profile")
    retrieved_context = state.get("retrieved_context") or []

    if not risk_summary:
        logger.info("🔍 llm_recommendation_rewrite: no risk_summary, using fallback")
        return {"recommendations": fallback}

    if _should_skip_llm_rewrite_for_timeout(state):
        logger.warning(f"🔍 llm_recommendation_rewrite: skipping due to timeout, risk_items={len(risk_summary)}")
        print(
            "Skipping LLM rewrite due to timeout pressure; "
            f"risk_items={len(risk_summary)}"
        )
        return {"recommendations": fallback}


    try:
        logger.info("🔍 llm_recommendation_rewrite: building LLM messages")
        messages = _build_llm_rewrite_messages(risk_summary, fallback, user_profile, retrieved_context)
        logger.info("🔍 llm_recommendation_rewrite: invoking LLM with retry")
        validated = _invoke_rewrite_model_with_retry(messages)
        logger.info("🔍 llm_recommendation_rewrite: LLM invocation successful")


        rewritten = []
        for rec in validated.recommendations:
            reason = (rec.reason or "").strip().rstrip(".")
            actions = [(a or "").strip().rstrip(".") for a in (rec.actions or []) if (a or "").strip()]

            line = f"{rec.event_title}: {rec.risk} risk."
            if reason:
                line += f" Reason: {reason}."
            if actions:
                line += " Actions: " + "; ".join(actions) + "."
            rewritten.append(line)

        if not rewritten:
            logger.warning(f"🔍 llm_recommendation_rewrite: LLM returned zero recommendations, using fallback, risk_items={len(risk_summary)}")
            print(
                "LLM rewrite returned zero recommendations; using fallback. "
                f"risk_items={len(risk_summary)}"
            )
            
        # Preserve deterministic high-risk safety guidance from previous node so
        # critical safety advice is never dropped by rewrite formatting.
        carry_over = [
            line for line in fallback
            if "High-risk travel guidance" in line
        ]
        for line in carry_over:
            if line not in rewritten:
                rewritten.append(line)

        return {"recommendations": rewritten or fallback}
    except ValidationError as exc:
        logger.error(f"🔍 llm_recommendation_rewrite: ValidationError: {exc}")
        print(f"LLM rewrite validation failed; using fallback. details={exc}")
        return {"recommendations": fallback}
    except Exception as exc:
        logger.error(f"🔍 llm_recommendation_rewrite: Exception: {exc}")
        print(f"LLM rewrite failed; using fallback. error={repr(exc)}")
        return {"recommendations": fallback}


# Load profile data for personalization and city fallback logic.
def load_user_profile(state: GraphState) -> GraphState:
    """
    Fetch user profile once and store in graph state.

    Downstream nodes use this for:
    - default city fallback (`apply_user_default_city`)
    - personalized recommendation prompt context
    """
    user_sub = state.get("user_sub")
    if not user_sub:
        return {"user_profile": None}

    try:
        with ProfileClient() as profile_client:
            profile = profile_client.get_profile_by_sub(user_sub)
        if profile:
            return {"user_profile": profile.model_dump()}
        return {"user_profile": None}
    except ProfileProviderError as exc:
        # keep fallback behavior, but make failure visible in logs
        print(f"Profile load failed for sub={user_sub}: {exc}")
        return {"user_profile": None}


def _should_skip_llm_rewrite_for_timeout(state: GraphState) -> bool:
    """
    Decide whether the LLM rewrite should be skipped to protect endpoint latency.

    If we do not have a deadline configured, we allow the rewrite.
    If remaining time is below the configured minimum buffer, we skip it and
    keep deterministic recommendations.
    """
    deadline = state.get("llm_deadline_monotonic")
    min_remaining = state.get("llm_min_time_remaining_seconds")

    if deadline is None or min_remaining is None:
        return False

    remaining = deadline - time.monotonic()

    # Defensive guard: invalid values should not break the request path.
    if not math.isfinite(remaining):
        return False

    return remaining < min_remaining



def retrieve_meeting_context(state: GraphState) -> GraphState:
    """
    Retrieve user-specific knowledge snippets for in-person meetings.

    Retrieval is best-effort:
    - if config/model/docs fail, return empty context
    - downstream recommendation logic must still work without retrieval
    """
    events = state.get("in_person_events") or []
    user_sub = (state.get("user_sub") or "").strip()

    if not events or not user_sub:
        return {"retrieved_context": [], "retrieval_query": None}

    query_parts: list[str] = []
    for event in events[:3]:
        title = (event.get("title") or "").strip()
        city = (event.get("city") or "").strip()
        location = (event.get("location") or "").strip()

        piece = " ".join(part for part in [title, city, location] if part)
        if piece:
            query_parts.append(piece)

    retrieval_query = " | ".join(query_parts).strip()
    if not retrieval_query:
        return {"retrieved_context": [], "retrieval_query": None}

    try:
        provider = GeminiEmbeddingProvider()

        # Keep legacy flat context (for backward compatibility),
        # and add event-level attribution.
        flat_context: list[dict] = []
        context_by_event: dict[str, list[dict]] = {}

        for event in events:
            event_title = (event.get("title") or "Untitled Event").strip()
            city = (event.get("city") or "").strip()
            location = (event.get("location") or "").strip()

            event_query = " ".join(part for part in [event_title, city, location] if part).strip()
            if not event_query:
                context_by_event[event_title] = []
                continue

            event_hits = retrieve_user_context(
                user_sub=user_sub,
                query_text=event_query,
                embedding_provider=provider,
                top_k=2,
            )
            print(f"RAG hits for '{event_title}': {len(event_hits)}")
            event_context = [
                {
                    "text": h.chunk.text,
                    "score": h.score,
                    "source_file": h.chunk.metadata.get("source_file"),
                    "event_title": event_title,
                }
                for h in event_hits
            ]

            context_by_event[event_title] = event_context
            flat_context.extend(event_context)

        return {
            "retrieved_context": flat_context,
            "retrieved_context_by_event": context_by_event,
            "retrieval_query": retrieval_query,
        }
    except Exception as exc:
        print(f"RAG retrieval failed; continuing without context. error={repr(exc)}")
        return {
    "retrieved_context": [],
    "retrieved_context_by_event": {},
    "retrieval_query": retrieval_query,
}
        
def add_retrieved_context_recommendations(state: GraphState) -> GraphState:
    recommendations = list(state.get("recommendations") or [])
    context_by_event = state.get("retrieved_context_by_event") or {}

    for event_title, contexts in context_by_event.items():
        if not contexts:
            continue

        best_context = contexts[0]
        source = best_context.get("source_file") or "uploaded document"

        recommendations.append(
            f"{event_title}: use context from {source} when preparing for this meeting."
        )

    return {"recommendations": recommendations}       