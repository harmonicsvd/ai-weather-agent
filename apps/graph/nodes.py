from __future__ import annotations

"""LangGraph node implementations for weather and meeting workflows."""

import re

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

from apps.tools.calendar_client import CalendarClient, CalendarProviderError

import json
from apps.tools.profile_client import ProfileClient, ProfileProviderError

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage


LLM_REWRITE_MODEL = init_chat_model(
    "google_genai:gemini-2.5-flash",
    temperature=0,
).with_structured_output(LLMRecommendationsResponseSchema)


def _build_system_prompt(user_profile: dict | None) -> str:
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
) -> list:
    system = SystemMessage(content=_build_system_prompt(user_profile))

    human = HumanMessage(
        content=json.dumps(
            {
                "risk_summary": risk_summary,
                "existing_recommendations": fallback,
                "user_profile": user_profile or {},
            },
            ensure_ascii=False,
        )
    )
    return [system, human]



def apply_user_default_city(state: GraphState) -> GraphState:
    """
    Ensure every in-person event has a weather city when possible.

    Priority:
    1) user_profile.default_city from internal profile API
    2) leave missing so downstream can mark event as blocked
    """
    events = state.get("in_person_events") or []

    user_profile = state.get("user_profile") or {}
    profile_default_city = (user_profile.get("default_city") or "").strip() or None

    updated = []
    for event in events:
        if event.get("city"):
            updated.append(event)
            continue

        city = None
        city_source = "missing"

        if profile_default_city:
            city = profile_default_city
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
    Load meetings from calendar API for today + next day.
    We intentionally start at day-begin (UTC) so "today" runs still include
    meetings that already happened earlier in the same day.
    """
    from_iso = state.get("from_iso")
    to_iso = state.get("to_iso")
    if not from_iso or not to_iso:
        now_utc = datetime.now(timezone.utc)
        day_start_utc = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
        from_iso = day_start_utc.isoformat().replace("+00:00", "Z")
        to_iso = (day_start_utc + timedelta(days=2)).isoformat().replace("+00:00", "Z")

    try:
        with CalendarClient() as client:
            events = client.list_events(from_iso, to_iso)
    except CalendarProviderError:
        return {"events": [], "error": "Calendar provider is currently unavailable."}

    normalized = []
    for event in events:
        normalized.append(
            {
                "title": event.title,
                "time": event.start,
                "location": event.location,
                "is_virtual": event.is_virtual,
                "meeting_mode": event.meeting_mode,
                # For now use location as city when available.
                # Later we can add geocoding parser.
                "city": event.city,
                "city_source": event.city_source,
                "user_sub": event.user_sub,
            }
        )

    requested_sub = (state.get("user_sub") or "").strip()
    if requested_sub:
        normalized = [event for event in normalized if event.get("user_sub") == requested_sub]

    return {"events": normalized, "error": None}

def filter_in_person_events(state: GraphState) -> GraphState:
    """Keep only in-person events for weather evaluation."""
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

    return {"in_person_events": in_person_events}




def fetch_weather_for_events(state: GraphState) -> GraphState:
    """
    For each in-person event:
    - fetch weather near event time if city/time exist
    - otherwise attach explicit reason so scoring can classify blocked/unknown
    """
    events = state.get("in_person_events") or []
    if not events:
        return {"event_weather": []}

    results = []
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
            except ValueError:
                results.append({"event": event, "weather": None, "reason": "invalid event time"})
            except (CityNotFoundError, WeatherProviderError):
                results.append({"event": event, "weather": None, "reason": "weather unavailable"})

    return {"event_weather": results}


def score_event_weather_risk(state: GraphState) -> GraphState:
    """
    Score weather risk for each event.
    - blocked: missing meeting location
    - unknown: weather unavailable for other reasons
    - low/moderate/high: based on weather code + wind speed
    """
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
    if not recommendations:
        return {"final_response": "No in-person meetings found or no weather data available."}

    formatted_response = "Weather Risk Recommendations for Your Meetings:\n" + "\n".join(recommendations)
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
    risk_summary = state.get("risk_summary") or []
    fallback = list(state.get("recommendations") or [])
    user_profile = state.get("user_profile")
    if not risk_summary:
        return {"recommendations": fallback}

    try:
        messages = _build_llm_rewrite_messages(risk_summary, fallback, user_profile)
        validated = LLM_REWRITE_MODEL.invoke(messages)

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
        print(f"LLM rewrite validation failed; using fallback. details={exc}")
        return {"recommendations": fallback}
    except Exception as exc:
        print(f"LLM rewrite failed; using fallback. error={repr(exc)}")
        return {"recommendations": fallback}


# Load profile data for personalization and city fallback logic.
def load_user_profile(state: GraphState) -> GraphState:
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
