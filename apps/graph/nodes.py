from __future__ import annotations

import re

from apps.graph.state import GraphState
from apps.tools.weather_client import (
    CityNotFoundError,
    OpenMeteoClient,
    WeatherProviderError,
)

from datetime import datetime, timedelta, timezone

from apps.tools.calendar_client import CalendarClient, CalendarProviderError


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
    Load meetings from calendar API for the next 2 days.
    """
    now = datetime.now(timezone.utc)
    from_iso = now.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    to_iso = (now + timedelta(days=2)).replace(microsecond=0).isoformat().replace("+00:00", "Z")

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
                # For now use location as city when available.
                # Later we can add geocoding parser.
                "city": event.location,
            }
        )

    return {"events": normalized, "error": None}


def filter_in_person_events(state: GraphState) -> GraphState:
    events = state.get("events") or []
    in_person_events = []
    for event in events:
        is_virtual = bool(event.get("is_virtual"))
        location_text = (event.get("location") or "").lower()
        if not is_virtual and location_text != "zoom":
            in_person_events.append(event)
    return {"in_person_events": in_person_events}



def fetch_weather_for_events(state: GraphState) -> GraphState:
    events = state.get("in_person_events") or []
    if not events:
        return {"event_weather": []}

    results = []
    with OpenMeteoClient() as client:
        for event in events:
            city = event.get("city")
            if not city:
                results.append({"event": event, "weather": None})
                continue

            try:
                weather = client.get_current_weather_by_city(city)
                # Store plain dict in graph state so checkpoint serialization
                # does not depend on custom class reconstruction.
                results.append({"event": event, "weather": weather.model_dump()})
            except (CityNotFoundError, WeatherProviderError):
                results.append({"event": event, "weather": None})
    return {"event_weather": results}


def score_event_weather_risk(state: GraphState) -> GraphState:
    """
    Simple heuristic to score weather risk for each event.
    This is a placeholder for more complex logic or an LLM-based evaluator.
    """
    event_weather = state.get("event_weather") or []
    risk_summary = []
    recommendations = []

    for item in event_weather:
        event = item["event"]
        weather = item["weather"]
        if weather is None:
            risk_summary.append(
                {
                    "event_title": event.get("title"),
                    "city": event.get("city"),
                    "risk": "unknown",
                    "reason": "weather unavailable",
                }
            )
            recommendations.append(
                f"Could not fetch weather for event '{event['title']}' in {event.get('city')}."
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
