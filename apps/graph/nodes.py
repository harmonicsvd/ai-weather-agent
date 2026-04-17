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

from pathlib import Path
import json
from apps.tools.profile_client import ProfileClient, ProfileProviderError

USER_PROFILES_PATH = Path(__file__).resolve().parents[2] / "data" / "user_profiles.json"

def _get_user_default_city(user_id: str | None) -> str | None:
    if not user_id:
        return None
    try:
        profiles = json.loads(USER_PROFILES_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return (profiles.get(str(user_id)) or {}).get("default_city")


def apply_user_default_city(state: GraphState) -> GraphState:
    events = state.get("in_person_events") or []
    local_default_city = _get_user_default_city(state.get("user_id"))
    updated = []

    with ProfileClient() as profile_client:
        for event in events:
            if event.get("city"):
                updated.append(event)
                continue

            city = None
            city_source = "missing"

            user_sub = event.get("user_sub")
            if user_sub:
                try:
                    profile = profile_client.get_profile_by_sub(user_sub)
                    if profile and profile.default_city:
                        city = profile.default_city
                        city_source = "profile_api"
                except ProfileProviderError:
                    pass

            if not city and local_default_city:
                city = local_default_city
                city_source = "user_default"

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