"""Pydantic schemas shared across weather tools, graph, and API boundaries."""

from pydantic import BaseModel, Field
from typing import Literal



class LocationSchema(BaseModel):
    """Canonical location shape used by tools and agent responses."""

    name: str
    latitude: float
    longitude: float
    country: str | None = None


class CurrentWeatherSchema(BaseModel):
    """Subset of current-weather metrics returned from Open-Meteo."""

    # Optional fields keep validation flexible if provider omits some values.
    temperature_c: float | None = Field(default=None)
    apparent_temperature_c: float | None = Field(default=None)
    humidity_percent: int | None = Field(default=None)
    wind_speed_kmh: float | None = Field(default=None)
    weather_code: int | None = Field(default=None)
    observation_time: str | None = Field(default=None)


class WeatherByCityResponseSchema(BaseModel):
    """Top-level contract: resolved location + its current weather payload."""

    location: LocationSchema
    current_weather: CurrentWeatherSchema
    
class CalendarEventSchema(BaseModel):
    """Normalized calendar event fields consumed by graph nodes."""
    title: str
    start: str | None = None
    end: str | None = None
    location: str | None = None
    is_virtual: bool = False
    meeting_mode: str = "unknown"
    city: str | None = None
    city_source: str | None = None
    user_sub: str | None = None


class CalendarEventsResponseSchema(BaseModel):
    """Response envelope for calendar list endpoint."""
    events: list[CalendarEventSchema] = Field(default_factory=list)


class LLMEventRecommendationSchema(BaseModel):
    """One event-level explanation item produced by structured LLM output."""
    event_title: str
    risk: Literal["low", "moderate", "high", "blocked", "unknown"]
    reason: str
    actions: list[str] = Field(default_factory=list)


class LLMRecommendationsResponseSchema(BaseModel):
    """Container of structured recommendations returned by LLM rewrite node."""
    recommendations: list[LLMEventRecommendationSchema] = Field(default_factory=list)
