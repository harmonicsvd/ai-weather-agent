from pydantic import BaseModel, Field


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
    events: list[CalendarEventSchema] = Field(default_factory=list)
