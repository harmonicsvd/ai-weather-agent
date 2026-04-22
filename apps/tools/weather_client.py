from __future__ import annotations

"""Open-Meteo API client with retries, validation, and event-time weather lookup."""

from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from apps.tools.schemas import (
    CurrentWeatherSchema,
    LocationSchema,
    WeatherByCityResponseSchema,
)

from datetime import datetime, timezone


# Open-Meteo endpoints used by this client.
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


class CityNotFoundError(Exception):
    """Raised when geocoding cannot resolve the input city."""


class WeatherProviderError(Exception):
    """Raised for provider/network/response failures."""



class OpenMeteoClient:
    """Small wrapper around Open-Meteo APIs with retries and error mapping."""

    def __init__(self, timeout_seconds: float = 10.0) -> None:
        # Reuse one HTTP client for connection pooling and shared configuration.
        self._client = httpx.Client(
            timeout=timeout_seconds,
            headers={"User-Agent": "weather-agent/1.0"},
        )

    def close(self) -> None:
        """Release HTTP resources explicitly when done."""
        self._client.close()

    def __enter__(self) -> "OpenMeteoClient":
        # Enables: `with OpenMeteoClient() as client: ...`
        # Returning self gives caller access to all instance methods.
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # Always close underlying connections, even if an exception occurs.
        self.close()

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.TransportError)),
    )
    def _request_json(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        """Low-level request function with retry policy."""
        response = self._client.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def _get_json(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        """
        Convert lower-level HTTP errors into app-specific errors so callers
        only handle stable exceptions from this module.
        """
        try:
            return self._request_json(url, params)
        except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as exc:
            raise WeatherProviderError("Weather provider is unavailable. Please try again.") from exc
        except ValueError as exc:
            raise WeatherProviderError("Weather provider returned invalid data.") from exc

    def geocode_city(self, city: str) -> LocationSchema:
        """Resolve user city text into provider coordinates."""
        city = city.strip()
        if not city:
            raise ValueError("city must not be empty")

        data = self._get_json(
            GEOCODE_URL,
            {"name": city, "count": 1, "language": "en", "format": "json"},
        )

        results = data.get("results") or []
        if not results:
            raise CityNotFoundError(f"City '{city}' was not found.")

        top = results[0]
        return LocationSchema(
            name=top.get("name", city),
            latitude=top["latitude"],
            longitude=top["longitude"],
            country=top.get("country"),
        )

    def get_current_weather(self, lat: float, lon: float) -> CurrentWeatherSchema:
        """Fetch current weather values for a latitude/longitude pair."""
        data = self._get_json(
            FORECAST_URL,
            {
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,apparent_temperature,relative_humidity_2m,wind_speed_10m,weather_code",
                "timezone": "auto",
            },
        )

        current = data.get("current")
        if not current:
            raise WeatherProviderError("Current weather data is missing from provider response.")

        # Keep this shape aligned with CurrentWeatherSchema in schemas.py.
        return CurrentWeatherSchema(
        temperature_c=current.get("temperature_2m"),
        apparent_temperature_c=current.get("apparent_temperature"),
        humidity_percent=current.get("relative_humidity_2m"),
        wind_speed_kmh=current.get("wind_speed_10m"),
        weather_code=current.get("weather_code"),
        observation_time=current.get("time"),
    )

    def get_current_weather_by_city(self, city: str) -> WeatherByCityResponseSchema:
        """
        High-level helper used by tools/agents:
        city -> geocode -> weather -> combined response payload.
        """
        location = self.geocode_city(city)
        weather = self.get_current_weather(location.latitude, location.longitude)
        
        # Build the raw payload first, then validate/normalize with Pydantic.
        unvalidated_data = {
            "location": {
                "name": location.name,
                "country": location.country,
                "latitude": location.latitude,
                "longitude": location.longitude,
            },
            "current_weather": weather,
        }

        # model_validate ensures downstream code receives a typed, safe object.
        return WeatherByCityResponseSchema.model_validate(unvalidated_data)

    @staticmethod
    def _parse_iso_to_utc(value: str) -> datetime:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    
    def get_weather_at_iso(self, lat: float, lon: float, target_iso: str) -> CurrentWeatherSchema:
        """
        Fetch hourly weather and select the closest point to target datetime (UTC).
        We do not poll every hour ourselves; this picks one point from provider data.
        """
        target_utc = self._parse_iso_to_utc(target_iso)
        target_date = target_utc.date().isoformat()

        data = self._get_json(
            FORECAST_URL,
            {
                "latitude": lat,
                "longitude": lon,
                "hourly": "temperature_2m,apparent_temperature,relative_humidity_2m,wind_speed_10m,weather_code",
                "timezone": "UTC",
                "start_date": target_date,
                "end_date": target_date,
            },
        )

        hourly = data.get("hourly") or {}
        times = hourly.get("time") or []
        if not times:
            raise WeatherProviderError("Hourly weather data is missing from provider response.")

        closest_index: int | None = None
        closest_time_utc: datetime | None = None
        closest_diff_seconds: float | None = None

        for idx, raw_time in enumerate(times):
            try:
                point_utc = self._parse_iso_to_utc(raw_time)
            except ValueError:
                continue

            # Choose nearest forecast bucket to the meeting time.
            diff = abs((point_utc - target_utc).total_seconds())
            if closest_diff_seconds is None or diff < closest_diff_seconds:
                closest_diff_seconds = diff
                closest_index = idx
                closest_time_utc = point_utc

        if closest_index is None or closest_time_utc is None:
            raise WeatherProviderError("Hourly weather data is invalid in provider response.")

        def _value(key: str):
            series = hourly.get(key) or []
            return series[closest_index] if closest_index < len(series) else None

        return CurrentWeatherSchema(
            temperature_c=_value("temperature_2m"),
            apparent_temperature_c=_value("apparent_temperature"),
            humidity_percent=_value("relative_humidity_2m"),
            wind_speed_kmh=_value("wind_speed_10m"),
            weather_code=_value("weather_code"),
            observation_time=closest_time_utc.isoformat().replace("+00:00", "Z"),
        )
        
        
    def get_weather_by_city_at_iso(self, city: str, target_iso: str) -> WeatherByCityResponseSchema:
        """
        Resolve city and fetch hourly weather nearest to target datetime.
        Returned shape remains compatible with WeatherByCityResponseSchema.
        """
        location = self.geocode_city(city)
        weather = self.get_weather_at_iso(location.latitude, location.longitude, target_iso)

        unvalidated_data = {
            "location": {
                "name": location.name,
                "country": location.country,
                "latitude": location.latitude,
                "longitude": location.longitude,
            },
            "current_weather": weather,
        }
        return WeatherByCityResponseSchema.model_validate(unvalidated_data)
