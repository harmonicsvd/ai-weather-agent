"""Unit tests for OpenMeteoClient behavior and error mapping."""

import pytest

from apps.tools.schemas import (
    CurrentWeatherSchema,
    LocationSchema,
    WeatherByCityResponseSchema,
)
from apps.tools.weather_client import (
    CityNotFoundError,
    OpenMeteoClient,
    WeatherProviderError,
)


def test_geocode_city_returns_location_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    with OpenMeteoClient() as client:
        monkeypatch.setattr(
            client,
            "_get_json",
            lambda url, params: {
                "results": [
                    {
                        "name": "Hamburg",
                        "latitude": 53.55073,
                        "longitude": 9.99302,
                        "country": "Germany",
                    }
                ]
            },
        )

        result = client.geocode_city("Hamburg")

    assert isinstance(result, LocationSchema)
    assert result.name == "Hamburg"
    assert result.country == "Germany"


def test_geocode_city_raises_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    with OpenMeteoClient() as client:
        monkeypatch.setattr(client, "_get_json", lambda url, params: {"results": []})

        with pytest.raises(CityNotFoundError):
            client.geocode_city("does-not-exist")


def test_geocode_city_raises_on_empty_input() -> None:
    with OpenMeteoClient() as client:
        with pytest.raises(ValueError, match="city must not be empty"):
            client.geocode_city("   ")


def test_get_current_weather_returns_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    with OpenMeteoClient() as client:
        monkeypatch.setattr(
            client,
            "_get_json",
            lambda url, params: {
                "current": {
                    "temperature_2m": 7.4,
                    "apparent_temperature": 3.3,
                    "relative_humidity_2m": 64,
                    "wind_speed_10m": 14.4,
                    "weather_code": 3,
                    "time": "2026-03-28T16:15",
                }
            },
        )

        result = client.get_current_weather(53.55073, 9.99302)

    assert isinstance(result, CurrentWeatherSchema)
    assert result.temperature_c == 7.4
    assert result.humidity_percent == 64


def test_get_current_weather_raises_provider_error_for_missing_current(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with OpenMeteoClient() as client:
        monkeypatch.setattr(client, "_get_json", lambda url, params: {"hourly": {}})

        with pytest.raises(
            WeatherProviderError,
            match="Current weather data is missing from provider response.",
        ):
            client.get_current_weather(53.55073, 9.99302)


def test_get_current_weather_by_city_returns_typed_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with OpenMeteoClient() as client:
        monkeypatch.setattr(
            client,
            "geocode_city",
            lambda city: LocationSchema(
                name="Hamburg",
                latitude=53.55073,
                longitude=9.99302,
                country="Germany",
            ),
        )
        monkeypatch.setattr(
            client,
            "get_current_weather",
            lambda lat, lon: CurrentWeatherSchema(
                temperature_c=7.4,
                apparent_temperature_c=3.3,
                humidity_percent=64,
                wind_speed_kmh=14.4,
                weather_code=3,
                observation_time="2026-03-28T16:15",
            ),
        )

        result = client.get_current_weather_by_city("Hamburg")

    assert isinstance(result, WeatherByCityResponseSchema)
    assert result.location.name == "Hamburg"
    assert result.current_weather.temperature_c == 7.4
