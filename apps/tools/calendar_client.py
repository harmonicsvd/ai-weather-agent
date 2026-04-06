from __future__ import annotations

import os
from typing import Any

import httpx
from pydantic import ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from apps.tools.schemas import CalendarEventsResponseSchema, CalendarEventSchema


class CalendarProviderError(Exception):
    """Raised when calendar provider call/response is invalid."""


class CalendarClient:
    def __init__(self, base_url: str | None = None, timeout_seconds: float = 10.0) -> None:
        self.base_url = (base_url or os.getenv("CALENDAR_API_BASE_URL", "http://127.0.0.1:8000")).rstrip("/")
        self._client = httpx.Client(timeout=timeout_seconds)

    def __enter__(self) -> "CalendarClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.TransportError)),
    )
    def _request_json(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        response = self._client.get(f"{self.base_url}{path}", params=params)
        response.raise_for_status()
        return response.json()

    def list_events(self, from_iso: str, to_iso: str) -> list[CalendarEventSchema]:
        try:
            payload = self._request_json("/events", {"from_iso": from_iso, "to_iso": to_iso})
            validated = CalendarEventsResponseSchema.model_validate(payload)
            return validated.events
        except ValidationError as exc:
            raise CalendarProviderError("Invalid calendar response format.") from exc
        except httpx.HTTPStatusError as exc:
            raise CalendarProviderError(f"Calendar API returned HTTP {exc.response.status_code}.") from exc
        except httpx.RequestError as exc:
            raise CalendarProviderError("Calendar API is unreachable.") from exc