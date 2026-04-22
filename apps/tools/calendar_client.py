from __future__ import annotations

"""HTTP client that fetches events from voice-agent internal calendar endpoint."""

import os
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from pydantic import ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from apps.tools.schemas import CalendarEventsResponseSchema, CalendarEventSchema

# Ensure env vars are available even when entrypoint is apps/graph/run_graph.py.
load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env", override=False)


class CalendarProviderError(Exception):
    """Raised when calendar provider call/response is invalid."""


class CalendarClient:
    """HTTP client for voice backend `/internal/events` endpoint."""

    def __init__(self, base_url: str | None = None, timeout_seconds: float = 10.0) -> None:
        self.base_url = (base_url or os.getenv("CALENDAR_API_BASE_URL", "http://127.0.0.1:8000")).rstrip("/")
        self.internal_api_key = os.getenv("CALENDAR_INTERNAL_API_KEY", "")
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
        headers = {}
        if self.internal_api_key:
            headers["X-Internal-API-Key"] = self.internal_api_key

        response = self._client.get(f"{self.base_url}{path}", params=params, headers=headers)
        response.raise_for_status()
        return response.json()

    def list_events(self, from_iso: str, to_iso: str) -> list[CalendarEventSchema]:
        """Return validated calendar events for a UTC window."""
        try:
            if not self.internal_api_key:
                raise CalendarProviderError("CALENDAR_INTERNAL_API_KEY is not configured.")
            payload = self._request_json("/internal/events", {"from_iso": from_iso, "to_iso": to_iso})
            validated = CalendarEventsResponseSchema.model_validate(payload)
            return validated.events
        except ValidationError as exc:
            raise CalendarProviderError("Invalid calendar response format.") from exc
        except httpx.HTTPStatusError as exc:
            raise CalendarProviderError(f"Calendar API returned HTTP {exc.response.status_code}.") from exc
        except httpx.RequestError as exc:
            raise CalendarProviderError("Calendar API is unreachable.") from exc
