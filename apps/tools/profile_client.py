from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

# Ensure env vars are available even when entrypoint is apps/graph/run_graph.py.
load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env", override=False)


class ProfileProviderError(Exception):
    """Raised when internal profile API call/response is invalid."""


class UserProfileSchema(BaseModel):
    sub: str
    email: str
    default_city: str
    timezone: str
    updated_at: str


class InternalProfileResponseSchema(BaseModel):
    profile: UserProfileSchema


class ProfileClient:
    def __init__(
        self,
        base_url: str | None = None,
        internal_api_key: str | None = None,
        timeout_seconds: float = 5.0,
    ) -> None:
        self.base_url = (base_url or os.getenv("PROFILE_API_BASE_URL", "http://127.0.0.1:8000")).rstrip("/")
        self.internal_api_key = internal_api_key or os.getenv("PROFILE_INTERNAL_API_KEY", "")
        self._client = httpx.Client(timeout=timeout_seconds)

    def __enter__(self) -> "ProfileClient":
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
    def _request_json(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self.internal_api_key:
            raise ProfileProviderError("PROFILE_INTERNAL_API_KEY is not configured.")
        response = self._client.get(
            f"{self.base_url}{path}",
            params=params,
            headers={"X-Internal-API-Key": self.internal_api_key},
        )
        response.raise_for_status()
        return response.json()

    def get_profile_by_sub(self, sub: str) -> UserProfileSchema | None:
        try:
            payload = self._request_json(f"/internal/profile/{sub}")
            validated = InternalProfileResponseSchema.model_validate(payload)
            return validated.profile
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return None
            raise ProfileProviderError(
                f"Profile API returned HTTP {exc.response.status_code}."
            ) from exc
        except ValidationError as exc:
            raise ProfileProviderError("Invalid profile response format.") from exc
        except httpx.RequestError as exc:
            raise ProfileProviderError("Profile API is unreachable.") from exc
