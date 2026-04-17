from typing import Literal, TypedDict

from apps.tools.schemas import WeatherByCityResponseSchema


class GraphState(TypedDict, total=False):
    """
    Shared state object passed between LangGraph nodes.

    `total=False` means each node can update only the keys it owns,
    instead of returning every field on every step.
    """

    # Request context captured at graph entry.
    user_id: str
    user_query: str
    user_sub: str | None
    from_iso: str | None
    to_iso: str | None
    timezone: str | None
    target_date: str | None

    # Routing + resolution state.
    # Literal restricts values to known route labels for safer branching.
    intent: Literal["weather", "other"]
    city: str | None

    # Output data and user-facing result.
    # Keys remain optional because intermediate nodes may not set them yet.
    weather: WeatherByCityResponseSchema | None
    error: str | None
    final_response: str | None

    # Meeting fields
    events: list[dict] | None
    in_person_events: list[dict] | None
    recommendations: list[str] | None

    event_weather: list[dict] | None
    risk_summary: list[dict] | None
