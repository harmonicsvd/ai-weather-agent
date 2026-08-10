"""
Meetings tools for backend agent.
Direct Google Calendar API calls using user's OAuth tokens.
"""

from typing import Optional
from pydantic import BaseModel, Field
from langchain.tools import tool
import os
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime, timedelta, timezone as tz
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google.auth.transport.requests import Request

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env", override=False)

GOOGLE_OAUTH_CLIENT_ID = os.getenv("GOOGLE_OAUTH_CLIENT_ID", "")
GOOGLE_OAUTH_CLIENT_SECRET = os.getenv("GOOGLE_OAUTH_CLIENT_SECRET", "")

SCOPES = ["https://www.googleapis.com/auth/calendar"]

class MeetingsSummaryInput(BaseModel):
    """Input schema for meetings-summary tool."""
    date: Optional[str] = Field(default=None, description="Date (default=today)")
    timezone: str = Field(default="Europe/Berlin", description="Timezone")
    user_sub: str = Field(description="User identifier for per-user calendar access")

@tool
async def meetings_summary_tool(
    date: Optional[str] = None,
    timezone: str = "Europe/Berlin",
    user_sub: str = ""
) -> str:
    """Get meetings summary. Returns user's calendar events for the specified date. If there are in-person meetings, the assistant can naturally provide weather recommendations based on the context."""

    import logging
    logger = logging.getLogger(__name__)

    # Fetch user's profile to get Google refresh token
    from apps.tools.profile_client import ProfileClient, ProfileProviderError

    try:
        with ProfileClient() as profile_client:
            profile = profile_client.get_profile_by_sub(user_sub)
            if not profile:
                return f"User profile not found. Please authenticate with Google Calendar first."
            refresh_token = profile.google_refresh_token
            if not refresh_token:
                return f"Google Calendar not authenticated. Please authenticate with Google Calendar first."
    except ProfileProviderError as e:
        logger.error(f"📅 Failed to fetch user profile: {e}")
        return f"Failed to fetch user profile: {str(e)}"

    # Convert date to UTC window
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")

    try:
        dt = datetime.strptime(date, "%Y-%m-%d")
    except:
        dt = datetime.now()

    day_start = dt.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=tz.utc)
    day_end = day_start + timedelta(days=1)

    from_iso = day_start.isoformat().replace("+00:00", "Z")
    to_iso = day_end.isoformat().replace("+00:00", "Z")

    # Build Google Calendar service with user's OAuth credentials
    try:
        credentials = Credentials(
            token=None,
            refresh_token=refresh_token,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=GOOGLE_OAUTH_CLIENT_ID,
            client_secret=GOOGLE_OAUTH_CLIENT_SECRET,
            scopes=SCOPES,
        )
        # Refresh the token to get a valid access token
        credentials.refresh(Request())

        service = build("calendar", "v3", credentials=credentials)

        # Fetch events from Google Calendar API
        events_result = service.events().list(
            calendarId='primary',
            timeMin=from_iso,
            timeMax=to_iso,
            singleEvents=True,
            orderBy='startTime'
        ).execute()

        events = events_result.get('items', [])
        logger.info(f"📅 Backend agent fetched {len(events)} events from Google Calendar API")

    except Exception as e:
        logger.error(f"📅 Failed to fetch events from Google Calendar API: {e}")
        return f"Failed to fetch calendar events: {str(e)}"

    if not events:
        return f"No meetings found for {date}."

    # Format meetings with full details
    summary_lines = [f"Meetings on {date}:"]
    for event in events:
        # Google Calendar format
        start = event.get("start", {}).get("dateTime", "")
        end = event.get("end", {}).get("dateTime", "")
        summary = event.get("summary", "No title")
        description = event.get("description", "")
        location = event.get("location", "No location")

        # Try to extract meeting_mode from description or location
        meeting_mode = "online"
        if location and location.lower() not in ["", "no location", "online", "zoom", "meet"]:
            meeting_mode = "in-person"

        # Format the meeting line with end time
        if description:
            if end:
                summary_lines.append(
                    f"- {summary} at {start} to {end}: {description} (Mode: {meeting_mode}, Location: {location})"
                )
            else:
                summary_lines.append(
                    f"- {summary} at {start}: {description} (Mode: {meeting_mode}, Location: {location})"
                )
        else:
            if end:
                summary_lines.append(
                    f"- {summary} at {start} to {end} (Mode: {meeting_mode}, Location: {location})"
                )
            else:
                summary_lines.append(
                    f"- {summary} at {start} (Mode: {meeting_mode}, Location: {location})"
                )

    return "\n".join(summary_lines)

# Register tools
from apps.tools import register_tool
register_tool(meetings_summary_tool)
