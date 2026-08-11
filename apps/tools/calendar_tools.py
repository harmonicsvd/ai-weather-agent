from __future__ import annotations

"""Calendar tools for Sham."""

import os
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path

from dotenv import load_dotenv
from google.oauth2.service_account import Credentials as ServiceAccountCredentials
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# Ensure env vars are available
load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env", override=False)

from apps.google_clients import get_calendar_service
from apps.tools.profile_client import ProfileClient
import re

def parse_duration_to_minutes(duration_str: str) -> int:
    """
    Convert natural-language duration text into minutes.

    Examples:
    - "1 hour" -> 60
    - "one hour thirty min" -> 90
    - "45 min" -> 45

    If parsing fails, default to 60 minutes.
    """

    # Normalize input to lowercase so matching is case-insensitive.
    duration_str = duration_str.lower()

    # Replace common words with numeric equivalents to make regex parsing easier.
    # Note: this is a simple parser, not a full NLP parser.
    word_to_num = {
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "half": "30 min",
        "thirty": "30",
        "forty five": "45",
        "twenty": "20",
        "fifteen": "15",
        "ninety": "90",
    }

    # Apply all replacements to the duration string.
    for word, num in word_to_num.items():
        duration_str = duration_str.replace(word, num)

    # Find hour part, supports integers or decimal values like "1.5 hour".
    hours = re.search(r"(\d+\.?\d*)\s*hour", duration_str)

    # Find minute part like "30 min".
    minutes = re.search(r"(\d+)\s*min", duration_str)

    # Build total duration in minutes.
    total_minutes = 0
    if hours:
        total_minutes += float(hours.group(1)) * 60
    if minutes:
        total_minutes += int(minutes.group(1))

    # Return parsed value, else fallback to 60 minutes.
    return int(total_minutes) if total_minutes > 0 else 60

class CreateEventInput(BaseModel):
    """Input schema for create-event tool."""
    name: str = Field(description="Participant name")
    date: str = Field(description="Date (YYYY-MM-DD)")
    time: str = Field(description="Time (HH:MM)")
    title: Optional[str] = Field(default="Meeting", description="Event title")
    description: Optional[str] = Field(default=None, description="Meeting description/purpose")
    duration: Optional[str] = Field(default="1 hour", description="Duration")
    meeting_mode: str = Field(default="online", description="online or in_person")
    location: Optional[str] = Field(default=None, description="Location for in-person meetings")
    city: Optional[str] = Field(default=None, description="City for in-person meetings")
    user_sub: str = Field(description="User identifier")
    calendar_id: Optional[str] = Field(default="primary", description="Google Calendar ID (email address or 'primary')")
    timezone: Optional[str] = Field(default="Europe/Berlin", description="User timezone (e.g., Europe/Berlin, America/Los_Angeles)")

@tool
async def create_event_tool(
    name: str,
    date: str,
    time: str,
    title: Optional[str] = "Meeting",
    description: Optional[str] = None,
    duration: Optional[str] = "1 hour",
    meeting_mode: str = "online",
    location: Optional[str] = None,
    city: Optional[str] = None,
    user_sub: str = "",
    calendar_id: str = None,
    timezone: str = "Europe/Berlin"
) -> str:
    """Create a calendar event. Use when user wants to book, schedule, or set up a meeting. Ask for: name, date, time, meeting mode (online/in-person), and optionally description/purpose.
    
    IMPORTANT CONVERSATION RULES:
    - Ask questions ONE AT A TIME - don't ask multiple questions in a single response
    - Let the user provide all information naturally before asking for confirmation
    - If the user is providing multiple details (date, time, purpose, etc.), let them finish before asking anything else
    """
    try:
        print(f"[create_event_tool] Starting - user_sub={user_sub}, calendar_id={calendar_id}")
        
        # Fetch user email, refresh token, and timezone from profile
        refresh_token = None
        if user_sub and not calendar_id:
            print(f"[create_event_tool] Fetching profile for user_sub: {user_sub}")
            try:
                with ProfileClient() as profile_client:
                    profile = profile_client.get_profile_by_sub(user_sub)
                    print(f"[create_event_tool] Profile fetch result: {profile}")
                    if profile and profile.email:
                        calendar_id = profile.email
                        print(f"[create_event_tool] Using user email as calendar_id: {calendar_id}")
                        refresh_token = profile.google_refresh_token
                        print(f"[create_event_tool] Refresh token available: {bool(refresh_token)}")
                        # Use default timezone since we removed it from profile schema
                        timezone = "Europe/Berlin"
                        print(f"[create_event_tool] Using default timezone: {timezone}")
                    else:
                        print(f"[create_event_tool] Profile found but no email, using fallback")
            except Exception as e:
                print(f"[create_event_tool] Failed to fetch user profile: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"[create_event_tool] Skipping profile fetch - user_sub={bool(user_sub)}, calendar_id={calendar_id}")
        
        # Fallback to CALENDAR_ID from env or primary
        if not calendar_id:
            calendar_id = os.getenv("CALENDAR_ID", "primary")
            print(f"[create_event_tool] Using fallback calendar_id: {calendar_id}")
                # Handle null/empty time
        if not time or time == "null":
            return f"Failed to create event: Time is required. Please provide a valid time (e.g., 4pm, 16:00)."

        # Normalize time format (handle 4pm -> 16:00)
        if "pm" in time.lower() and ":" not in time:
            time_num = int(time.lower().replace("pm", "").strip())
            if time_num != 12:
                time_num += 12
            time = f"{time_num}:00"
        elif "am" in time.lower() and ":" not in time:
            time_num = int(time.lower().replace("am", "").strip())
            if time_num == 12:
                time_num = 0
            time = f"{time_num:02d}:00"

        # Calculate end time based on duration
        duration_str = duration if duration else "1 hour"
        duration_minutes = parse_duration_to_minutes(duration_str)
        start_dt = datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M")
        end_dt = start_dt + timedelta(minutes=duration_minutes)

        # Use timezone passed from execute_plan node
        user_timezone = timezone or "Europe/Berlin"
        print(f"[create_event_tool] Using timezone: {user_timezone}")

        service = get_calendar_service(refresh_token=refresh_token)
        
        # Log the event details before creation
        title_str = title if title else "Meeting"
        print(f"[create_event_tool] Attempting to create event:")
        print(f"  Title: {title_str}")
        print(f"  Date: {date}")
        print(f"  Time: {time}")
        print(f"  Start: {start_dt.strftime('%Y-%m-%dT%H:%M:%S')}")
        print(f"  End: {end_dt.strftime('%Y-%m-%dT%H:%M:%S')}")
        print(f"  Duration: {duration_minutes} minutes")
        
        event_body = {
            "summary": title_str,
            "description": description or "",
            "start": {
                "dateTime": start_dt.strftime("%Y-%m-%dT%H:%M:%S"),
                "timeZone": user_timezone,
            },
            "end": {
                "dateTime": end_dt.strftime("%Y-%m-%dT%H:%M:%S"),
                "timeZone": user_timezone,
            },
        }
        
        event = (
            service.events()
            .insert(
                calendarId=calendar_id,
                body=event_body,
            )
            .execute()
        )
        
        print(f"[create_event_tool] Event created successfully with ID: {event.get('id')}")
        return f"Successfully created event: {title} on {date} at {time} (Event ID: {event.get('id')})"
    except Exception as e:
        return f"Failed to create event: {str(e)}"

# Register tools
from apps.tools import register_tool
register_tool(create_event_tool)
