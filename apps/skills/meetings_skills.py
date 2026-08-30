"""
Meetings skills for backend agent.
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

def search_knowledge_base(query: str, user_sub: str, logger) -> str:
    """Search user's knowledge base for meeting notes and discussion content."""
    try:
        from apps.rag.retriever import retrieve_user_context
        from apps.rag.embeddings import GeminiEmbeddingProvider
        
        embedding_provider = GeminiEmbeddingProvider()
        results = retrieve_user_context(
            user_sub=user_sub,
            query_text=query,
            embedding_provider=embedding_provider,
            top_k=3
        )
        
        if not results:
            return f"\n\nNo relevant meeting notes found for: '{query}'"
        
        # Format results
        rag_lines = [f"\n\nMeeting Notes for: '{query}'"]
        for i, result in enumerate(results, 1):
            chunk = result.chunk
            score = result.score
            source_file = chunk.metadata.get("source_file", "unknown")
            
            rag_lines.append(f"\n--- From {source_file} (Relevance: {score:.2f}) ---")
            rag_lines.append(chunk.text)
        
        return "\n".join(rag_lines)
        
    except Exception as e:
        logger.error(f"RAG search failed: {e}")
        return f"\n\nFailed to search meeting notes: {str(e)}"

class MeetingsSummaryInput(BaseModel):
    """Input schema for meetings-summary skill."""
    date: Optional[str] = Field(default=None, description="Date (default=today)")
    timezone: str = Field(default="Europe/Berlin", description="Timezone")
    user_sub: str = Field(description="User identifier for per-user calendar access")
    query: Optional[str] = Field(default=None, description="Optional query to search meeting notes/discussion content")

@tool("meeting_discussion")
async def meetings_summary_tool(
    date: Optional[str] = None,
    timezone: str = "Europe/Berlin",
    user_sub: str = "",
    query: Optional[str] = None
) -> str:
    """Get meetings summary. Returns user's calendar events for the specified date. If a query is provided, also searches meeting notes/discussion content from knowledge base. If there are in-person meetings, the assistant can naturally provide weather recommendations based on the context."""

    import logging
    logger = logging.getLogger(__name__)

    # Fetch user's profile to get Google refresh token
    from apps.skills.profile_client import ProfileClient, ProfileProviderError

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
    # If only query is provided (no date), skip calendar fetch and only search knowledge base
    if query and not date:
        logger.info(f"📅 Query provided without date - skipping calendar fetch, searching knowledge base only")
        return search_knowledge_base(query, user_sub, logger)
    
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
        # If no events but query provided, still search knowledge base
        if query:
            return search_knowledge_base(query, user_sub, logger)
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

    calendar_summary = "\n".join(summary_lines)
    
    # If query is provided, search knowledge base for meeting notes
    if query:
        rag_result = search_knowledge_base(query, user_sub, logger)
        calendar_summary += rag_result
    
    return calendar_summary

# Register skills
from apps.skills import register_skill
register_skill(meetings_summary_tool)
