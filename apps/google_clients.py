"""Google API client factories (Calendar service)."""

import json
import os
from pathlib import Path
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env", override=False)

SCOPES = ["https://www.googleapis.com/auth/calendar"]


def get_calendar_service(refresh_token: str | None = None):
    """
    Build Google Calendar service client.
    
    If refresh_token is provided, uses OAuth credentials (user's own calendar).
    Otherwise, falls back to service account credentials.
    """
    if refresh_token:
        # Use OAuth credentials with refresh token
        client_id = os.getenv("GOOGLE_OAUTH_CLIENT_ID")
        client_secret = os.getenv("GOOGLE_OAUTH_CLIENT_SECRET")
        
        if not client_id or not client_secret:
            raise ValueError("GOOGLE_OAUTH_CLIENT_ID and GOOGLE_OAUTH_CLIENT_SECRET must be set for OAuth")
        
        credentials = Credentials(
            token=None,  # Will be refreshed
            refresh_token=refresh_token,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=client_id,
            client_secret=client_secret,
            scopes=SCOPES,
        )
        # Refresh the token to get a valid access token
        credentials.refresh(Request())
    else:
        # Fallback to service account
        service_account_json = os.getenv("SERVICE_ACCOUNT_JSON")
        service_account_file = os.getenv("SERVICE_ACCOUNT_FILE", "service_account.json")
        
        if service_account_json:
            service_account_info = json.loads(service_account_json)
            credentials = service_account.Credentials.from_service_account_info(
                service_account_info,
                scopes=SCOPES,
            )
        else:
            credentials = service_account.Credentials.from_service_account_file(
                service_account_file,
                scopes=SCOPES,
            )

    return build("calendar", "v3", credentials=credentials)
