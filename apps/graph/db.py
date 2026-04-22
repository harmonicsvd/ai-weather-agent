"""Direct SQLite access helper for reading voice-agent user profile data."""

import sqlite3
import os

def get_db():
    """Open SQLite connection to voice-agent DB with dict-like row access."""
    # Assume the DB is in the voice-scheduling-agent directory
    db_path = os.path.join(os.path.dirname(__file__), '../../../voice-scheduling-agent/app.db')
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def get_user_profile(sub: str):
    """Fetch one profile row by Google `sub`, return as dict or None."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT sub, email, default_city, timezone, role, commute_mode, ppe_required, risk_tolerance FROM user_profiles WHERE sub = ?",
            (sub,)
        ).fetchone()
        if row:
            return dict(row)
        return None
