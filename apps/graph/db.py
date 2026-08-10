"""PostgreSQL access helper for user profile data via Supabase."""

import os
import psycopg
from contextlib import contextmanager


def get_database_url() -> str:
    return os.getenv("DATABASE_URL", "").strip()


@contextmanager
def get_db():
    """Open PostgreSQL connection to Supabase with dict-like row access."""
    database_url = get_database_url()
    if not database_url:
        raise RuntimeError("DATABASE_URL is required for database access.")

    conn = psycopg.connect(database_url)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def get_user_profile(sub: str):
    """Fetch one profile row by Google `sub`, return as dict or None."""
    database_url = get_database_url()
    if not database_url:
        # Fallback to SQLite if DATABASE_URL not set
        import sqlite3
        db_path = os.path.join(os.path.dirname(__file__), '../../../voice-scheduling-agent/app.db')
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                "SELECT sub, email, default_city, timezone, role, commute_mode, ppe_required, risk_tolerance FROM user_profiles WHERE sub = ?",
                (sub,)
            ).fetchone()
            if row:
                return dict(row)
            return None
        finally:
            conn.close()
    
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT sub, email, default_city, timezone, role, commute_mode, ppe_required, risk_tolerance FROM user_profiles WHERE sub = %s",
                (sub,)
            )
            row = cur.fetchone()
            if row:
                columns = [desc[0] for desc in cur.description]
                return dict(zip(columns, row))
            return None


def create_user_profile(sub: str, email: str, default_city: str = None, timezone: str = "Europe/Berlin", role: str = None, commute_mode: str = None, ppe_required: bool = False, risk_tolerance: str = None):
    """Create or update a user profile in PostgreSQL."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO user_profiles (sub, email, default_city, timezone, role, commute_mode, ppe_required, risk_tolerance)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (sub) DO UPDATE SET
                    email = EXCLUDED.email,
                    default_city = EXCLUDED.default_city,
                    timezone = EXCLUDED.timezone,
                    role = EXCLUDED.role,
                    commute_mode = EXCLUDED.commute_mode,
                    ppe_required = EXCLUDED.ppe_required,
                    risk_tolerance = EXCLUDED.risk_tolerance,
                    updated_at = NOW()
                """,
                (sub, email, default_city, timezone, role, commute_mode, ppe_required, risk_tolerance)
            )
            conn.commit()
            return True
