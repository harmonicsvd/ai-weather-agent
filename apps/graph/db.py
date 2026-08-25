"""PostgreSQL access helper for user profile data via Supabase."""

import os
import psycopg
from contextlib import contextmanager
from datetime import datetime, timezone
from uuid import uuid4

# Default skills that are auto-installed for all users
DEFAULT_SKILLS = os.getenv("DEFAULT_SKILLS", "google_calendar").split(",")


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


def get_user_installed_skills(user_sub: str) -> list[str]:
    """Get list of skill names installed by a user from database."""
    database_url = get_database_url()
    if not database_url:
        # Fallback to SQLite if DATABASE_URL not set
        import sqlite3
        db_path = os.path.join(os.path.dirname(__file__), '../../../voice-scheduling-agent/app.db')
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT skill_name FROM user_installed_skills WHERE user_sub = ? AND status = 'active'",
                (user_sub,)
            ).fetchall()
            return [row["skill_name"] for row in rows]
        finally:
            conn.close()
    
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT skill_name FROM user_installed_skills WHERE user_sub = %s AND status = 'active'",
                (user_sub,)
            )
            rows = cur.fetchall()
            return [row[0] for row in rows]


def install_default_skills_for_user(user_sub: str):
    """Install default skills for a user."""
    database_url = get_database_url()
    if not database_url:
        # Fallback to SQLite if DATABASE_URL not set
        import sqlite3
        db_path = os.path.join(os.path.dirname(__file__), '../../../voice-scheduling-agent/app.db')
        conn = sqlite3.connect(db_path)
        try:
            for skill_name in DEFAULT_SKILLS:
                conn.execute(
                    """
                    INSERT INTO user_installed_skills (id, user_sub, skill_name, status, installed_at)
                    VALUES (?, ?, ?, 'active', ?)
                    ON CONFLICT(user_sub, skill_name) DO NOTHING
                    """,
                    (str(uuid4()), user_sub, skill_name, datetime.now(timezone.utc).isoformat())
                )
            conn.commit()
            return True
        finally:
            conn.close()
    
    with get_db() as conn:
        with conn.cursor() as cur:
            for skill_name in DEFAULT_SKILLS:
                cur.execute(
                    """
                    INSERT INTO user_installed_skills (id, user_sub, skill_name, status, installed_at)
                    VALUES (%s, %s, %s, 'active', %s)
                    ON CONFLICT (user_sub, skill_name) DO NOTHING
                    """,
                    (str(uuid4()), user_sub, skill_name, datetime.now(timezone.utc).isoformat())
                )
            conn.commit()
            return True


def create_user_profile(sub: str, email: str, default_city: str = None, timezone: str = "Europe/Berlin", role: str = None, commute_mode: str = None, ppe_required: bool = False, risk_tolerance: str = None):
    """Create or update a user profile in PostgreSQL with default skills."""
    database_url = get_database_url()
    if not database_url:
        # Fallback to SQLite if DATABASE_URL not set
        import sqlite3
        db_path = os.path.join(os.path.dirname(__file__), '../../../voice-scheduling-agent/app.db')
        conn = sqlite3.connect(db_path)
        try:
            # Create/update profile
            conn.execute(
                """
                INSERT INTO user_profiles (sub, email, default_city, timezone, role, commute_mode, ppe_required, risk_tolerance)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (sub) DO UPDATE SET
                    email = excluded.email,
                    default_city = excluded.default_city,
                    timezone = excluded.timezone,
                    role = excluded.role,
                    commute_mode = excluded.commute_mode,
                    ppe_required = excluded.ppe_required,
                    risk_tolerance = excluded.risk_tolerance,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (sub, email, default_city, timezone, role, commute_mode, ppe_required, risk_tolerance)
            )
            
            # Auto-install default skills for new users
            for skill_name in DEFAULT_SKILLS:
                conn.execute(
                    """
                    INSERT INTO user_installed_skills (id, user_sub, skill_name, status, installed_at)
                    VALUES (?, ?, ?, 'active', ?)
                    ON CONFLICT(user_sub, skill_name) DO NOTHING
                    """,
                    (str(uuid4()), sub, skill_name, datetime.now(timezone.utc).isoformat())
                )
            
            conn.commit()
            return True
        finally:
            conn.close()
    
    with get_db() as conn:
        with conn.cursor() as cur:
            # Create/update profile
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
            
            # Auto-install default skills for new users
            for skill_name in DEFAULT_SKILLS:
                cur.execute(
                    """
                    INSERT INTO user_installed_skills (id, user_sub, skill_name, status, installed_at)
                    VALUES (%s, %s, %s, 'active', %s)
                    ON CONFLICT (user_sub, skill_name) DO NOTHING
                    """,
                    (str(uuid4()), sub, skill_name, datetime.now(timezone.utc).isoformat())
                )
            
            conn.commit()
            return True
