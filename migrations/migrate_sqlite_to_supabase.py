"""Migrate user profiles from SQLite to Supabase PostgreSQL."""

import sqlite3
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import psycopg
from dotenv import load_dotenv

load_dotenv()


def get_sqlite_data():
    """Fetch all user profiles from SQLite database."""
    db_path = Path(__file__).parent.parent.parent / "voice-scheduling-agent/data" / "app.db"
    if not db_path.exists():
        print(f"SQLite database not found at {db_path}")
        return []
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT sub, email, default_city, timezone, role, commute_mode, ppe_required, risk_tolerance FROM user_profiles"
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def migrate_to_supabase():
    """Migrate SQLite data to Supabase PostgreSQL."""
    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url or "[YOUR-PASSWORD]" in database_url:
        print("ERROR: Please update DATABASE_URL in .env with your actual Supabase password")
        return False
    
    # Get SQLite data
    profiles = get_sqlite_data()
    if not profiles:
        print("No profiles found in SQLite database")
        return False
    
    print(f"Found {len(profiles)} profiles in SQLite database")
    
    # Connect to Supabase
    try:
        conn = psycopg.connect(database_url)
        with conn.cursor() as cur:
            # Insert each profile
            inserted = 0
            for profile in profiles:
                try:
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
                        (
                            profile.get("sub"),
                            profile.get("email"),
                            profile.get("default_city"),
                            profile.get("timezone"),
                            profile.get("role"),
                            profile.get("commute_mode"),
                            profile.get("ppe_required"),
                            profile.get("risk_tolerance"),
                        )
                    )
                    inserted += 1
                except Exception as e:
                    print(f"Error inserting profile {profile.get('sub')}: {e}")
            
            conn.commit()
            print(f"Successfully migrated {inserted} profiles to Supabase")
            return True
    except Exception as e:
        print(f"Error connecting to Supabase: {e}")
        return False
    finally:
        if 'conn' in locals():
            conn.close()


if __name__ == "__main__":
    success = migrate_to_supabase()
    sys.exit(0 if success else 1)
