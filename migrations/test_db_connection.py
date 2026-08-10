"""Test Supabase database connection."""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import psycopg
from dotenv import load_dotenv

load_dotenv()


def test_connection():
    """Test PostgreSQL connection to Supabase."""
    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        print("ERROR: DATABASE_URL not set in .env")
        return False
    
    print(f"Testing connection to: {database_url[:30]}...")
    
    try:
        conn = psycopg.connect(database_url)
        with conn.cursor() as cur:
            # Test simple query
            cur.execute("SELECT version()")
            version = cur.fetchone()
            print(f"✓ Connected successfully!")
            print(f"  PostgreSQL version: {version[0][:50]}...")
            
            # Check if tables exist
            cur.execute("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)
            tables = cur.fetchall()
            print(f"  Existing tables: {[t[0] for t in tables]}")
            
            # Check if pgvector is enabled
            cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector'")
            pgvector = cur.fetchone()
            print(f"  pgvector enabled: {pgvector is not None}")
            
        conn.close()
        return True
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return False


if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)
