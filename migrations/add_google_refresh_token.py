import os
import psycopg
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get the DATABASE_URL from environment
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    print("ERROR: DATABASE_URL not found in environment variables")
    exit(1)

print(f"Connecting to database...")

try:
    # Connect to the database
    conn = psycopg.connect(DATABASE_URL)
    
    # Read and execute the SQL migration
    with open("add_google_refresh_token.sql", "r") as f:
        sql = f.read()
    
    print("Executing migration: add_google_refresh_token.sql")
    conn.execute(sql)
    conn.commit()
    
    print("✓ Migration completed successfully!")
    print("  Added 'google_refresh_token' column to user_profiles table")
    
    # Verify the column was added
    result = conn.execute("""
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_name = 'user_profiles' 
        ORDER BY ordinal_position
    """).fetchall()
    
    print("\n  Current user_profiles columns:")
    for col in result:
        print(f"    - {col[0]} ({col[1]})")
    
    conn.close()
    
except Exception as e:
    print(f"✗ Migration failed: {e}")
    exit(1)
