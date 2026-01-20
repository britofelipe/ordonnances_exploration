import sqlite3
import pandas as pd
import os

# Define database path relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "../../data/thesorimed/THESORIMED_SQ3")

def inspect_db():
    if not os.path.exists(DB_PATH):
        print(f"Error: Database not found at {DB_PATH}")
        return

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        print(f"Successfully connected to Thesorimed DB!")
        print(f"Found {len(tables)} tables:\n")
        
        for table in tables:
            table_name = table[0]
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            print(f"- {table_name} ({count} rows)")
            
        print("\n--- Sample Data (First 3 rows of first 3 tables) ---")
        for i, table in enumerate(tables[:3]):
            table_name = table[0]
            print(f"\nTable: {table_name}")
            df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 3", conn)
            print(df)

        conn.close()

    except Exception as e:
        print(f"Error connecting to database: {e}")

if __name__ == "__main__":
    inspect_db()
