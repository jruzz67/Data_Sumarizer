import os
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
from dotenv import load_dotenv

load_dotenv()

DB_NAME = os.getenv("DB_NAME", "query_sql_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "your_password")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")

def init_db():
    """
    Initialize the database, create the 'chunks' table if it doesn't exist, and enable the vector extension.
    """
    conn = None
    try:
        conn = psycopg2.connect(
            dbname="postgres",
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        conn.autocommit = True
        cursor = conn.cursor()

        cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = '{DB_NAME}'")
        if not cursor.fetchone():
            cursor.execute(f"CREATE DATABASE {DB_NAME}")

        cursor.close()
        conn.close()

        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        cursor = conn.cursor()

        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # Check if chunks table exists
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'chunks'
            );
        """)
        table_exists = cursor.fetchone()[0]

        if not table_exists:
            cursor.execute("""
                CREATE TABLE chunks (
                    id SERIAL PRIMARY KEY,
                    document_name TEXT NOT NULL,
                    chunk_text TEXT NOT NULL,
                    embedding vector(768)
                );
            """)
            print("Created chunks table.")
        else:
            print("Chunks table already exists, skipping creation.")

        conn.commit()
    except Exception as e:
        print(f"Error initializing database: {str(e)}")
        raise
    finally:
        if conn:
            cursor.close()
            conn.close()

@contextmanager
def get_db():
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    conn.cursor_factory = RealDictCursor
    try:
        yield conn
    finally:
        conn.close()

def truncate_table():
    conn = None
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        cursor = conn.cursor()
        cursor.execute("TRUNCATE TABLE chunks RESTART IDENTITY;")
        conn.commit()
        print("Successfully truncated the chunks table.")
    except Exception as e:
        print(f"Error truncating table: {str(e)}")
        raise
    finally:
        if conn:
            cursor.close()
            conn.close()

if __name__ == "__main__":
    init_db()