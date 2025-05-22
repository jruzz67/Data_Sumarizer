import os
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
from dotenv import load_dotenv
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

DB_NAME = os.getenv("DB_NAME", "query_sql_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "your_password")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")

def init_db():
    """
    Initialize the database, create the 'chunks' table if it doesn't exist, enable the vector extension,
    and create an HNSW index for faster vector searches.
    """
    conn = None
    try:
        start_time = time.time()
        # Connect to default 'postgres' database to check/create target database
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
            logger.info(f"Created database: {DB_NAME}")

        cursor.close()
        conn.close()

        # Connect to target database
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        cursor = conn.cursor()

        # Enable vector extension
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        logger.info("Vector extension enabled")

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
            logger.info("Created chunks table")
        else:
            logger.info("Chunks table already exists, skipping creation")

        # Create HNSW index for cosine distance searches
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS chunks_embedding_idx 
            ON chunks 
            USING hnsw (embedding vector_cosine_ops) 
            WITH (m = 16, ef_construction = 64);
        """)
        logger.info("Created HNSW index on chunks.embedding")

        conn.commit()
        logger.info(f"Database initialized in {time.time() - start_time:.3f}s")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
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
        start_time = time.time()
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
        logger.info(f"Truncated chunks table in {time.time() - start_time:.3f}s")
    except Exception as e:
        logger.error(f"Error truncating table: {str(e)}")
        raise
    finally:
        if conn:
            cursor.close()
            conn.close()

if __name__ == "__main__":
    init_db()