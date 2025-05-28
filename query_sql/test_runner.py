import sys
import os

# Set up import path to access the module from the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from db import init_db
from chunker import search_top_chunks
from query_handler import handle_query

def run_test():
    # Step 1: Initialize DB (if not already initialized)
    try:
        print("✅ Initializing database...")
        init_db()
    except Exception as e:
        print(f"❌ Failed to initialize DB: {e}")
        return

    # Step 2: Define query
    query = "Tell me about the user"

    # Step 3: Search top chunks
    print("🔍 Searching top chunks for the query...")
    top_chunks, chunk_metadata = search_top_chunks(query)

    if not top_chunks:
        print("❌ No chunks found for the query. Ensure data is inserted into the 'chunks' table.")
        return

    # Step 4: Generate Gemini response
    print("🧠 Generating response from Gemini...")
    try:
        response = handle_query(query, top_chunks, chunk_metadata)
        print("\n=== ✅ Generated Response ===\n")
        print(response)
    except Exception as e:
        print(f"❌ Failed to generate response: {e}")

if __name__ == "__main__":
    run_test()
