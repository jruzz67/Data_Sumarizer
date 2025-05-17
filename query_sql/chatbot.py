import streamlit as st
import time
import logging
from embedder import embed_chunks
from query_handler import handle_query
from db import get_db

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

def chatbot_page():
    st.title("Chatbot")
    st.write("Ask questions about the documents stored in the database.")

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = {}

    # Initialize query embedding cache
    if "query_embedding_cache" not in st.session_state:
        st.session_state.query_embedding_cache = {}

    # Display chat history for the current session
    if "chatbot" not in st.session_state.chat_history:
        st.session_state.chat_history["chatbot"] = []

    for message in st.session_state.chat_history["chatbot"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "runtime_ms" in message:
                st.write(f"**Runtime**: {message['runtime_ms']:.2f} ms")

    user_query = st.chat_input("Ask a question about the documents...")

    if user_query:
        with st.chat_message("user"):
            st.markdown(user_query)

        start_time = time.time()
        try:
            # Check if query embedding is cached
            if user_query in st.session_state.query_embedding_cache:
                query_embedding = st.session_state.query_embedding_cache[user_query]
            else:
                query_embedding = embed_chunks([user_query])[0]
                st.session_state.query_embedding_cache[user_query] = query_embedding

            with get_db() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT chunk_text
                    FROM chunks
                    ORDER BY embedding <=> CAST(%s AS vector)
                    LIMIT 5;
                    """,
                    (query_embedding,)
                )
                top_chunks = [row["chunk_text"] for row in cursor.fetchall()]

            if not top_chunks:
                response = "No relevant chunks found in the database."
            else:
                # response = handle_query(user_query, top_chunks)
                response = top_chunks

            runtime_ms = (time.time() - start_time) * 1000
            with st.chat_message("assistant"):
                st.markdown(response)
                st.write(f"**Runtime**: {runtime_ms:.2f} ms")

            st.session_state.chat_history["chatbot"].append({"role": "user", "content": user_query})
            st.session_state.chat_history["chatbot"].append({"role": "assistant", "content": response, "runtime_ms": runtime_ms})

        except Exception as e:
            runtime_ms = (time.time() - start_time) * 1000
            error_message = f"Error processing query: {str(e)}"
            with st.chat_message("assistant"):
                st.markdown(error_message)
                st.write(f"**Runtime**: {runtime_ms:.2f} ms")
            st.session_state.chat_history["chatbot"].append({"role": "user", "content": user_query})
            st.session_state.chat_history["chatbot"].append({"role": "assistant", "content": error_message, "runtime_ms": runtime_ms})