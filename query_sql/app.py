import streamlit as st
import time
import os
import shutil
from pathlib import Path
from typing import List, Tuple
from file_handler import extract_text_from_file
from chunker import chunk_text
from embedder import embed_chunks, store_embeddings
from db import init_db, get_db, truncate_table
import psycopg2
import logging
from chatbot import chatbot_page

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

# Ensure uploads directory exists
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Initialize database only once
if "db_initialized" not in st.session_state:
    logger.info("Initializing database")
    init_db()
    st.session_state.db_initialized = True

def process_with_progress_bar(stage_name: str, func, *args, **kwargs) -> Tuple[bool, any]:
    """
    Execute a function and handle success/failure.
    Returns (success: bool, result: any).
    """
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write(f"**{stage_name}**")
    with col2:
        status_text = st.empty()
        status_text.text("Pending...")

    try:
        result = func(*args, **kwargs)
        status_text.success("Completed Successfully")
        return True, result
    except Exception as e:
        status_text.error(f"Failed: {str(e)}")
        return False, None

def document_processing_page():
    st.title("Document Processing App")
    st.write("Upload a PDF, Excel, or TXT file to process it through the following stages:")

    stages = ["Text Extraction", "Embedding", "Database Storage"]
    for stage in stages:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**{stage}**")
        with col2:
            st.text("Pending...")

    st.markdown("---")

    if "confirm_clear" not in st.session_state:
        st.session_state.confirm_clear = False

    if st.button("Clear Data"):
        logger.info("Clear Data button clicked")
        st.session_state.confirm_clear = True

    if st.session_state.confirm_clear:
        st.warning("Are you sure you want to clear all data in the database? This action cannot be undone.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Yes, Clear Data"):
                logger.info("User confirmed to clear data")
                try:
                    truncate_table()
                    st.success("Database cleared successfully!")
                    st.session_state.confirm_clear = False
                except Exception as e:
                    st.error(f"Failed to clear database: {str(e)}")
                    st.session_state.confirm_clear = False
        with col2:
            if st.button("Cancel"):
                logger.info("User canceled data clearing")
                st.session_state.confirm_clear = False

    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Drag and drop your file here",
        type=['pdf', 'txt', 'xlsx', 'xls'],
        accept_multiple_files=False
    )

    if uploaded_file is not None:
        file_path = UPLOAD_DIR / uploaded_file.name
        with open(file_path, "wb") as f:
            shutil.copyfileobj(uploaded_file, f)
        
        st.success(f"File uploaded: {uploaded_file.name}")
        
        stage_results = []
        results = {}

        # Stage 1: Text Extraction
        success, result = process_with_progress_bar(
            "Text Extraction",
            extract_text_from_file,
            str(file_path)
        )
        stage_results.append(success)
        if success:
            results["text"] = result
        else:
            st.error("Text Extraction failed. Stopping process.")
            return

        # Stage 2: Embedding (Chunking + Embedding)
        if results["text"]:
            chunks = chunk_text(results["text"])
            if not chunks:
                st.error("No chunks generated. Stopping process.")
                return
            
            success, embeddings = process_with_progress_bar(
                "Embedding",
                embed_chunks,
                chunks
            )
            stage_results.append(success)
            if success:
                results["chunks"] = chunks
                results["embeddings"] = embeddings
            else:
                st.error("Embedding failed. Stopping process.")
                return

        # Stage 3: Database Storage
        if results.get("chunks") and results.get("embeddings"):
            with get_db() as conn:
                success, _ = process_with_progress_bar(
                    "Database Storage",
                    store_embeddings,
                    results["chunks"],
                    results["embeddings"],
                    uploaded_file.name,
                    conn
                )
                stage_results.append(success)
                if not success:
                    st.error("Database Storage failed. Stopping process.")
                    return

        success_percentage = (sum(1 for result in stage_results if result) / len(stages)) * 100
        st.write("### Overall Progress")
        st.write(f"Completed: {success_percentage:.1f}%")

        if file_path.exists():
            os.remove(file_path)

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Select a page", ["Document Processing", "Chatbot"])

    if "last_page" not in st.session_state:
        st.session_state.last_page = page
    if st.session_state.last_page != page:
        logger.info(f"Switching from {st.session_state.last_page} to {page}")
        st.session_state.last_page = page
        if "confirm_clear" in st.session_state:
            st.session_state.confirm_clear = False

    if page == "Document Processing":
        document_processing_page()
    elif page == "Chatbot":
        chatbot_page()

if __name__ == "__main__":
    main()