import os
import logging
from typing import List
from dotenv import load_dotenv
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

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

# Initialize Gemini API
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables")
genai.configure(api_key=gemini_api_key)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def embed_chunks(chunks: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of text chunks using Gemini's text-embedding-004 model.
    
    Args:
        chunks (List[str]): List of text chunks to embed.
    
    Returns:
        List[List[float]]: List of embeddings, each a 768-dimensional vector.
    """
    if not chunks:
        return []

    try:
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        embeddings = []
        
        # Gemini API doesn't support batch embedding in a single call, so we process chunks individually
        for chunk in chunks:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=chunk,
                task_type="retrieval_document"
            )
            embedding = result["embedding"]
            embeddings.append(embedding)
            logger.info(f"Processed chunk {len(embeddings)} of {len(chunks)}")

        # Verify embedding dimensions
        for emb in embeddings:
            if len(emb) != 768:
                raise ValueError(f"Unexpected embedding dimension: {len(emb)} (expected 768)")

        logger.info("Embeddings generated successfully")
        return embeddings
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise

def store_embeddings(chunks: List[str], embeddings: List[List[float]], document_name: str, conn):
    try:
        cursor = conn.cursor()
        for chunk, embedding in zip(chunks, embeddings):
            cursor.execute(
                """
                INSERT INTO chunks (document_name, chunk_text, embedding)
                VALUES (%s, %s, %s);
                """,
                (document_name, chunk, embedding)
            )
        conn.commit()
        logger.info(f"Stored {len(chunks)} chunks and embeddings for document: {document_name}")
    except Exception as e:
        logger.error(f"Error storing embeddings for {document_name}: {str(e)}")
        conn.rollback()
        raise