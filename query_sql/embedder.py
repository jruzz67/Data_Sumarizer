import os
import logging
from typing import List
import ollama

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

# Test connection to Ollama and nomic-embed-text model
logger.info("Testing connection to nomic-embed-text via Ollama...")
try:
    test_input = "Test input for embedding model."
    test_embedding = ollama.embeddings(model="nomic-embed-text", prompt=test_input)
    logger.info(f"Test embedding successful: {len(test_embedding['embedding'])} dimensions")
except Exception as e:
    logger.error(f"Failed to connect to nomic-embed-text via Ollama: {str(e)}")
    raise
logger.info("Ollama connection and nomic-embed-text model verified.")

def embed_chunks(chunks: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of text chunks using nomic-embed-text via Ollama.
    
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
        
        # Process chunks in a batch for efficiency
        for chunk in chunks:
            response = ollama.embeddings(model="nomic-embed-text", prompt=chunk)
            embedding = response["embedding"]
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