import os
import logging
from typing import List
from dotenv import load_dotenv
import ollama
from tenacity import retry, stop_after_attempt, wait_exponential
from concurrent.futures import ThreadPoolExecutor, as_completed

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

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def embed_single_chunk(chunk: str) -> List[float]:
    """
    Generate embedding for a single text chunk using Ollama's nomic-embed-text model.
    
    Args:
        chunk (str): Text chunk to embed.
    
    Returns:
        List[float]: Embedding vector (768-dimensional).
    """
    try:
        response = ollama.embeddings(
            model="nomic-embed-text:latest",
            prompt=chunk
        )
        embedding = response["embedding"]
        if len(embedding) != 768:
            raise ValueeError(f"Unexpected embedding dimension: {len(embedding)} (expected 768)")
        return embedding
    except Exception as e:
        logger.error(f"Error generating embedding for chunk: {str(e)}")
        raise

def embed_chunks(chunks: List[str], max_workers: int = 4) -> List[List[float]]:
    """
    Generate embeddings for a list of text chunks using Ollama's nomic-embed-text model in parallel.
    
    Args:
        chunks (List[str]): List of text chunks to embed.
        max_workers (int): Maximum number of concurrent workers.
    
    Returns:
        List[List[float]]: List of embeddings, each a 768-dimensional vector.
    """
    if not chunks:
        return []

    try:
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        embeddings = [None] * len(chunks)  # Pre-allocate list to maintain order
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create a future for each chunk
            future_to_index = {
                executor.submit(embed_single_chunk, chunk): idx
                for idx, chunk in enumerate(chunks)
            }
            
            # Process completed futures
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    embedding = future.result()
                    embeddings[idx] = embedding
                    logger.info(f"Processed chunk {idx + 1} of {len(chunks)}")
                except Exception as e:
                    logger.error(f"Failed to process chunk {idx + 1}: {str(e)}")
                    raise

        logger.info("Embeddings generated successfully")
        return embeddings
    except Exception as e:
        logger.error(f"Error generating embeddings with Ollama: {str(e)}")
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