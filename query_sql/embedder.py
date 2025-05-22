import os
import logging
from typing import List
from dotenv import load_dotenv
import ollama
from tenacity import retry, stop_after_attempt, wait_exponential
from concurrent.futures import ThreadPoolExecutor, as_completed
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

EXPECTED_DIMENSION = 768

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
        start_time = time.time()
        response = ollama.embeddings(
            model="nomic-embed-text:latest",
            prompt=chunk
        )
        embedding = response["embedding"]
        if len(embedding) != EXPECTED_DIMENSION:
            raise ValueError(f"Unexpected embedding dimension: {len(embedding)} (expected {EXPECTED_DIMENSION})")
        logger.info(f"Generated embedding for chunk in {time.time() - start_time:.3f}s")
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
        start_time = time.time()
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        embeddings = [None] * len(chunks)  # Pre-allocate list to maintain order
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(embed_single_chunk, chunk): idx
                for idx, chunk in enumerate(chunks)
            }
            
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    embedding = future.result()
                    embeddings[idx] = embedding
                    logger.info(f"Processed chunk {idx + 1} of {len(chunks)}")
                except Exception as e:
                    logger.error(f"Failed to process chunk {idx + 1}: {str(e)}")
                    raise

        logger.info(f"Embeddings generated for {len(chunks)} chunks in {time.time() - start_time:.3f}s")
        return embeddings
    except Exception as e:
        logger.error(f"Error generating embeddings with Ollama: {str(e)}")
        raise

def store_embeddings(chunks: List[str], embeddings: List[List[float]], document_name: str, conn):
    """
    Store chunks and their embeddings in the database using batch inserts.
    
    Args:
        chunks (List[str]): List of text chunks.
        embeddings (List[List[float]]): List of embeddings.
        document_name (str): Name of the document.
        conn: Database connection.
    """
    try:
        start_time = time.time()
        cursor = conn.cursor()
        # Prepare batch insert data
        records = [
            (document_name, chunk, embedding)
            for chunk, embedding in zip(chunks, embeddings)
        ]
        # Validate embeddings
        for i, emb in enumerate(embeddings):
            if len(emb) != EXPECTED_DIMENSION:
                raise ValueError(f"Embedding {i} has dimension {len(emb)}, expected {EXPECTED_DIMENSION}")
        # Batch insert
        cursor.executemany(
            """
            INSERT INTO chunks (document_name, chunk_text, embedding)
            VALUES (%s, %s, %s);
            """,
            records
        )
        conn.commit()
        logger.info(f"Stored {len(chunks)} chunks and embeddings for document: {document_name} in {time.time() - start_time:.3f}s")
    except Exception as e:
        logger.error(f"Error storing embeddings for {document_name}: {str(e)}")
        conn.rollback()
        raise