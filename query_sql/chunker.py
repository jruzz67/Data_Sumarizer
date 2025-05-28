import logging
from embedder import embed_single_chunk
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
logger = logging.getLogger("__name__")

def chunk_text(text: str, max_chunk_size: int = None) -> list:
    """
    Split text into chunks of specified size.
    Returns a list of text chunks.
    
    """
    # Type checking
    if not isinstance(text, str):
        raise TypeError(f"Expected text to be str, got {type(text)}")

    if not text:
        return []

    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        word_length = len(word) + 1  # +1 for space
        if current_length + word_length <= max_chunk_size:
            current_chunk.append(word)
            current_length += word_length
        
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = word_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def search_top_chunks(query_text: str, top_k: int = 5) -> tuple[list, list]:
    """
    Perform a vector search to retrieve the top_k most relevant chunks for a query.
    
    Args:
        query_text (str): The query text to search for.
        top_k (int): Number of top chunks to retrieve.
    
    Returns:
        tuple: (list of chunk texts, list of metadata dictionaries with document_name and cosine_distance)
    """
    try:
        # Embed the query text
        query_embedding = embed_single_chunk(query_text)
        # Convert embedding to vector string format
        query_embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
        with get_db() as conn:
            cursor = conn.cursor()
            # Vector search using cosine distance
            cursor.execute(
                """
                SELECT document_name, chunk_text, embedding <=> %s::vector AS cosine_distance
                FROM chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
                """,
                (query_embedding_str, query_embedding_str, top_k)
            )
            results = cursor.fetchall()
        # Extract chunks and metadata
        top_chunks = [row['chunk_text'] for row in results]
        chunk_metadata = [
            {"document_name": row['document_name'], "cosine_distance": row['cosine_distance']}
            for row in results
        ]
        logger.info(f"Retrieved {len(top_chunks)} chunks for query: {query_text}")
        return top_chunks, chunk_metadata
    except Exception as e:
        logger.error(f"Error in vector search: {str(e)}")
        return [], []