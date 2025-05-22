import os
import logging
from dotenv import load_dotenv
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential
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

# Initialize Gemini API
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables")
genai.configure(api_key=gemini_api_key)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def handle_query(query_text: str, top_chunks: list, chunk_metadata: list = None) -> str:
    """
    Generate a response to the user's query using Gemini 1.5 Flash based on the top relevant chunks.
    
    Args:
        query_text (str): The user's query.
        top_chunks (list): List of top relevant text chunks from vector search.
        chunk_metadata (list): List of dictionaries with chunk text, document name, and cosine distance.
    
    Returns:
        str: The generated response.
    """
    if not top_chunks:
        logger.info("No relevant chunks found for query")
        return "No relevant chunks found"

    try:
        start_time = time.time()
        logger.info(f"Generating response for query: {query_text}")
        
        # Build context with metadata
        context_parts = []
        for chunk, meta in zip(top_chunks, chunk_metadata or [{}] * len(top_chunks)):
            doc_name = meta.get("document_name", "Unknown")
            distance = meta.get("cosine_distance", "N/A")
            context_parts.append(f"Document: {doc_name}\nText: {chunk}\n(Similarity: {distance})")
        context = "\n\n".join(context_parts)
        
        prompt = (
            "You are a helpful assistant. Based on the following document context, "
            f"answer the query: '{query_text}'\n\n"
            "Document Context:\n"
            f"{context}\n\n"
            "Provide a concise and accurate response, referencing the document names where relevant."
        )

        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        
        response_text = response.text.strip()
        logger.info(f"Response generated in {time.time() - start_time:.3f}s")
        return response_text
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise