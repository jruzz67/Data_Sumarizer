import logging
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

# Test connection to Ollama and Mistral model
logger.info("Testing connection to Mistral via Ollama...")
try:
    test_input = "Test input for model loading."
    test_response = ollama.generate(model="mistral", prompt=test_input)
    logger.info(f"Test inference successful: {test_response['response']}")
except Exception as e:
    logger.error(f"Failed to connect to Mistral via Ollama: {str(e)}")
    raise
logger.info("Ollama connection and Mistral model verified.")

def handle_query(query_text: str, top_chunks: list) -> str:
    """
    Generate a response to the user's query using Mistral via Ollama based on the top relevant chunks.
    
    Args:
        query_text (str): The user's query.
        top_chunks (list): List of top relevant text chunks from vector search.
    
    Returns:
        str: The generated response.
    """
    if not top_chunks:
        logger.info("No relevant chunks found for query")
        return "No relevant chunks found"

    try:
        logger.info(f"Generating response for query: {query_text}")
        logger.info(f"Top chunks: {top_chunks}")
        context = "\n".join(top_chunks)
        
        # Updated prompt with role-playing, few-shot learning, delimiters, and step-by-step instructions
        prompt = (
            "You are a knowledgeable assistant specializing in document analysis. Your task is to answer questions based on the provided document context concisely and accurately.\n\n"
            "### Examples\n"
            "Example 1:\n"
            "Context: John Doe, a software engineer with 5 years of experience, specializes in Python and JavaScript. He graduated with a Bachelor's degree in Computer Science from MIT in 2020.\n"
            "Question: What is John's qualification?\n"
            "Answer: John has a Bachelor's degree in Computer Science from MIT, graduated in 2020.\n\n"
            "Example 2:\n"
            "Context: The company offers a 30-year fixed-rate mortgage at 6.5% APR and a 15-year fixed-rate mortgage at 5.8% APR.\n"
            "Question: What is the 30-year fixed-rate APR?\n"
            "Answer: The 30-year fixed-rate APR is 6.5%.\n\n"
            "### Task Instructions\n"
            "Step 1: Read and understand the context provided below.\n"
            "Step 2: Use the context to answer the user's question concisely and accurately.\n"
            "Step 3: Provide the answer in plain text, without repeating the question or context.\n\n"
            "### Context\n"
            f"{context}\n\n"
            "### Question\n"
            f"{query_text}\n\n"
            "### Answer\n"
        )

        # Generate response with Mistral via Ollama
        response = ollama.generate(
            model="mistral",
            prompt=prompt,
            options={
                "num_predict": 100,  # Limit response length for faster generation
                "temperature": 0.7,  # Balance creativity and coherence
                "top_p": 0.9  # Use nucleus sampling for diversity
            }
        )
        response_text = response["response"].strip()
        
        # Clean up the response to remove the prompt part
        response_text = response_text.split("### Answer")[-1].strip() if "### Answer" in response_text else response_text
        
        logger.info("Response generated successfully")
        return response_text if response_text else "I couldn't generate a meaningful response."
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise