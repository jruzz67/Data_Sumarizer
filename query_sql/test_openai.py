import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

try:
    embedding = client.embeddings.create(model="text-embedding-ada-002", input="Test").data[0].embedding
    print(f"Embedding length: {len(embedding)}")  # Should print 1536
except Exception as e:
    print(f"Error: {str(e)}")