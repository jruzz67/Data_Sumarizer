import os
from dotenv import load_dotenv

load_dotenv()
print(f"OPENAI_API_KEY: {os.getenv('OPENAI_API_KEY')}")
print(f"DB_NAME: {os.getenv('DB_NAME')}")