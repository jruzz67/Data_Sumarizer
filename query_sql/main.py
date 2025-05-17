from fastapi import FastAPI, UploadFile, File, HTTPException, Body, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import shutil
from pathlib import Path
from typing import Dict
from dotenv import load_dotenv
from file_handler import extract_text_from_file
from chunker import chunk_text
from embedder import embed_chunks, store_embeddings
from query_handler import handle_query
from utils import delete_file
from db import init_db, get_db
import logging
import psycopg2

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

app = FastAPI(
    title="QuerySQL Backend",
    description="A FastAPI backend for document processing and querying.",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure uploads directory exists
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Maximum file size: 10MB
MAX_FILE_SIZE = 10 * 1024 * 1024

# Response models
class UploadResponse(BaseModel):
    filename: str
    message: str

class AnalyzeResponse(BaseModel):
    filename: str
    chunk_count: int
    message: str

class QueryResponse(BaseModel):
    response: str

# Initialize database
@app.on_event("startup")
async def startup_event():
    try:
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
        raise

@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    allowed_extensions = {".pdf", ".txt", ".xlsx", ".xls"}
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        logger.warning(f"Unsupported file type uploaded: {file_ext}")
        raise HTTPException(status_code=400, detail="Unsupported file type. Allowed: PDF, TXT, Excel")
    
    file_size = 0
    for chunk in file.file:
        file_size += len(chunk)
        if file_size > MAX_FILE_SIZE:
            logger.warning(f"File too large: {file.filename}, size: {file_size} bytes")
            raise HTTPException(status_code=400, detail="File too large. Max size: 10MB")
    file.file.seek(0)

    file_path = UPLOAD_DIR / file.filename
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"File uploaded successfully: {file.filename}")
        return UploadResponse(filename=file.filename, message="File uploaded successfully")
    except Exception as e:
        logger.error(f"Error uploading file {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_file(request: Dict[str, str] = Body(...)):
    filename = request.get("filename")
    if not filename:
        logger.warning("Analyze request missing filename")
        raise HTTPException(status_code=400, detail="Filename is required")

    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        logger.warning(f"File not found for analysis: {filename}")
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        text = extract_text_from_file(str(file_path))
        if not text:
            logger.warning(f"No text extracted from file: {filename}")
            raise HTTPException(status_code=400, detail="No text extracted from file")
        
        chunks = chunk_text(text)
        if not chunks:
            logger.warning(f"No chunks generated for file: {filename}")
            raise HTTPException(status_code=400, detail="No chunks generated")
        
        embeddings = embed_chunks(chunks)
        
        try:
            with get_db() as conn:
                store_embeddings(chunks, embeddings, filename, conn)
        except psycopg2.Error as e:
            logger.error(f"Database error while storing embeddings for {filename}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
        
        try:
            delete_file(str(file_path))
        except Exception as e:
            logger.warning(f"Failed to delete file {filename}: {str(e)}")
        
        logger.info(f"File analyzed successfully: {filename}, chunks: {len(chunks)}")
        return AnalyzeResponse(
            filename=filename,
            chunk_count=len(chunks),
            message="File analyzed and embeddings stored"
        )
    except Exception as e:
        logger.error(f"Error analyzing file {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing file: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query(request: Dict[str, str] = Body(...)):
    query_text = request.get("query")
    if not query_text:
        logger.warning("Query request missing query text")
        raise HTTPException(status_code=400, detail="Query text is required")
    
    try:
        query_embedding = embed_chunks([query_text])[0]
        
        try:
            with get_db() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT chunk_text
                    FROM chunks
                    ORDER BY embedding <-> CAST(%s AS vector)
                    LIMIT 5;
                    """,
                    (query_embedding,)
                )
                top_chunks = [row["chunk_text"] for row in cursor.fetchall()]
        except psycopg2.Error as e:
            logger.error(f"Database error during vector search: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
        
        if not top_chunks:
            logger.info("No relevant chunks found for query")
            return QueryResponse(response="No relevant chunks found")
        
        response = handle_query(query_text, top_chunks)
        
        logger.info(f"Query processed successfully: {query_text}")
        return QueryResponse(response=response)
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)