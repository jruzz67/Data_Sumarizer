from fastapi import APIRouter, UploadFile, File, HTTPException, Body, WebSocket
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Dict
from services import process_audio, text_to_speech, cleanup_old_audio_files
from file_handler import extract_text_from_file
from chunker import chunk_text
from embedder import embed_chunks, store_embeddings
from query_handler import handle_query
from utils import delete_file
from db import get_db
import os
import time
from pathlib import Path
import logging
import psycopg2
import shutil
import json

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_FILE_SIZE = 10 * 1024 * 1024
UPLOAD_DIR = Path("uploads")
AUDIO_DIR = Path("audio")

class UploadResponse(BaseModel):
    filename: str
    message: str

class AnalyzeResponse(BaseModel):
    filename: str
    chunk_count: int
    message: str

class ClearResponse(BaseModel):
    message: str

@router.post("/upload", response_model=UploadResponse)
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

@router.post("/analyze", response_model=AnalyzeResponse)
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

@router.get("/audio/{filename}")
async def get_audio(filename: str):
    audio_path = AUDIO_DIR / filename
    if not audio_path.exists():
        logger.warning(f"Audio file not found: {filename}")
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(audio_path, media_type="audio/wav", headers={"Access-Control-Allow-Origin": "*"})

@router.post("/clear", response_model=ClearResponse)
async def clear_data():
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("TRUNCATE TABLE chunks;")
            conn.commit()
            logger.info("Database table 'chunks' truncated successfully")
        return ClearResponse(message="Database cleared successfully")
    except psycopg2.Error as e:
        logger.error(f"Database error while clearing data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        logger.error(f"Error clearing data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error clearing data: {str(e)}")

@router.post("/clear_audio", response_model=ClearResponse)
async def clear_audio_files():
    try:
        for audio_file in AUDIO_DIR.glob("*.wav"):
            os.remove(audio_file)
            logger.info(f"Deleted audio file: {audio_file}")
        logger.info("All audio files cleared successfully")
        return ClearResponse(message="Audio files cleared successfully")
    except Exception as e:
        logger.error(f"Error clearing audio files: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error clearing audio files: {str(e)}")

@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established for chat")
    last_speech_time = time.time()  # Track last non-empty transcription
    try:
        while True:
            data = await websocket.receive()
            if data["type"] == "websocket.disconnect":
                logger.info("WebSocket disconnected")
                break

            if data["type"] == "websocket.receive" and "bytes" in data:
                audio_chunk = data["bytes"]
                logger.info(f"Received PCM of size: {len(audio_chunk)} bytes")
                if not audio_chunk:
                    logger.warning("Empty PCM received")
                    continue

                transcription = await process_audio(audio_chunk)
                current_time = time.time()

                if not transcription:
                    # Only send "No speech detected" if 10 seconds have passed since last speech
                    if current_time - last_speech_time >= 10:
                        logger.info("No speech detected for 10 seconds")
                        await websocket.send_json({
                            "type": "response",
                            "response": "No speech detected",
                            "audio_url": None
                        })
                    continue

                # Update last speech time on valid transcription
                last_speech_time = current_time

                try:
                    with get_db() as conn:
                        cursor = conn.cursor()
                        query_embedding = embed_chunks([transcription])[0]
                        cursor.execute(
                            """
                            SELECT chunk_text, document_name, embedding <=> CAST(%s AS vector) AS cosine_distance
                            FROM chunks
                            ORDER BY embedding <=> CAST(%s AS vector)
                            LIMIT 5;
                            """,
                            (query_embedding, query_embedding)
                        )
                        results = cursor.fetchall()
                        top_chunks = [row["chunk_text"] for row in results]
                        chunk_metadata = [
                            {"text": row["chunk_text"], "document_name": row["document_name"], "cosine_distance": row["cosine_distance"]}
                            for row in results
                        ]

                    if not top_chunks:
                        logger.info("No relevant chunks found")
                        audio_filename = f"no_chunks_{int(time.time())}.wav"
                        audio_path = AUDIO_DIR / audio_filename
                        text_to_speech("No relevant chunks found in the database.", str(audio_path), "female")
                        audio_url = f"/audio/{audio_filename}"
                        cleanup_old_audio_files()
                        await websocket.send_json({
                            "type": "response",
                            "response": "No relevant chunks found",
                            "audio_url": audio_url
                        })
                    else:
                        logger.info(f"Retrieved {len(top_chunks)} chunks")
                        for meta in chunk_metadata:
                            logger.info(f"Chunk from {meta['document_name']}: {meta['text'][:50]}... (cosine distance: {meta['cosine_distance']:.4f})")
                        response = handle_query(transcription, top_chunks, chunk_metadata)
                        audio_filename = f"response_{int(time.time())}.wav"
                        audio_path = AUDIO_DIR / audio_filename
                        text_to_speech(response, str(audio_path), "female")
                        audio_url = f"/audio/{audio_filename}"
                        cleanup_old_audio_files()
                        await websocket.send_json({
                            "type": "response",
                            "response": response,
                            "audio_url": audio_url
                        })
                        logger.info(f"Query processed: {transcription}")

                    if transcription:
                        chunks = chunk_text(transcription)
                        embeddings = embed_chunks(chunks)
                        with get_db() as conn:
                            store_embeddings(chunks, embeddings, f"audio_{int(time.time())}.wav", conn)
                        logger.info(f"Stored {len(chunks)} chunks from transcription")

                except Exception as e:
                    logger.error(f"Error processing query: {str(e)}")
                    await websocket.send_json({"type": "error", "message": str(e)})

    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        await websocket.send_json({"type": "error", "message": str(e)})
        await websocket.close()
        logger.info("WebSocket connection closed")