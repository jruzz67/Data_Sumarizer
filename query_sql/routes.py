from fastapi import APIRouter, UploadFile, File, HTTPException, Body, WebSocket
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict
from services import process_audio, text_to_speech, decode_base64_pcm, concatenate_chunks, verify_trailing_silence
from file_handler import extract_text_from_file
from chunker import chunk_text, search_top_chunks
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
import base64
import numpy as np

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
    """Upload a file and store it in the uploads directory."""
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
        logger.info(f"File uploaded: {file.filename}")
        return UploadResponse(filename=file.filename, message="File uploaded successfully")
    except Exception as e:
        logger.error(f"Error uploading file {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_file(request: Dict[str, str] = Body(...)):
    """Analyze an uploaded file, extract text, chunk it, and store embeddings."""
    filename = request.get("filename")
    if not filename:
        logger.warning("Analyze request missing filename")
        raise HTTPException(status_code=400, detail="Filename is required")
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        logger.warning(f"File not found: {filename}")
        raise HTTPException(status_code=404, detail="File not found")
    try:
        text = extract_text_from_file(str(file_path))
        if not text:
            logger.warning(f"No text extracted from: {filename}")
            raise HTTPException(status_code=400, detail="No text extracted from file")
        chunks = chunk_text(text)
        if not chunks:
            logger.warning(f"No chunks created for: {filename}")
            raise HTTPException(status_code=400, detail="No chunks created")
        embeddings = embed_chunks(chunks)
        try:
            with get_db() as conn:
                store_embeddings(chunks, embeddings, filename, conn)
        except psycopg2.Error as e:
            logger.error(f"Error saving embeddings for {filename}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error saving to database: {str(e)}")
        try:
            delete_file(str(file_path))
        except Exception as e:
            logger.warning(f"Failed to delete file {filename}: {str(e)}")
        logger.info(f"File analyzed: {filename}, {len(chunks)} chunks")
        return AnalyzeResponse(
            filename=filename,
            chunk_count=len(chunks),
            message="File analyzed, embeddings stored"
        )
    except Exception as e:
        logger.error(f"Error analyzing {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing file: {str(e)}")

@router.post("/clear", response_model=ClearResponse)
async def clear_data():
    """Clear all data from the chunks table."""
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("TRUNCATE TABLE chunks;")
            conn.commit()
            logger.info("Table 'chunks' cleared")
        return ClearResponse(message="Database cleared successfully")
    except psycopg2.Error as e:
        logger.error(f"Database error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        logger.error(f"Error clearing data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error clearing data: {str(e)}")

@router.post("/clear_audio", response_model=ClearResponse)
async def clear_audio_files():
    """Delete all WAV files in audio directory."""
    try:
        for audio_file in AUDIO_DIR.glob("*.wav"):
            os.remove(audio_file)
            logger.info(f"Deleted: {audio_file}")
        logger.info("Audio files cleared")
        return ClearResponse(message="Audio files cleared successfully")
    except Exception as e:
        logger.error(f"Error clearing audio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error clearing audio: {str(e)}")

@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """Handle WebSocket for real-time audio chat."""
    await websocket.accept()
    logger.info("WebSocket connection established")
    buffer = []
    last_sequence = 0
    silence_count = 0
    min_chunks = 4          # Require 4 chunks (~2s) for transcription
    silence_threshold = 14.0  # RMS threshold for silence detection

    try:
        while True:
            data = await websocket.receive()
            if data["type"] == "websocket.disconnect":
                logger.info("WebSocket disconnected")
                break
            if data["type"] == "websocket.receive" and "text" in data:
                try:
                    message = json.loads(data["text"])
                    sequence = message.get('sequence')
                    silence = message['silence']
                    pcm = decode_base64_pcm(message['pcm'])

                    # Validate sequence
                    if not isinstance(sequence, int):
                        logger.warning(f"Invalid sequence: {sequence}, skipping")
                        continue
                    logger.info(f"Received chunk: sequence={sequence}, silence={silence}, pcm_size={len(pcm)*2} bytes")
                    if sequence < last_sequence:
                        logger.warning(f"Out-of-order chunk: sequence={sequence}, expected>={last_sequence}")
                        continue

                    buffer.append({'sequence': sequence, 'pcm': pcm})
                    # Update silence count: increment if silent, reset if speech
                    silence_count = silence_count + 1 if silence == 0 else 0

                    # Process buffer when we have at least 4 chunks and 2 consecutive silent chunks
                    if len(buffer) >= min_chunks and silence_count >= 2:
                        logger.info(f"Processing buffer: {len(buffer)} chunks, silence_count={silence_count}")
                        concatenated_pcm = concatenate_chunks(buffer)
                        if verify_trailing_silence(concatenated_pcm, threshold=silence_threshold):
                            transcription = await process_audio(concatenated_pcm)
                            if transcription and transcription.strip():
                                await websocket.send_json({
                                    "type": "transcription",
                                    "text": transcription,
                                    "last_sequence": sequence
                                })
                                try:
                                    # Retrieve top chunks for query
                                    top_chunks, chunk_metadata = search_top_chunks(transcription)
                                    # Process query
                                    response = handle_query(transcription, top_chunks, chunk_metadata)
                                    # Send response text
                                    await websocket.send_json({
                                        "type": "response",
                                        "response": response
                                    })
                                    # Generate and stream audio response
                                    pcm_data = text_to_speech(response, "female")
                                    chunk_size = 8192
                                    for i in range(0, len(pcm_data), chunk_size):
                                        chunk = pcm_data[i:i + chunk_size]
                                        pcm_bytes = chunk.tobytes()
                                        pcm_base64 = base64.b64encode(pcm_bytes).decode('utf-8')
                                        await websocket.send_json({
                                            "type": "audio_chunk",
                                            "sequence": i // chunk_size,
                                            "pcm": pcm_base64
                                        })
                                        logger.info(f"Sent audio chunk: sequence={i // chunk_size}, size={len(pcm_bytes)} bytes")
                                    logger.info(f"Query processed: {transcription}")
                                    # Store transcription in database
                                    chunks = chunk_text(transcription)
                                    if chunks:
                                        embeddings = embed_chunks(chunks)
                                        with get_db() as conn:
                                            store_embeddings(chunks, embeddings, f"audio_{int(time.time())}.wav", conn)
                                        logger.info(f"Stored {len(chunks)} chunks from transcription")
                                except Exception as e:
                                    logger.error(f"Error processing query: {str(e)}")
                                    await websocket.send_json({"type": "error", "message": str(e)})
                        buffer = []
                        silence_count = 0
                    last_sequence = sequence
                except Exception as e:
                    logger.error(f"Error processing message: {str(e)}")
                    await websocket.send_json({"type": "error", "message": str(e)})
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        await websocket.send_json({"type": "error", "message": str(e)})
    finally:
        await websocket.close()
        logger.info("WebSocket closed")