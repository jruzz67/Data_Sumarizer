from fastapi import UploadFile, File, HTTPException, Body
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import os
import shutil
import time
import logging
import psycopg2.extras  # Import for DictCursor
from typing import Dict
from utils.config import UPLOAD_DIR, AUDIO_DIR, MAX_FILE_SIZE
from core.file_handler import extract_text_from_file
from core.chunker import chunk_text
from core.embedder import embed_chunks, store_embeddings
from core.query_handler import handle_query
from utils.utils import delete_file, cleanup_old_audio_files
from core.db import get_db
from utils.audio import speech_processor  # Direct import of speech_processor
from pydub import AudioSegment  # Direct import of AudioSegment for file conversion
from io import BytesIO  # For in-memory file handling

logger = logging.getLogger(__name__)

class UploadResponse(BaseModel):
    filename: str
    message: str

class AnalyzeResponse(BaseModel):
    filename: str
    chunk_count: int
    message: str

class QueryRequest(BaseModel):
    query: str
    voice_model: str = "female"

class QueryResponse(BaseModel):
    response: str
    audio_url: str

class TranscribeResponse(BaseModel):
    transcribed_text: str

class ClearResponse(BaseModel):
    message: str

def register_routes(app):
    @app.post("/upload", response_model=UploadResponse)
    async def upload_file(file: UploadFile = File(...)):
        allowed_extensions = {".pdf", ".txt", ".xlsx", ".xls"}
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            logger.warning(f"Unsupported file type uploaded: {file_ext}")
            raise HTTPException(status_code=400, detail="Unsupported file type. Allowed: PDF, TXT, Excel")
        
        file_content = await file.read()  # Read content once
        file_size = len(file_content)

        if file_size > MAX_FILE_SIZE:
            logger.warning(f"File too large: {file.filename}, size: {file_size} bytes")
            raise HTTPException(status_code=400, detail="File too large. Max size: 10MB")
        
        file_path = UPLOAD_DIR / file.filename
        try:
            with open(file_path, "wb") as buffer:
                buffer.write(file_content)
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
                    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
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

    @app.post("/transcribe", response_model=TranscribeResponse)
    async def transcribe_audio_file(file: UploadFile = File(...)):
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in {".webm", ".wav", ".mp3"}:
            logger.warning(f"Unsupported audio file type: {file_ext}")
            raise HTTPException(status_code=400, detail="Unsupported audio file type. Allowed: WebM, WAV, MP3")
        
        try:
            # Read the entire file content into a BytesIO buffer
            audio_buffer = BytesIO(await file.read())
            audio_buffer.seek(0)

            # Use pydub to convert to 16kHz mono 16-bit PCM in memory
            audio = AudioSegment.from_file(audio_buffer, format=file_ext.lstrip('.'))
            audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
            
            # Get raw PCM data
            pcm_data = audio.raw_data

            # Transcribe the raw PCM data using speech_processor
            # For file uploads, session_id is not relevant, so use 0
            session_id = 0
            transcribed_text = await speech_processor.transcribe_audio(pcm_data, session_id=session_id, current_session_id=session_id)

            logger.info(f"Audio file '{file.filename}' transcribed successfully (session {session_id}): {transcribed_text}")
            return TranscribeResponse(transcribed_text=transcribed_text)
        except Exception as e:
            logger.error(f"Error transcribing audio file '{file.filename}': {str(e)}")
            if "ffmpeg" in str(e).lower() or "ffprobe" in str(e).lower():
                raise HTTPException(status_code=500, detail=f"Audio processing error: Ensure FFmpeg is installed and in your system's PATH. Details: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error transcribing audio file: {str(e)}")

    @app.post("/query", response_model=QueryResponse)
    async def query(request: QueryRequest = Body(...)):
        query_text = request.query
        voice_model = request.voice_model
        logger.info(f"Received query with voice_model: {voice_model}")
        if not query_text:
            logger.warning("Query request missing query text")
            raise HTTPException(status_code=400, detail="Query text is required")
        try:
            query_embedding = embed_chunks([query_text])[0]
            try:
                with get_db() as conn:
                    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
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
                response_text = "No relevant chunks found in the database."
            else:
                response_text = handle_query(query_text, top_chunks)
            
            # Generate audio response
            tts_file_path = await speech_processor.text_to_speech(response_text, voice_model)
            
            # Move the generated file to the public AUDIO_DIR
            audio_filename = os.path.basename(tts_file_path)
            final_audio_path = AUDIO_DIR / audio_filename
            shutil.move(tts_file_path, final_audio_path)
            
            audio_url = f"/audio/{audio_filename}"
            logger.info(f"Query processed successfully: {query_text}")
            cleanup_old_audio_files()
            return QueryResponse(response=response_text, audio_url=audio_url)
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

    @app.get("/audio/{filename}")
    async def get_audio(filename: str):
        audio_path = AUDIO_DIR / filename
        if not audio_path.exists():
            logger.warning(f"Audio file not found: {filename}")
            raise HTTPException(status_code=404, detail="Audio file not found")
        return FileResponse(audio_path, media_type="audio/wav", headers={"Access-Control-Allow-Origin": "*"})

    @app.post("/clear", response_model=ClearResponse)
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

    @app.post("/clear_audio", response_model=ClearResponse)
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