from fastapi import FastAPI, UploadFile, File, HTTPException, Body, Depends, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import shutil
import time
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
from vosk import Model, KaldiRecognizer
import wave
from pydub import AudioSegment
from TTS.api import TTS
import torch.serialization
import collections  # For allowlisting defaultdict

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

AUDIO_DIR = Path("audio")
AUDIO_DIR.mkdir(exist_ok=True)

MAX_FILE_SIZE = 10 * 1024 * 1024

VOSK_MODEL_PATH = "models/vosk-model-small-en-us-0.15/vosk-model-small-en-us-0.15"
if not os.path.exists(VOSK_MODEL_PATH):
    logger.error(f"Vosk model not found at {VOSK_MODEL_PATH}")
    raise FileNotFoundError(f"Vosk model not found at {VOSK_MODEL_PATH}")
vosk_model = Model(VOSK_MODEL_PATH)
logger.info("Vosk model loaded successfully")

VOICE_MODELS = {
    "female": {"model_name": "tts_models/en/ljspeech/tacotron2-DDC", "speaker": None},
    "male": {"model_name": "tts_models/en/vctk/vits", "speaker": "p239"},
}

# Ensure espeak-ng is accessible by adding its path to the environment
espeak_path = r"C:\Program Files\eSpeak NG"
if os.path.exists(espeak_path):
    os.environ["PATH"] = espeak_path + os.pathsep + os.environ.get("PATH", "")
    logger.info(f"Added {espeak_path} to PATH for espeak-ng")
else:
    logger.warning(f"espeak-ng path {espeak_path} not found. VITS models may fail to load.")

# Allowlist RAdam, collections.defaultdict, and dict for PyTorch
try:
    from TTS.utils.radam import RAdam
    torch.serialization.add_safe_globals([RAdam, collections.defaultdict, dict])
    logger.info("Added TTS.utils.radam.RAdam, collections.defaultdict, and dict to PyTorch safe globals")
except ImportError:
    logger.error("Failed to import TTS.utils.radam.RAdam for allowlisting")
    raise

# TTS model directory
TTS_MODEL_DIR = Path("C:/Users/ASUS/AppData/Local/tts")

def clean_tts_model_directory(model_name: str):
    """Remove a TTS model directory if it exists."""
    model_dir = TTS_MODEL_DIR / model_name.replace('/', '--')
    if model_dir.exists():
        try:
            shutil.rmtree(model_dir)
            logger.info(f"Cleaned up TTS model directory: {model_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up TTS model directory {model_dir}: {str(e)}")

# Clean up unused TTS models
UNUSED_MODELS = [
    "tts_models/en/jenny/jenny",
    "tts_models/en/ljspeech/glow-tts",
    "tts_models/en/ek1/tacotron2",
    "tts_models/en/ljspeech/fast_pitch",
    # "tts_models/en/vctk/vits" is cleaned up manually above and will be redownloaded
]
for model_name in UNUSED_MODELS:
    clean_tts_model_directory(model_name)

TTS_MODELS = {}
try:
    for voice_label, config in VOICE_MODELS.items():
        model_name = config["model_name"]
        if model_name not in TTS_MODELS:
            # Check if model directory exists
            model_dir = TTS_MODEL_DIR / model_name.replace('/', '--')
            if model_dir.exists():
                logger.info(f"TTS model directory exists: {model_dir}")
            else:
                logger.info(f"TTS model directory not found, will download: {model_dir}")

            try:
                logger.info(f"Attempting to load TTS model: {model_name}")
                TTS_MODELS[model_name] = TTS(model_name=model_name, progress_bar=False)
                logger.info(f"Successfully loaded TTS model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load TTS model {model_name}: {str(e)}")
                # Clean up the model directory to prevent repeated download attempts
                clean_tts_model_directory(model_name)
                # Fallback to a default model if loading fails
                if model_name != "tts_models/en/ljspeech/tacotron2-DDC":
                    logger.info(f"Falling back to default TTS model for {voice_label}")
                    TTS_MODELS[model_name] = TTS_MODELS.get("tts_models/en/ljspeech/tacotron2-DDC")
                else:
                    raise
except Exception as e:
    logger.error(f"Critical failure in loading TTS models: {str(e)}")
    raise

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

def convert_audio_to_wav(audio_data: bytes, output_path: str) -> None:
    try:
        if not audio_data:
            logger.error("Audio data is empty")
            raise ValueError("Audio data is empty")

        temp_webm = UPLOAD_DIR / f"temp_audio_{int(time.time())}.webm"
        with open(temp_webm, "wb") as f:
            f.write(audio_data)

        if not os.path.exists(temp_webm) or os.path.getsize(temp_webm) == 0:
            logger.error("Temporary WebM file is empty or not created")
            raise ValueError("Temporary WebM file is empty or not created")

        audio = AudioSegment.from_file(temp_webm, format="webm")
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        audio.export(output_path, format="wav")
        os.remove(temp_webm)
    except Exception as e:
        logger.error(f"Error converting audio to WAV: {str(e)}")
        if os.path.exists(temp_webm):
            os.remove(temp_webm)
        raise

def speech_to_text(audio_path: str) -> str:
    try:
        wf = wave.open(audio_path, "rb")
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
            raise ValueError("Audio file must be WAV format, mono, 16-bit, 16 kHz")
        recognizer = KaldiRecognizer(vosk_model, wf.getframerate())
        recognizer.SetWords(True)
        transcribed_text = ""
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if recognizer.AcceptWaveform(data):
                result = recognizer.Result()
                text = eval(result).get("text", "")
                transcribed_text += text + " "
        final_result = recognizer.FinalResult()
        text = eval(final_result).get("text", "")
        transcribed_text += text
        transcribed_text = transcribed_text.strip()
        wf.close()
        return transcribed_text if transcribed_text else "No speech detected."
    except Exception as e:
        logger.error(f"Error in speech-to-text: {str(e)}")
        raise

def text_to_speech(text: str, output_path: str, voice_model: str = "female") -> None:
    try:
        logger.info(f"Using voice model in text_to_speech: {voice_model}")
        if voice_model not in VOICE_MODELS:
            logger.warning(f"Invalid voice model: {voice_model}, falling back to default 'female'")
            voice_model = "female"

        config = VOICE_MODELS[voice_model]
        model_name = config["model_name"]
        tts_model = TTS_MODELS.get(model_name)

        if not tts_model:
            logger.error(f"TTS model {model_name} not loaded, falling back to default")
            voice_model = "female"
            config = VOICE_MODELS[voice_model]
            tts_model = TTS_MODELS[config["model_name"]]

        speaker = config["speaker"]
        logger.info(f"Generating TTS with model: {model_name}, speaker: {speaker}")

        if speaker:
            tts_model.tts_to_file(text=text, file_path=output_path, speaker=speaker)
        else:
            tts_model.tts_to_file(text=text, file_path=output_path, speaker_wav=None)
        logger.info(f"Generated speech saved to {output_path} with voice model {voice_model}")
    except Exception as e:
        logger.error(f"Error in text-to-speech with voice model {voice_model}: {str(e)}")
        raise

def cleanup_old_audio_files(max_age_seconds=3600):
    try:
        current_time = time.time()
        for audio_file in AUDIO_DIR.glob("*.wav"):
            file_age = current_time - os.path.getmtime(audio_file)
            if file_age > max_age_seconds:
                os.remove(audio_file)
                logger.info(f"Deleted old audio file: {audio_file}")
    except Exception as e:
        logger.error(f"Error cleaning up audio files: {str(e)}")

@app.on_event("startup")
async def startup_event():
    try:
        init_db()
        logger.info("Database initialized successfully")
        cleanup_old_audio_files()
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

@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in {".webm"}:
        logger.warning(f"Unsupported audio file type: {file_ext}")
        raise HTTPException(status_code=400, detail="Unsupported audio file type. Allowed: WebM")
    try:
        audio_data = await file.read()
        temp_wav = UPLOAD_DIR / f"temp_{int(time.time())}.wav"
        convert_audio_to_wav(audio_data, str(temp_wav))
        transcribed_text = speech_to_text(str(temp_wav))
        delete_file(str(temp_wav))
        logger.info(f"Audio transcribed successfully: {transcribed_text}")
        return TranscribeResponse(transcribed_text=transcribed_text)
    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error transcribing audio: {str(e)}")

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
            audio_filename = f"no_chunks_{int(time.time())}.wav"
            audio_path = AUDIO_DIR / audio_filename
            text_to_speech("No relevant chunks found in the database.", str(audio_path), voice_model)
            audio_url = f"/audio/{audio_filename}"
            cleanup_old_audio_files()
            return QueryResponse(response="No relevant chunks found", audio_url=audio_url)
        response = handle_query(query_text, top_chunks)
        audio_filename = f"response_{int(time.time())}.wav"
        audio_path = AUDIO_DIR / audio_filename
        text_to_speech(response, str(audio_path), voice_model)
        audio_url = f"/audio/{audio_filename}"
        logger.info(f"Query processed successfully: {query_text}")
        cleanup_old_audio_files()
        return QueryResponse(response=response, audio_url=audio_url)
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)