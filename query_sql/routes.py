from fastapi import APIRouter, UploadFile, File, HTTPException, Body, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response
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
from pydub import AudioSegment
import asyncio
import webrtcvad
from scipy.signal import resample as scipy_resample
import audioop  # For explicit MULAW conversion
import soundfile as sf  # For saving WAV files for debugging

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_FILE_SIZE = 10 * 1024 * 1024
UPLOAD_DIR = Path("uploads")
AUDIO_DIR = Path("audio")

def resample(audio_data: np.ndarray, target_samples: int) -> np.ndarray:
    """
    Resample audio data to a target number of samples.
    
    Args:
        audio_data: Input audio as a numpy array (int16)
        target_samples: Number of samples in the resampled audio
    
    Returns:
        Resampled audio as a numpy array (int16)
    """
    # Convert to float for resampling
    audio_float = audio_data.astype(np.float32)
    resampled = scipy_resample(audio_float, target_samples)
    return resampled.astype(np.int16)

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

@router.post("/voice")
async def voice_webhook():
    """Handle Twilio voice webhook with TwiML response to stream audio to WebSocket."""
    ws_url = "wss://d65d-2409-40f4-101c-c09-8cf6-8a8f-d142-9955.ngrok-free.app/ws/twilio"
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Amy">Good morning, how can I help you? Please speak now.</Say>
    <Connect>
        <Stream url="{ws_url}" />
    </Connect>
    <Say voice="Polly.Amy">No speech detected. Goodbye.</Say>
</Response>"""
    logger.info(f"Sending TwiML with WebSocket URL: {ws_url}")
    return Response(content=twiml, media_type="application/xml")
# ... (Previous imports and other endpoints remain unchanged)

async def send_audio_response(websocket, stream_sid, chunk_size=8192):
    """Helper function to send predefined audio file as response to Twilio."""
    try:
        # Path to the predefined audio file
        audio_file_path = r"D:\Projects\OCR_with_PostgresSQL\query_sql\audio\Thank you for contac.wav"
        
        # Check if file exists
        if not os.path.exists(audio_file_path):
            logger.error(f"Audio file not found: {audio_file_path}")
            # Send a fallback message via Twilio
            await websocket.send_text(json.dumps({
                "event": "media",
                "streamSid": stream_sid,
                "media": {
                    "payload": base64.b64encode(b"").decode('utf-8')  # Empty payload to trigger TwiML
                }
            }))
            await websocket.send_text(json.dumps({
                "event": "stop",
                "streamSid": stream_sid
            }))
            return

        # Load the audio file
        logger.info(f"Attempting to load audio file: {audio_file_path}")
        audio_segment = AudioSegment.from_wav(audio_file_path)
        logger.info(f"Loaded audio file: {audio_file_path}, duration={audio_segment.duration_seconds:.2f}s, sample_rate={audio_segment.frame_rate} Hz")

        # Resample to 8000 Hz for Twilio
        audio_segment = audio_segment.set_frame_rate(8000)
        audio_segment = audio_segment.set_channels(1)  # Ensure mono
        audio_segment = audio_segment.set_sample_width(2)  # 16-bit
        pcm_data = np.array(audio_segment.get_array_of_samples(), dtype=np.int16)
        logger.info(f"Resampled PCM data: {len(pcm_data)} samples at 8000 Hz")

        # Normalize audio to prevent clipping
        max_val = np.max(np.abs(pcm_data))
        if max_val > 0:
            pcm_data = (pcm_data / max_val * 32767).astype(np.int16)
        logger.info(f"Normalized PCM data: min={np.min(pcm_data)}, max={np.max(pcm_data)}")

        # Convert PCM to MULAW for Twilio using audioop
        pcm_bytes = pcm_data.tobytes()
        mulaw_audio = audioop.lin2ulaw(pcm_bytes, 2)  # 2 bytes per sample (16-bit)
        logger.info(f"Converted to MULAW: {len(mulaw_audio)} bytes")

        # Send audio back to Twilio in chunks
        for i in range(0, len(mulaw_audio), chunk_size):
            chunk = mulaw_audio[i:i + chunk_size]
            pcm_base64 = base64.b64encode(chunk).decode('utf-8')
            await websocket.send_text(json.dumps({
                "event": "media",
                "streamSid": stream_sid,
                "media": {
                    "payload": pcm_base64
                }
            }))
            logger.info(f"Sent Twilio audio chunk: size={len(chunk)} bytes")
            await asyncio.sleep(0.1)  # Simulate real-time streaming
    except Exception as e:
        logger.error(f"Error in send_audio_response: {str(e)}")
        # Send a stop event to end the call gracefully
        await websocket.send_text(json.dumps({
            "event": "stop",
            "streamSid": stream_sid
        }))

@router.websocket("/ws/twilio")
async def twilio_websocket(websocket: WebSocket):
    """Handle Twilio Media Streams WebSocket connection for audio input and output."""
    await websocket.accept()
    logger.info("Twilio WebSocket connection established")
    buffer = []
    last_sequence = 0
    vad = webrtcvad.Vad(0)  # Mode 0: Less aggressive to improve speech detection
    sample_rate = 8000
    frame_duration_ms = 20  # Twilio sends 20ms frames (160 samples at 8kHz)
    frame_size = int(sample_rate * frame_duration_ms / 1000)  # 160 samples
    min_chunks = 50  # Collect ~1s at 8kHz to process faster
    silence_frames = 0
    silence_frames_threshold = 100  # Require 100 frames (2 seconds) of silence to end speech
    chunk_size = 8192
    speech_detected = False  # Track if speech was detected in the buffer
    last_speech_time = time.time()  # Track the last time speech was detected or a prompt was sent
    silence_timeout = 30  # Send a prompt after 30 seconds of silence
    max_prompts = 3  # Maximum number of prompts before ending the call
    prompt_count = 0  # Track the number of prompts sent
    stream_sid = None  # Store streamSid from the 'start' event

    # Log VAD setup
    logger.info(f"VAD initialized: mode=0, sample_rate={sample_rate}, frame_size={frame_size} samples, min_chunks={min_chunks}, silence_frames_threshold={silence_frames_threshold}")

    try:
        while True:
            # Check for silence timeout
            current_time = time.time()
            elapsed_time = current_time - last_speech_time
            logger.debug(f"Silence check: elapsed_time={elapsed_time:.2f}s, speech_detected={speech_detected}, prompt_count={prompt_count}")
            if elapsed_time >= silence_timeout and not speech_detected and stream_sid:
                prompt_count += 1
                if prompt_count >= max_prompts:
                    logger.info(f"Max prompts ({max_prompts}) reached, ending call")
                    await websocket.send_text(json.dumps({
                        "event": "stop",
                        "streamSid": stream_sid
                    }))
                    break
                logger.info(f"No speech detected for {silence_timeout} seconds, sending prompt {prompt_count}/{max_prompts}")
                await websocket.send_text(json.dumps({
                    "event": "media",
                    "streamSid": stream_sid,
                    "media": {
                        "payload": base64.b64encode(b"").decode('utf-8')  # Send empty media to trigger TwiML prompt
                    }
                }))
                last_speech_time = time.time()  # Reset the timer after sending a prompt

            data = await websocket.receive_text()
            message = json.loads(data)
            event = message.get("event")

            if event == "connected":
                logger.info("Twilio Media Stream connected")
                continue
            elif event == "start":
                stream_sid = message.get("streamSid")
                logger.info(f"Twilio Media Stream started, streamSid={stream_sid}")
                continue
            elif event == "stop":
                logger.info("Twilio Media Stream stopped")
                break
            elif event == "media":
                # Twilio sends base64-encoded audio in MULAW/8000 format
                payload = message.get("media", {}).get("payload")
                if not payload:
                    logger.warning("Received empty media payload")
                    continue
                # Decode MULAW/8000 audio
                try:
                    mulaw_data = base64.b64decode(payload)
                    audio_segment = AudioSegment(
                        mulaw_data,
                        sample_width=1,  # MULAW is 8-bit
                        frame_rate=8000,
                        channels=1
                    )
                    pcm = np.array(audio_segment.get_array_of_samples(), dtype=np.int16)
                    logger.info(f"Received audio chunk: sequence={message.get('sequenceNumber')}, size={len(pcm)*2} bytes")
                except Exception as e:
                    logger.error(f"Error decoding audio: {str(e)}")
                    continue

                # Convert sequenceNumber to int, fallback to last_sequence + 1
                sequence_str = message.get("sequenceNumber")
                try:
                    sequence = int(sequence_str) if sequence_str is not None else last_sequence + 1
                except (ValueError, TypeError) as e:
                    logger.error(f"Invalid sequenceNumber: {sequence_str}, using fallback: {last_sequence + 1}")
                    sequence = last_sequence + 1

                # Convert PCM to bytes for WebRTC VAD (expects 16-bit PCM)
                pcm_bytes = pcm.tobytes()
                if len(pcm) != frame_size:
                    logger.warning(f"Frame size mismatch: expected {frame_size}, got {len(pcm)}")
                    continue

                # Apply VAD
                is_speech = vad.is_speech(pcm_bytes, sample_rate)
                buffer.append({'sequence': sequence, 'pcm': pcm})

                if is_speech:
                    speech_detected = True
                    silence_frames = 0
                    last_speech_time = time.time()  # Reset the timer on speech detection
                    prompt_count = 0  # Reset prompt count when speech is detected
                    logger.info(f"Speech detected: sequence={sequence}")
                else:
                    silence_frames += 1
                    logger.debug(f"Silence detected: sequence={sequence}, silence_frames={silence_frames}")

                # Process only if we have enough chunks and silence is detected
                if len(buffer) >= min_chunks and silence_frames >= silence_frames_threshold:
                    # Count speech frames in the buffer
                    speech_frames = sum(1 for chunk in buffer if vad.is_speech(chunk['pcm'].tobytes(), sample_rate))
                    logger.info(f"Processing Twilio buffer: {len(buffer)} chunks, {speech_frames} speech frames, after {silence_frames} silence frames")
                    concatenated_pcm = concatenate_chunks(buffer)

                    # Calculate and log audio duration and RMS
                    duration_s = len(concatenated_pcm) / sample_rate
                    rms = np.sqrt(np.mean(concatenated_pcm.astype(np.float32)**2))
                    logger.info(f"Concatenated PCM: {len(concatenated_pcm)} samples at {sample_rate} Hz, duration={duration_s:.2f}s, RMS={rms:.4f}")

                    if speech_detected and speech_frames > 0 and rms > 3:  # Lowered RMS threshold
                        logger.info("Speech detected and finished, playing predefined audio response")
                        await send_audio_response(websocket, stream_sid, chunk_size)

                    buffer = []
                    silence_frames = 0
                    speech_detected = False
                last_sequence = sequence
    except WebSocketDisconnect:
        logger.info("Twilio WebSocket disconnected")
    except Exception as e:
        logger.error(f"Twilio WebSocket error: {str(e)}")
        try:
            if websocket.client_state == 1 and stream_sid:  # WebSocketState.CONNECTED
                await websocket.send_text(json.dumps({
                    "event": "stop",
                    "streamSid": stream_sid
                }))
        except Exception as send_error:
            logger.error(f"Failed to send stop message: {str(send_error)}")
    finally:
        try:
            await websocket.close()
            logger.info("Twilio WebSocket closed")
        except Exception as e:
            logger.error(f"Error closing Twilio WebSocket: {str(e)}")