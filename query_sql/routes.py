from fastapi import APIRouter, UploadFile, File, HTTPException, Body, WebSocket, WebSocketDisconnect, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Optional
from services import process_audio, text_to_speech
from file_handler import extract_text_from_file
from chunker import chunk_text, search_top_chunks
from embedder import embed_chunks, store_embeddings
from query_handler import handle_query
from utils import delete_file
from db import get_db
import os
import time
import json
import base64
import numpy as np
from pydub import AudioSegment
import asyncio
import webrtcvad
from scipy.signal import resample as scipy_resample
import audioop
import soundfile as sf
import logging
import uuid
import wave
from pathlib import Path

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_FILE_SIZE = 10 * 1024 * 1024
UPLOAD_DIR = Path("uploads")
AUDIO_DIR = Path("audio")

def resample_audio(audio_data: np.ndarray, target_samples: int) -> np.ndarray:
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
    ws_url = "wss://d65d-2409-40f4-101c-c09-8cf6-8a8f-d142-9955.ngrok-free.app/ws"
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

async def send_audio_response(websocket, stream_sid: Optional[str] = None, chunk_size: int = 8192, client_type: str = "web"):
    try:
        response_pcm = text_to_speech()
        response_pcm_data = response_pcm
        max_val = np.max(np.abs(response_pcm_data))
        if max_val > 0:
            response_pcm_data = (response_pcm_data / max_val * 32767).astype(np.int16)
        logger.info(f"Normalized response PCM: min={np.min(response_pcm_data)}, max={np.max(response_pcm_data)}")

        # Resample to 8000 Hz if necessary
        if len(response_pcm_data) > 0:
            target_samples = int(len(response_pcm_data) * 8000 / 16000)
            response_pcm_data = resample_audio(response_pcm_data, target_samples)
        logger.info(f"Resampled response PCM to 8000 Hz: {len(response_pcm_data)} samples")

        pcm_bytes = response_pcm_data.tobytes()
        mulaw_audio = audioop.lin2ulaw(pcm_bytes, 2)
        logger.info(f"Converted response to MULAW: {len(mulaw_audio)} bytes")

        # Buffer for response MULAW chunks
        response_buffer = []
        sequence = 0
        for i in range(0, len(mulaw_audio), chunk_size):
            chunk = mulaw_audio[i:i + chunk_size]
            response_buffer.append(chunk)
            mulaw_base64 = base64.b64encode(chunk).decode('utf-8')
            if client_type == "twilio" and stream_sid:
                await websocket.send_text(json.dumps({
                    "event": "media",
                    "streamSid": stream_sid,
                    "media": {
                        "payload": mulaw_base64
                    }
                }))
            else:
                await websocket.send_text(json.dumps({
                    "type": "audio_chunk",
                    "sequence": sequence,
                    "pcm": mulaw_base64
                }))
            logger.info(f"Sent response audio chunk: sequence={sequence}, size={len(chunk)} bytes, client_type={client_type}")
            sequence += 1
            await asyncio.sleep(0.1)
        return response_buffer, response_pcm_data
    except Exception as e:
        logger.error(f"Error in send_audio_response: {str(e)}")
        if client_type == "twilio" and stream_sid:
            await websocket.send_text(json.dumps({
                "event": "stop",
                "streamSid": stream_sid
            }))
        else:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": str(e)
            }))
        return [], np.array([])

async def save_buffers(input_buffer, mulaw_buffer, pcm_buffer, response_buffer, response_pcm_data):
    try:
        session_id = str(uuid.uuid4())
        # Save input MULAW (as received)
        if input_buffer:
            input_concat = b''.join([chunk['data'] for chunk in sorted(input_buffer, key=lambda x: x['sequence'])])
            input_path = UPLOAD_DIR / f"MU_{session_id[:8]}.wav"
            with wave.open(str(input_path), 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(1)  # MULAW is 8-bit
                wf.setframerate(8000)
                wf.writeframes(input_concat)
            logger.info(f"Saved input MULAW to {input_path}")
        # Save processed MULAW
        if mulaw_buffer:
            mulaw_concat = b''.join([chunk['data'] for chunk in sorted(mulaw_buffer, key=lambda x: x['sequence'])])
            mulaw_path = UPLOAD_DIR / f"MU_processed_{session_id[:8]}.wav"
            with wave.open(str(mulaw_path), 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(1)
                wf.setframerate(8000)
                wf.writeframes(mulaw_concat)
            logger.info(f"Saved processed MULAW to {mulaw_path}")
        # Save PCM
        if pcm_buffer:
            pcm_concat = np.concatenate([chunk['data'] for chunk in sorted(pcm_buffer, key=lambda x: x['sequence'])])
            pcm_path = UPLOAD_DIR / f"PCM_{session_id[:8]}.wav"
            with wave.open(str(pcm_path), 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(8000)
                wf.writeframes(pcm_concat.tobytes())
            logger.info(f"Saved PCM to {pcm_path}")
        # Save response MULAW
        if response_buffer:
            response_concat = b''.join(response_buffer)
            response_path = AUDIO_DIR / f"response_MU_{session_id[:8]}.wav"
            with wave.open(str(response_path), 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(1)
                wf.setframerate(8000)
                wf.writeframes(response_concat)
            logger.info(f"Saved response MULAW to {response_path}")
        # Save response PCM
        if len(response_pcm_data) > 0:
            response_pcm_path = AUDIO_DIR / f"response_PCM_{session_id[:8]}.wav"
            with wave.open(str(response_pcm_path), 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(8000)
                wf.writeframes(response_pcm_data.tobytes())
            logger.info(f"Saved response PCM to {response_pcm_path}")
    except Exception as e:
        logger.error(f"Error saving buffers: {str(e)}")

@router.websocket("/ws")
async def unified_websocket(websocket: WebSocket):
    await websocket.accept()
    logger.info("Unified WebSocket connection established")

    client_type = None
    stream_sid = None
    input_buffer = []  # Raw MULAW as received
    mulaw_buffer = []  # Processed MULAW
    pcm_buffer = []    # Converted PCM
    response_buffer = []  # Response MULAW chunks
    response_pcm_data = np.array([])  # Response PCM data
    last_sequence = 0
    vad = webrtcvad.Vad(0)
    sample_rate = 8000
    frame_duration_ms = 20
    frame_size = int(sample_rate * frame_duration_ms / 1000)
    min_chunks = 50
    silence_frames = 0
    silence_frames_threshold = 100
    chunk_size = 8192
    speech_detected = False
    last_speech_time = time.time()
    silence_timeout = 30
    max_prompts = 3
    prompt_count = 0

    logger.info(f"VAD initialized: mode=0, sample_rate={sample_rate}, frame_size={frame_size} samples, min_chunks={min_chunks}, silence_frames_threshold={silence_frames_threshold}")

    try:
        data = await websocket.receive_text()
        message = json.loads(data)

        if message.get("event") == "connected":
            client_type = "twilio"
            logger.info("Client identified as Twilio")
        elif message.get("client_type") == "web":
            client_type = "web"
            logger.info("Client identified as Web")
            sequence = message.get("sequence")
            silence_flag = message.get("silence")
            mulaw_base64 = message.get("pcm")
        else:
            logger.error("Unknown client type")
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "Unknown client type"
            }))
            await websocket.close()
            return

        while True:
            if client_type == "twilio":
                current_time = time.time()
                elapsed_time = current_time - last_speech_time
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
                            "payload": base64.b64encode(b"").decode('utf-8')
                        }
                    }))
                    last_speech_time = time.time()

            if client_type != "web" or input_buffer:
                data = await websocket.receive_text()
                message = json.loads(data)

            if client_type == "twilio":
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
                    payload = message.get("media", {}).get("payload")
                    if not payload:
                        logger.warning("Received empty media payload")
                        continue
                    try:
                        mulaw_data = base64.b64decode(payload)
                        audio_segment = AudioSegment(
                            mulaw_data,
                            sample_width=1,
                            frame_rate=8000,
                            channels=1
                        )
                        pcm = np.array(audio_segment.get_array_of_samples(), dtype=np.int16)
                        logger.info(f"Received audio chunk: sequence={message.get('sequenceNumber')}, size={len(pcm)*2} bytes")
                    except Exception as e:
                        logger.error(f"Error decoding audio: {str(e)}")
                        continue

                    sequence_str = message.get("sequenceNumber")
                    try:
                        sequence = int(sequence_str) if sequence_str is not None else last_sequence + 1
                    except (ValueError, TypeError) as e:
                        logger.error(f"Invalid sequenceNumber: {sequence_str}, using fallback: {last_sequence + 1}")
                        sequence = last_sequence + 1
                else:
                    logger.warning(f"Unknown Twilio event: {event}")
                    continue
            else:
                sequence = message.get("sequence")
                silence_flag = message.get("silence")
                mulaw_base64 = message.get("pcm")
                if mulaw_base64 is None:
                    logger.warning("Received empty MULAW payload")
                    continue
                try:
                    mulaw_data = base64.b64decode(mulaw_base64)
                    audio_segment = AudioSegment(
                        mulaw_data,
                        sample_width=1,
                        frame_rate=8000,
                        channels=1
                    )
                    pcm = np.array(audio_segment.get_array_of_samples(), dtype=np.int16)
                    logger.info(f"Received MULAW chunk: sequence={sequence}, size={len(pcm)*2} bytes")
                except Exception as e:
                    logger.error(f"Error decoding MULAW audio: {str(e)}")
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": f"Error decoding audio: {str(e)}"
                    }))
                    continue

                if sequence is None:
                    sequence = last_sequence + 1
                else:
                    try:
                        sequence = int(sequence)
                    except (ValueError, TypeError) as e:
                        logger.error(f"Invalid sequence: {sequence}, using fallback: {last_sequence + 1}")
                        sequence = last_sequence + 1

            # Store in buffers
            input_buffer.append({'sequence': sequence, 'data': mulaw_data})
            mulaw_buffer.append({'sequence': sequence, 'data': mulaw_data})
            pcm_buffer.append({'sequence': sequence, 'data': pcm})

            pcm_bytes = pcm.tobytes()
            if len(pcm) != frame_size:
                logger.warning(f"Frame size mismatch: expected {frame_size}, got {len(pcm)}")
                continue

            is_speech = vad.is_speech(pcm_bytes, sample_rate)
            if client_type == "web":
                is_speech = is_speech and silence_flag == 1

            if is_speech:
                speech_detected = True
                silence_frames = 0
                last_speech_time = time.time()
                prompt_count = 0
                logger.info(f"Speech detected: sequence={sequence}, client_type={client_type}")
            else:
                silence_frames += 1
                logger.debug(f"Silence detected: sequence={sequence}, silence_frames={silence_frames}, client_type={client_type}")

            if len(pcm_buffer) >= min_chunks and silence_frames >= silence_frames_threshold:
                speech_frames = sum(1 for chunk in pcm_buffer if vad.is_speech(chunk['data'].tobytes(), sample_rate))
                logger.info(f"Processing buffer: {len(pcm_buffer)} chunks, {speech_frames} speech frames, after {silence_frames} silence frames, client_type={client_type}")
                
                # Concatenate PCM for transcription
                concatenated_pcm = np.concatenate([chunk['data'] for chunk in sorted(pcm_buffer, key=lambda x: x['sequence'])])

                duration_s = len(concatenated_pcm) / sample_rate
                rms = np.sqrt(np.mean(concatenated_pcm.astype(np.float32)**2))
                logger.info(f"Concatenated PCM: {len(concatenated_pcm)} samples at {sample_rate} Hz, duration={duration_s:.2f}s, RMS={rms:.4f}")

                if speech_detected and speech_frames > 0 and rms > 3:
                    # Resample to 16000 Hz for VOSK
                    target_samples = int(len(concatenated_pcm) * (16000 / 8000))
                    resampled_pcm = resample_audio(concatenated_pcm, target_samples)  # Fixed: Use concatenated_pcm
                    logger.info(f"Resampled PCM to 16000 Hz for VOSK: {len(resampled_pcm)} samples")

                    # Transcribe with VOSK
                    transcription = await process_audio(resampled_pcm)
                    logger.info(f"Transcription: {transcription}")
                    if transcription:
                        await websocket.send_text(json.dumps({
                            "type": "transcription",
                            "text": transcription
                        }))
                    else:
                        await websocket.send_text(json.dumps({
                            "type": "transcription",
                            "text": "No speech detected."
                        }))

                    # Send response
                    response_chunks, response_pcm = await send_audio_response(websocket, chunk_size=chunk_size, client_type=client_type)
                    response_buffer.extend(response_chunks)
                    response_pcm_data = response_pcm

                input_buffer = []
                mulaw_buffer = []
                pcm_buffer = []
                silence_frames = 0
                speech_detected = False
            last_sequence = sequence

    except WebSocketDisconnect:
        logger.info(f"{client_type.capitalize()} WebSocket disconnected")
    except Exception as e:
        logger.error(f"{client_type.capitalize()} WebSocket error: {str(e)}")
        try:
            if client_type == "twilio" and websocket.client_state == 1 and stream_sid:
                await websocket.send_text(json.dumps({
                    "event": "stop",
                    "streamSid": streamSid
                }))
            else:
                await websocket.send_text(json.dumps({
                "type": "error",
                "message": str(e)
            }))
        except Exception as send_error:
            logger.error(f"Failed to send error/stop message: {str(send_error)}")
    finally:
        await save_buffers(input_buffer, mulaw_buffer, pcm_buffer, response_buffer, response_pcm_data)
        try:
            await websocket.close()
            logger.info(f"{client_type.capitalize()} WebSocket closed")
        except Exception as e:
            logger.error(f"Error closing {client_type} WebSocket: {str(e)}")