from utils.audio import speech_processor
import asyncio
import json
import logging
import os
import time
import uuid
from typing import Dict, List

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.speech_processor = speech_processor
        self.is_speaking: Dict[str, bool] = {}
        self.voice_model: Dict[str, str] = {}
        self.locks: Dict[str, asyncio.Lock] = {}
        self.current_session_id: Dict[str, int] = {}
        self.audio_chunks: Dict[str, List[bytes]] = {}
        self.last_activity: Dict[str, float] = {}  # Track last message time (for logging)
        self.last_speech_activity: Dict[str, float] = {}  # Track last speech activity for timeout
        self.ip_to_client: Dict[str, str] = {}  # Track client IPs

    async def connect(self, websocket: WebSocket, client_id: str):
        # Check for existing connection from the same IP
        client_ip = websocket.client.host if websocket.client else "unknown"
        existing_client_id = self.ip_to_client.get(client_ip)
        if existing_client_id and existing_client_id in self.active_connections:
            logger.warning(
                f"Client IP {client_ip} already has an active connection (client {existing_client_id}). Closing the old connection."
            )
            await self.disconnect(existing_client_id)
            await asyncio.sleep(0.1)  # Small delay to ensure disconnection

        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.ip_to_client[client_ip] = client_id
        self.audio_chunks[client_id] = []
        self.locks[client_id] = asyncio.Lock()
        self.voice_model[client_id] = "female"  # Default voice model
        self.is_speaking[client_id] = False
        self.last_activity[client_id] = time.time()
        self.last_speech_activity[client_id] = time.time()  # Initialize for inactivity timeout
        self.current_session_id[client_id] = 0
        logger.info(f"WebSocket connection established for client {client_id} from IP {client_ip}")

    async def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.is_speaking:
            del self.is_speaking[client_id]
        if client_id in self.voice_model:
            del self.voice_model[client_id]
        if client_id in self.locks:
            del self.locks[client_id]
        if client_id in self.current_session_id:
            del self.current_session_id[client_id]
        if client_id in self.audio_chunks:
            del self.audio_chunks[client_id]
        if client_id in self.last_activity:
            del self.last_activity[client_id]
        if client_id in self.last_speech_activity:
            del self.last_speech_activity[client_id]
        if client_id in self.ip_to_client:
            del self.ip_to_client[client_id]
        logger.info(f"WebSocket connection closed for client {client_id}")

    async def send_message(self, message: Dict[str, str], client_id: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_json(message)

    async def send_audio(self, audio_data: bytes, client_id: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_bytes(audio_data)

# Initialize the ConnectionManager
manager = ConnectionManager()

async def process_final_user_utterance(client_id: str, voice_model: str, session_id: int, websocket: WebSocket, chunks: List[bytes]):
    """
    Process the final user utterance by combining chunks, transcribing, and generating a response.
    """
    try:
        valid_chunks = [chunk for chunk in chunks if chunk]
        if not valid_chunks:
            logger.error(f"No valid PCM chunks for client {client_id}")
            await websocket.send_json({"type": "error", "message": "No valid audio chunks to process"})
            return

        # Since the frontend sends raw PCM data (s16le, 16kHz, mono), we can use it directly
        combined_pcm = b''.join(valid_chunks)
        logger.debug(f"Combined PCM data size for client {client_id}: {len(combined_pcm)} bytes")

        # Transcribe the audio
        transcription = None
        for attempt in range(2):
            try:
                transcription = await manager.speech_processor.transcribe_audio(
                    combined_pcm, session_id, manager.current_session_id.get(client_id, -1)
                )
                break
            except Exception as e:
                logger.error(f"Transcription failed for client {client_id}, attempt {attempt + 1}: {str(e)}")
                if attempt == 1:
                    await websocket.send_json({"type": "error", "message": "Failed to transcribe audio, please try again"})
                    return
        logger.info(f"Final transcription for client {client_id}: {transcription}")

        if transcription:
            await websocket.send_json({"type": "transcription", "text": transcription})
            response_text = await manager.speech_processor.process_query(transcription)
            await websocket.send_json({"type": "response", "text": response_text})

            manager.is_speaking[client_id] = True
            audio_response_path = await manager.speech_processor.text_to_speech(response_text, voice_model)
            with open(audio_response_path, "rb") as f:
                audio_data_to_send = f.read()
            await manager.send_audio(audio_data_to_send, client_id)
            os.remove(audio_response_path)
            manager.is_speaking[client_id] = False

    except Exception as e:
        logger.error(f"Error processing final utterance for client {client_id}: {str(e)}")
        if client_id in manager.active_connections:
            await websocket.send_json({"type": "error", "message": f"Error processing audio: {str(e)}"})

@router.websocket("/ws/voice")
async def websocket_voice(websocket: WebSocket):
    client_id = str(uuid.uuid4())
    client_ip = websocket.client.host if websocket.client else "unknown"
    logger.info(f"WebSocket connection established for client {client_id} from IP {client_ip}")

    # Check for existing connection from the same IP and connect
    await manager.connect(websocket, client_id)

    try:
        # Start a timeout task to monitor inactivity based on speech activity
        async def monitor_inactivity():
            while client_id in manager.active_connections:
                await asyncio.sleep(5)  # Check every 5 seconds
                if client_id not in manager.last_speech_activity:
                    break
                elapsed = time.time() - manager.last_speech_activity[client_id]
                if elapsed > 30:  # 30 seconds timeout
                    logger.warning(f"Client {client_id} inactive (no speech) for {elapsed:.1f} seconds. Closing connection.")
                    await websocket.send_json({"type": "error", "message": "Inactivity timeout"})
                    await websocket.close(code=1000, reason="Inactivity timeout")
                    break

        inactivity_task = asyncio.create_task(monitor_inactivity())

        while True:
            message = await websocket.receive()
            manager.last_activity[client_id] = time.time()

            if "text" in message:
                data = json.loads(message["text"])
                message_type = data.get("type")

                if message_type == "start":
                    logger.info(f"Received start signal for client {client_id}")
                    manager.voice_model[client_id] = data.get("voice_model", "female")
                    manager.current_session_id[client_id] += 1
                    manager.last_speech_activity[client_id] = time.time()  # Reset on start
                    await websocket.send_json({"type": "status", "message": "Ready to receive audio"})

                elif message_type == "user_interrupted":
                    logger.info(f"User interrupted bot for client {client_id}")
                    manager.is_speaking[client_id] = False
                    await websocket.send_json({"type": "status", "message": "Bot interrupted"})

                elif message_type == "speech_end":
                    logger.info(f"Received speech_end signal for client {client_id}")
                    session_id = manager.current_session_id.get(client_id, 0)
                    if manager.audio_chunks[client_id]:
                        async with manager.locks[client_id]:
                            await process_final_user_utterance(client_id, manager.voice_model[client_id], session_id, websocket, manager.audio_chunks[client_id])
                        manager.audio_chunks[client_id] = []
                        await manager.speech_processor.reset_recognizer()
                    else:
                        logger.warning(f"No audio chunks received for client {client_id} before speech_end")
                        await websocket.send_json({"type": "error", "message": "No audio data received"})

                elif message_type == "end":
                    logger.info(f"Received end signal for client {client_id}")
                    session_id = manager.current_session_id.get(client_id, 0)
                    if manager.audio_chunks[client_id]:
                        async with manager.locks[client_id]:
                            await process_final_user_utterance(client_id, manager.voice_model[client_id], session_id, websocket, manager.audio_chunks[client_id])
                        manager.audio_chunks[client_id] = []
                        await manager.speech_processor.reset_recognizer()
                    break

            elif "bytes" in message:
                audio_chunk_pcm = message["bytes"]
                logger.debug(f"Received PCM chunk for client {client_id}, size: {len(audio_chunk_pcm)} bytes")
                manager.audio_chunks[client_id].append(audio_chunk_pcm)
                manager.last_speech_activity[client_id] = time.time()  # Update on speech activity

    except WebSocketDisconnect:
        logger.info(f"WebSocket connection closed for client {client_id}")
    except Exception as e:
        logger.error(f"Unexpected error for client {client_id}: {str(e)}")
    finally:
        inactivity_task.cancel()
        await manager.disconnect(client_id)
        logger.info(f"Cleaned up resources for client {client_id}")

def register_websocket(app):
    app.include_router(router)
    logger.info("WebSocket routes registered successfully")

# Import speech_processor after ConnectionManager definition to avoid circular imports
