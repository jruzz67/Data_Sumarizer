import os
import time
import shutil
import logging
import json
import wave
import numpy as np
from pathlib import Path
from vosk import Model, KaldiRecognizer
import aiofiles
import base64
import subprocess

logger = logging.getLogger(__name__)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

AUDIO_DIR = Path("audio")
AUDIO_DIR.mkdir(exist_ok=True)

VOSK_MODEL_PATH = r"D:\Projects\OCR_with_PostgresSQL\query_sql\models\vosk-model-en-in-0.5"
if not os.path.exists(VOSK_MODEL_PATH):
    logger.error(f"Vosk model not found at {VOSK_MODEL_PATH}")
    raise FileNotFoundError(f"Vosk model not found at {VOSK_MODEL_PATH}")
vosk_model = Model(VOSK_MODEL_PATH)
logger.info("Vosk model loaded successfully")

PIPER_BINARY = r"D:\Projects\OCR_with_PostgresSQL\query_sql\models\piper\piper.exe"
PIPER_MODELS = {
    "female": r"D:\Projects\OCR_with_PostgresSQL\query_sql\models\piper_models\en_US-amy-medium.onnx",
    "male": r"D:\Projects\OCR_with_PostgresSQL\query_sql\models\piper_models\en_US-joe-medium.onnx"
}

def decode_base64_pcm(base64_str: str) -> np.ndarray:
    try:
        pcm_bytes = base64.b64decode(base64_str)
        return np.frombuffer(pcm_bytes, dtype=np.int16)
    except Exception as e:
        logger.error(f"Error decoding base64 PCM: {str(e)}")
        raise

def concatenate_chunks(chunks: list) -> np.ndarray:
    try:
        sorted_chunks = sorted(chunks, key=lambda x: x['sequence'])
        return np.concatenate([chunk['pcm'] for chunk in sorted_chunks])
    except Exception as e:
        logger.error(f"Error concatenating chunks: {str(e)}")
        raise

def verify_trailing_silence(pcm: np.ndarray, sample_rate: int = 16000, silence_duration: float = 0.5, threshold: float = 20.0) -> bool:
    try:
        samples_to_check = int(sample_rate * silence_duration)
        if len(pcm) < samples_to_check:
            logger.warning(f"PCM too short for silence verification: {len(pcm)} samples")
            return False
        trailing_samples = pcm[-samples_to_check:]
        rms = np.sqrt(np.mean(trailing_samples**2))
        logger.info(f"Trailing RMS: {rms:.4f}, threshold: {threshold}")
        return rms < threshold
    except Exception as e:
        logger.error(f"Error verifying trailing silence: {str(e)}")
        return False

async def write_pcm_to_wav(pcm_data: bytes, output_path: str, sample_rate: int = 16000) -> None:
    try:
        start_time = time.time()
        if not pcm_data:
            logger.error("PCM data is empty")
            raise ValueError("PCM data is empty")
        int16_array = np.frombuffer(pcm_data, dtype=np.int16)
        async with aiofiles.open(output_path, 'wb') as f:
            with wave.open(f.name, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(int16_array.tobytes())
        logger.info(f"Successfully wrote PCM to WAV: {output_path} in {time.time() - start_time:.3f}s")
    except Exception as e:
        logger.error(f"Error writing PCM to WAV: {str(e)}")
        raise

async def process_audio(pcm: np.ndarray) -> str:
    try:
        start_time = time.time()
        if len(pcm) < 8000:  # ~0.5s at 16kHz
            logger.warning(f"PCM too short for transcription: {len(pcm)} samples")
            return ""
        wav_path = UPLOAD_DIR / f"audio_{int(time.time())}.wav"
        await write_pcm_to_wav(pcm.tobytes(), str(wav_path))
        recognizer = KaldiRecognizer(vosk_model, 16000)
        recognizer.SetWords(True)
        transcribed_text = ""
        async with aiofiles.open(wav_path, 'rb') as wf:
            while True:
                data = await wf.read(4000)
                if not data:
                    break
                if recognizer.AcceptWaveform(data):
                    result = recognizer.Result()
                    logger.info(f"Raw Vosk result: {result}")
                    try:
                        text = json.loads(result).get("text", "")
                        if text:
                            logger.info(f"Transcribed: {text}")
                            transcribed_text += text + " "
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing Vosk result: {str(e)}")
            final_result = recognizer.FinalResult()
            logger.info(f"Final Vosk result: {final_result}")
            try:
                text = json.loads(final_result).get("text", "")
                if text:
                    logger.info(f"Final transcribed: {text}")
                    transcribed_text += text
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing final Vosk result: {str(e)}")
        transcribed_text = transcribed_text.strip()
        logger.info(f"Transcription completed: {transcribed_text} in {time.time() - start_time:.3f}s")
        os.remove(wav_path)
        logger.info(f"Deleted WAV: {wav_path}")
        return transcribed_text
    except Exception as e:
        logger.error(f"Error processing PCM: {str(e)}")
        if 'wav_path' in locals() and os.path.exists(wav_path):
            os.remove(wav_path)
            logger.info(f"Cleaned up WAV: {wav_path}")
        return ""

def text_to_speech(text: str, output_path: str, voice_model: str = "female") -> None:
    try:
        start_time = time.time()
        logger.info(f"Using voice model: {voice_model}")
        model_path = PIPER_MODELS.get(voice_model, PIPER_MODELS["female"])
        if not os.path.exists(model_path):
            logger.error(f"Piper model not found at {model_path}")
            raise FileNotFoundError(f"Piper model not found at {model_path}")
        if not os.path.exists(PIPER_BINARY):
            logger.error(f"Piper binary not found at {PIPER_BINARY}")
            raise FileNotFoundError(f"Piper binary not found at {PIPER_BINARY}")
        logger.info(f"Generating TTS with model: {model_path}")
        subprocess.run(
            [PIPER_BINARY, "--model", model_path, "--output_file", output_path],
            input=text.encode(),
            check=True,
            capture_output=True
        )
        logger.info(f"Generated speech saved to {output_path} in {time.time() - start_time:.3f}s")
    except subprocess.CalledProcessError as e:
        logger.error(f"Piper TTS failed: {e.stderr.decode()}")
        raise
    except Exception as e:
        logger.error(f"Error in text-to-speech: {str(e)}")
        raise

def cleanup_old_audio_files(max_age_seconds=3600):
    try:
        start_time = time.time()
        current_time = time.time()
        for audio_file in AUDIO_DIR.glob("*.wav"):
            file_age = current_time - os.path.getmtime(audio_file)
            if file_age > max_age_seconds:
                os.remove(audio_file)
                logger.info(f"Deleted old audio file: {audio_file}")
        logger.info(f"Cleanup completed in {time.time() - start_time:.3f}s")
    except Exception as e:
        logger.error(f"Error cleaning up audio files: {str(e)}")