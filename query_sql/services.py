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
import librosa

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

PREDEFINED_WAV_PATH = r"D:\Projects\OCR_with_PostgresSQL\query_sql\audio\Thank you for contac.wav"

def decode_base64_pcm(base64_str: str) -> np.ndarray:
    try:
        pcm_bytes = base64.b64decode(base64_str)
        pcm = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        amplification_factor = 2.0
        pcm = pcm * amplification_factor
        pcm = np.clip(pcm, -32768, 32767).astype(np.int16)
        logger.info(f"Amplified PCM by {amplification_factor}x")
        return pcm
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

def verify_trailing_silence(pcm: np.ndarray, sample_rate: int = 16000, silence_duration: float = 0.5, threshold: float = 14.0) -> bool:
    try:
        samples_to_check = int(sample_rate * silence_duration)
        if len(pcm) < samples_to_check:
            logger.warning(f"PCM too short for silence verification: {len(pcm)} samples")
            return False
        trailing_samples = pcm[-samples_to_check:].astype(np.float32)
        rms = np.sqrt(np.mean(trailing_samples**2))
        if np.isnan(rms):
            logger.warning("RMS calculation resulted in NaN")
            return False
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

async def process_audio(pcm: np.ndarray, sample_rate: int = 16000) -> str:
    try:
        start_time = time.time()
        if len(pcm) < 32000:  # ~2s at 16kHz
            logger.warning(f"PCM too short for transcription: {len(pcm)} samples")
            return ""
        wav_path = UPLOAD_DIR / f"audio_{int(time.time())}.wav"
        await write_pcm_to_wav(pcm.tobytes(), str(wav_path), sample_rate)
        recognizer = KaldiRecognizer(vosk_model, sample_rate, '{"beam": 20}')
        recognizer.SetWords(True)
        transcribed_text = ""
        async with aiofiles.open(wav_path, 'rb') as wf:
            while True:
                data = await wf.read(8000)
                if not data:
                    break
                if recognizer.AcceptWaveform(data):
                    result = recognizer.Result()
                    logger.info(f"Partial Vosk result: {result}")
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
        debug_wav = UPLOAD_DIR / f"debug_{wav_path.name}"
        shutil.copy(wav_path, debug_wav)
        logger.info(f"Saved debug WAV: {debug_wav}")
        os.remove(wav_path)
        logger.info(f"Deleted WAV: {wav_path}")
        return transcribed_text
    except Exception as e:
        logger.error(f"Error processing PCM: {str(e)}")
        if 'wav_path' in locals() and os.path.exists(wav_path):
            os.remove(wav_path)
            logger.info(f"Cleaned up WAV: {wav_path}")
        return ""

def text_to_speech(text: str = None, voice_model: str = "female") -> np.ndarray:
    try:
        start_time = time.time()
        logger.info(f"Loading predefined WAV file for response")
        if not os.path.exists(PREDEFINED_WAV_PATH):
            logger.error(f"Predefined WAV file not found at {PREDEFINED_WAV_PATH}")
            raise FileNotFoundError(f"Predefined WAV file not found at {PREDEFINED_WAV_PATH}")
        audio, sr = librosa.load(PREDEFINED_WAV_PATH, sr=16000, mono=True)
        logger.info(f"Loaded WAV: sample_rate={sr}, channels=1, min={audio.min()}, max={audio.max()}")
        if np.max(np.abs(audio)) > 1.0:
            audio = audio / np.max(np.abs(audio))
            logger.info("Normalized audio to prevent clipping")
        pcm_data = (audio * 32767).astype(np.int16)
        logger.info(f"Generated PCM data in {time.time() - start_time:.3f}s")
        return pcm_data
    except Exception as e:
        logger.error(f"Error in loading predefined WAV: {str(e)}")
        raise