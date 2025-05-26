import os
import time
import shutil
import logging
import json
import wave
import numpy as np
from pathlib import Path
from vosk import Model, KaldiRecognizer
from TTS.api import TTS
import torch.serialization
import collections
import aiofiles
import base64

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

VOICE_MODELS = {
    "female": {"model_name": "tts_models/en/ljspeech/tacotron2-DDC", "speaker": None},
    "male": {"model_name": "tts_models/en/vctk/vits", "speaker": "p239"},
}

espeak_path = r"C:\Program Files\eSpeak NG"
if os.path.exists(espeak_path):
    os.environ["PATH"] = espeak_path + os.pathsep + os.environ.get("PATH", "")
    logger.info(f"Added {espeak_path} to PATH for espeak-ng")
else:
    logger.warning(f"espeak-ng path {espeak_path} not found. VITS models may fail to load.")

try:
    from TTS.utils.radam import RAdam
    torch.serialization.add_safe_globals([RAdam, collections.defaultdict, dict])
    logger.info("Added TTS.utils.radam.RAdam, collections.defaultdict, and dict to PyTorch safe globals")
except ImportError:
    logger.error("Failed to import TTS.utils.radam.RAdam for allowlisting")
    raise

TTS_MODEL_DIR = Path("C:/Users/ASUS/AppData/Local/tts")

def clean_tts_model_directory(model_name: str):
    model_dir = TTS_MODEL_DIR / model_name.replace('/', '--')
    if model_dir.exists():
        try:
            shutil.rmtree(model_dir)
            logger.info(f"Cleaned up TTS model directory: {model_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up TTS model directory {model_dir}: {str(e)}")

UNUSED_MODELS = [
    "tts_models/en/jenny/jenny",
    "tts_models/en/ljspeech/glow-tts",
    "tts_models/en/ek1/tacotron2",
    "tts_models/en/ljspeech/fast_pitch",
]
for model_name in UNUSED_MODELS:
    clean_tts_model_directory(model_name)

TTS_MODELS = {}
try:
    for voice_label, config in VOICE_MODELS.items():
        model_name = config["model_name"]
        if model_name not in TTS_MODELS:
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
                clean_tts_model_directory(model_name)
                if model_name != "tts_models/en/ljspeech/tacotron2-DDC":
                    logger.info(f"Falling back to default TTS model for {voice_label}")
                    TTS_MODELS[model_name] = TTS_MODELS.get("tts_models/en/ljspeech/tacotron2-DDC")
                else:
                    raise
except Exception as e:
    logger.error(f"Critical failure in loading TTS models: {str(e)}")
    raise

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
        if voice_model not in VOICE_MODELS:
            logger.warning(f"Invalid voice model: {voice_model}, falling back to 'female'")
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
        logger.info(f"Generated speech saved to {output_path} in {time.time() - start_time:.3f}s")
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