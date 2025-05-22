import os
from pathlib import Path

# Directories
UPLOAD_DIR = Path("uploads")
AUDIO_DIR = Path("audio")

# Create directories if they don't exist
UPLOAD_DIR.mkdir(exist_ok=True)
AUDIO_DIR.mkdir(exist_ok=True)

# Constants
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
VOSK_MODEL_PATH = "models/vosk-model-small-en-us-0.15/vosk-model-small-en-us-0.15"
VOICE_MODELS = {
    "female": {"model_name": "tts_models/en/ljspeech/tacotron2-DDC", "speaker": None},
    "male": {"model_name": "tts_models/en/vctk/vits", "speaker": "p239"},
}
TTS_MODEL_DIR = Path("C:/Users/ASUS/AppData/Local/tts")