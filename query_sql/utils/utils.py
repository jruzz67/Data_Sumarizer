import os
import time
import logging
from pathlib import Path
from .config import AUDIO_DIR

logger = logging.getLogger(__name__)

def delete_file(file_path: str) -> None:
    """Delete a file from the specified path."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Deleted file: {file_path}")
        else:
            logger.warning(f"File not found for deletion: {file_path}")
    except Exception as e:
        logger.error(f"Error deleting file {file_path}: {str(e)}")
        raise

def cleanup_old_audio_files(max_age_seconds=3600):
    """Delete audio files older than max_age_seconds."""
    try:
        current_time = time.time()
        for audio_file in AUDIO_DIR.glob("*.wav"):
            file_age = current_time - os.path.getmtime(audio_file)
            if file_age > max_age_seconds:
                os.remove(audio_file)
                logger.info(f"Deleted old audio file: {audio_file}")
    except Exception as e:
        logger.error(f"Error cleaning up audio files: {str(e)}")