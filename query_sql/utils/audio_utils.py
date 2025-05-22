# import logging
# from pydub import AudioSegment
# from utils.audio import speech_processor
# from pathlib import Path

# logger = logging.getLogger(__name__)

# async def convert_audio_to_wav(audio_path: str, output_path: str) -> None:
#     """
#     Convert an audio file to WAV format using pydub.
    
#     Args:
#         audio_path (str): Path to the input audio file (e.g., WebM).
#         output_path (str): Path where the converted WAV file will be saved.
#     """
#     try:
#         audio = AudioSegment.from_file(audio_path, format="webm")
#         audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)  # Ensure 16kHz, mono, 16-bit PCM
#         audio.export(output_path, format="wav")
#         logger.info(f"Converted audio to WAV: {output_path}")
#     except Exception as e:
#         logger.error(f"Error converting audio to WAV: {str(e)}")
#         raise

# async def speech_to_text_stream(audio_path: str) -> str:
#     """
#     Transcribe a WAV audio file to text using SpeechProcessor.
    
#     Args:
#         audio_path (str): Path to the WAV audio file.
    
#     Returns:
#         str: Transcribed text.
#     """
#     try:
#         # Read the WAV file and convert to raw PCM bytes
#         audio = AudioSegment.from_file(audio_path, format="wav")
#         pcm_data = audio.raw_data  # Raw 16kHz mono 16-bit PCM bytes
        
#         # Transcribe using SpeechProcessor
#         transcription = await speech_processor.transcribe_audio(pcm_data)
#         logger.info(f"Transcribed audio: {transcription}")
#         return transcription
#     except Exception as e:
#         logger.error(f"Error in speech-to-text: {str(e)}")
#         raise

# async def text_to_speech_stream(text: str, voice_model: str = "female") -> str:
#     """
#     Convert text to speech and save as a WAV file.
    
#     Args:
#         text (str): Text to convert to speech.
#         voice_model (str): Voice model to use ("male" or "female").
    
#     Returns:
#         str: Path to the generated WAV file.
#     """
#     try:
#         output_path = await speech_processor.text_to_speech(text, voice_model)
#         logger.info(f"Generated speech file: {output_path}")
#         return output_path
#     except Exception as e:
#         logger.error(f"Error in text-to-speech: {str(e)}")
#         raise