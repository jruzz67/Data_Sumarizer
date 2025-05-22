import logging
import os
from pathlib import Path
import json
from vosk import Model, KaldiRecognizer
from TTS.api import TTS
import psycopg2.extras  # Import for DictCursor
from core.embedder import embed_chunks
from core.query_handler import handle_query
from core.db import get_db

logger = logging.getLogger(__name__)

class SpeechProcessor:
    def __init__(self):
        self.model_path = "models/vosk-model-small-en-us-0.15/vosk-model-small-en-us-0.15"
        self.sample_rate = 16000
        self.vosk_model = None
        self.recognizer = None
        self.tts_model_male = None
        self.tts_model_female = None
        self._setup()

    def _setup(self):
        # Set up Vosk model for speech recognition
        if not os.path.exists(self.model_path):
            logger.error(f"Vosk model not found at {self.model_path}")
            raise FileNotFoundError(f"Vosk model not found at {self.model_path}")

        self.vosk_model = Model(self.model_path)
        # Configure Vosk with proper settings for streaming
        self.recognizer = KaldiRecognizer(self.vosk_model, self.sample_rate)
        self.recognizer.SetWords(True)  # Enable word-level results
        self.recognizer.SetPartialWords(True)  # Enable partial results
        self.recognizer.SetMaxAlternatives(0)  # Disable alternatives
        logger.info("Vosk model loaded successfully with streaming configuration")

        # Add eSpeak NG to PATH for TTS. Adjust path for your OS if not Windows.
        espeak_path = r"C:\Program Files\eSpeak NG"
        if os.path.exists(espeak_path):
            os.environ["PATH"] += os.pathsep + espeak_path
            logger.info(f"Added {espeak_path} to PATH for espeak-ng")
        else:
            logger.warning(f"eSpeak NG not found at {espeak_path}. TTS might not work if espeak-ng is a backend requirement for your TTS model.")

        # Load TTS models
        try:
            # Check for female model first, as it's the default
            tts_female_model_name = "tts_models/en/vctk/vits"
            tts_female_model_path = Path.home() / "AppData" / "Local" / "tts" / tts_female_model_name.replace('/', '--')
            if tts_female_model_path.exists():
                logger.info(f"Female TTS model directory exists: {tts_female_model_path}")
            else:
                logger.warning(f"Female TTS model directory not found: {tts_female_model_path}")

            logger.info(f"Attempting to load female TTS model: {tts_female_model_name}")
            self.tts_model_female = TTS(model_name=tts_female_model_name, progress_bar=False)
            logger.info(f"Successfully loaded female TTS model: {tts_female_model_name}")

            # Check for male model
            tts_male_model_name = "tts_models/en/ljspeech/tacotron2-DDC"
            tts_male_model_path = Path.home() / "AppData" / "Local" / "tts" / tts_male_model_name.replace('/', '--')
            if tts_male_model_path.exists():
                logger.info(f"Male TTS model directory exists: {tts_male_model_path}")
            else:
                logger.warning(f"Male TTS model directory not found: {tts_male_model_path}")

            logger.info(f"Attempting to load male TTS model: {tts_male_model_name}")
            self.tts_model_male = TTS(model_name=tts_male_model_name, progress_bar=False)
            logger.info(f"Successfully loaded male TTS model: {tts_male_model_name}")

        except Exception as e:
            logger.error(f"Failed to load TTS models: {str(e)}")
            raise

    async def reset_recognizer(self):
        """Reset the Vosk recognizer for a new streaming session."""
        if self.recognizer:
            self.recognizer.Reset()
            logger.info("Recognizer reset for streaming")
        else:
            # Re-initialize if for some reason it's not set (shouldn't happen after _setup)
            self.recognizer = KaldiRecognizer(self.vosk_model, self.sample_rate)
            self.recognizer.SetWords(True)
            self.recognizer.SetPartialWords(True)
            self.recognizer.SetMaxAlternatives(0)
            logger.info("Recognizer re-initialized for streaming")

    async def stream_transcribe(self, audio_bytes: bytes, session_id: int, current_session_id: int) -> str | None:
        """
        Stream raw PCM audio data to Vosk and return a partial or final transcription.
        Expects raw 16kHz mono 16-bit PCM audio bytes.
        """
        # Check if this session is still valid
        if session_id != current_session_id:
            logger.info(f"Discarding stream transcription for outdated session {session_id} (current: {current_session_id})")
            return None

        try:
            logger.debug(f"Streaming {len(audio_bytes)} bytes for session {session_id}")
            if self.recognizer.AcceptWaveform(audio_bytes):
                # Full utterance recognized
                result = self.recognizer.Result()
                transcription = json.loads(result).get("text", "")
                if transcription:
                    logger.info(f"Complete query detected (stream, session {session_id}): {transcription}")
                    return transcription
            else:
                # Partial result
                partial_result = self.recognizer.PartialResult()
                partial_text = json.loads(partial_result).get("partial", "")
                if partial_text:
                    logger.debug(f"Partial transcription (session {session_id}): {partial_text}")
                    return partial_text
            return None  # No meaningful transcription yet
        except Exception as e:
            logger.error(f"Error in streaming transcription (Vosk AcceptWaveform, session {session_id}): {str(e)}")
            return None  # Return None on error for streaming transcription

    async def transcribe_audio(self, audio_bytes: bytes, session_id: int, current_session_id: int) -> str:
        """
        Transcribe a complete audio segment.
        Expects raw 16kHz mono 16-bit PCM audio bytes.
        """
        # Check if this session is still valid
        if session_id != current_session_id:
            logger.info(f"Discarding final transcription for outdated session {session_id} (current: {current_session_id})")
            return ""

        try:
            logger.debug(f"Transcribing {len(audio_bytes)} bytes for session {session_id}")
            # Ensure the recognizer is ready to receive the full audio
            self.recognizer.Reset()
            self.recognizer.AcceptWaveform(audio_bytes)
            
            result = self.recognizer.FinalResult()
            transcription = json.loads(result).get("text", "")
            logger.info(f"Final transcription (session {session_id}): {transcription}")
            return transcription
        except Exception as e:
            logger.error(f"Error transcribing final audio (Vosk FinalResult, session {session_id}): {str(e)}")
            raise

    async def process_query(self, transcription: str) -> str:
        """Process the transcribed query using the database with retry logic."""
        try:
            if not transcription:
                logger.info("Empty transcription received, prompting user to speak again")
                return "I didn't hear anything. Please speak again."

            # Embed the query
            query_embedding = embed_chunks([transcription])[0]
            logger.debug(f"Query embedded successfully: {transcription}")

            # Search the database for relevant chunks with retry logic
            top_chunks = None
            for attempt in range(2):
                try:
                    with get_db() as conn:
                        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
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
                        break
                except psycopg2.Error as e:
                    logger.error(f"Database error during vector search, attempt {attempt + 1}: {str(e)})")
                    if attempt == 1:
                        logger.error("Failed to query database after retries")
                        return "Sorry, I encountered a database error. Please try again later."

            if not top_chunks:
                logger.info("No relevant chunks found for query")
                return "No relevant chunks found in the database."

            # Generate a response using the query handler
            response = handle_query(transcription, top_chunks)
            logger.info(f"Query processed successfully: {transcription}, Response: {response}")
            return response
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise

    async def text_to_speech(self, text: str, voice_model: str) -> str:
        """
        Converts text to speech and saves it to a unique WAV file.
        Returns the path to the generated file.
        """
        try:
            # Ensure the output directory exists
            audio_output_dir = Path("audio_responses")
            audio_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate a unique filename
            output_path = audio_output_dir / f"response_{os.urandom(8).hex()}_{voice_model}.wav"

            tts_model = self.tts_model_female if voice_model.lower() == "female" else self.tts_model_male
            if voice_model.lower() == "female":
                if not self.tts_model_female:
                    logger.error("Female TTS model not loaded.")
                    raise RuntimeError("Female TTS model not available.")
                
                # Try different speakers for VCTK VITS model
                speakers = ["p225", "p226", "p227"]  # Common VCTK speakers
                success = False
                for speaker in speakers:
                    try:
                        tts_model.tts_to_file(text=text, file_path=str(output_path), speaker=speaker)
                        success = True
                        logger.info(f"Generated female TTS with speaker {speaker}")
                        break
                    except Exception as e:
                        logger.warning(f"Failed to generate female TTS with speaker {speaker}: {str(e)}")
                
                if not success:
                    # Fallback to no specific speaker
                    try:
                        tts_model.tts_to_file(text=text, file_path=str(output_path))
                        logger.warning("Generated female TTS without specific speaker due to errors with all speakers")
                    except Exception as fallback_e:
                        logger.error(f"Failed female TTS even without speaker: {fallback_e}")
                        raise

            else:  # Male voice model (tacotron2-DDC)
                if not self.tts_model_male:
                    logger.error("Male TTS model not loaded.")
                    raise RuntimeError("Male TTS model not available.")
                tts_model.tts_to_file(text=text, file_path=str(output_path))
                logger.info("Generated male TTS")

            logger.info(f"Generated speech file: {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"Error in text-to-speech: {str(e)}")
            raise

# Instantiate a single SpeechProcessor instance
speech_processor = SpeechProcessor()