from faster_whisper import WhisperModel
import numpy as np
import platform
import warnings

class Transcriber:
    """
    Transcribes and translates audio from any language into English using
    Faster Whisper. This is the most robust approach for a standardized
    analysis pipeline.
    """
    def __init__(self, model_size="small", compute_type="auto"):
        """
        Initializes the translating transcriber.

        Args:
            model_size (str): "base", "small", "medium". "small" is recommended
                              for a good balance of speed and translation quality.
            compute_type (str): "int8" is recommended for fast performance on CPUs.
        """
        if compute_type == "auto":
            compute_type = "int8"
        
        print(f"Loading Faster Whisper model: {model_size} (for translation) with {compute_type}...")
        
        warnings.filterwarnings("ignore", category=FutureWarning)
        
        # Initialize model for CPU execution
        self.model = WhisperModel(
            model_size,
            device="cpu",
            compute_type=compute_type,
            cpu_threads=4
        )
        
        print(f"✓ Model loaded successfully and configured for English translation.")

    def translate_entire_file(self, file_path: str, initial_prompt: str = None) -> dict:
        """
        Transcribes and translates an entire audio file at once for maximum performance.

        Args:
            file_path (str): The path to the audio file.
            initial_prompt (str, optional): A prompt to guide the model.

        Returns:
            dict: A dictionary containing the full translated text and detected language info.
        """
        try:
            # The transcribe method can directly accept a file path.
            segments, info = self.model.transcribe(
                audio=file_path,
                task="translate",
                initial_prompt=initial_prompt,
                vad_filter=True # Use Voice Activity Detection for better accuracy
            )

            # Efficiently join all segment texts into one string.
            full_text = " ".join(segment.text.strip() for segment in segments)
            
            return {
                "full_text": full_text,
                "detected_language": info.language,
                "language_probability": info.language_probability
            }

        except Exception as e:
            print(f"Full file translation failed: {e}")
            return None

    def transcribe_and_translate_chunk(self, audio_chunk, language=None, initial_prompt=None):
        """
        Transcribes audio and translates it to English.

        Args:
            audio_chunk (pydub.AudioSegment): The audio chunk to process.
            language (str, optional): Language code (e.g., "hi", "es"). If None,
                                     Whisper will auto-detect the language.
                                     Autodetection is best for translation.
            initial_prompt (str, optional): A prompt in English to guide the model.

        Returns:
            str: The translated English text, or None if it fails.
        """
        try:
            # Prepare audio in the format Whisper expects (16kHz mono float32)
            audio_chunk = audio_chunk.set_frame_rate(16000).set_channels(1)
            samples = np.array(audio_chunk.get_array_of_samples()).astype(np.float32) / 32768.0
            
            # --- THE CORE CHANGE ---
            # Use task="translate" to get English output directly.
            segments, info = self.model.transcribe(
                samples,
                language=language,
                initial_prompt=initial_prompt,
                task="translate", # This tells Whisper to output English
                vad_filter=True
            )
            
            segments_list = list(segments)
            if not segments_list:
                return None
            
            # Combine the translated text from all segments
            translated_text = " ".join([segment.text.strip() for segment in segments_list])

            # Print the detected source language for debugging/info
            print(f"    (Detected source language: {info.language} with probability {info.language_probability:.2f})")
            
            return translated_text
            
        except Exception as e:
            print(f"Translation failed: {e}")
            return None