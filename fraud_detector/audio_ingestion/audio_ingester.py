from pydub import AudioSegment
from pydub.silence import split_on_silence
import os

class AudioIngester:
    """
    Handles the ingestion and initial processing of audio files.
    Now supports MP4 and other ffmpeg-compatible formats.
    """
    def __init__(self, file_path):
        """
        Initializes the AudioIngester with the path to the media file.

        Args:
            file_path (str): The path to the media file (e.g., .mp4, .wav, .mp3).
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The specified file was not found: {file_path}")
            
        print(f"Loading audio from: {file_path}")
        # Use 'from_file' which can handle various formats, including mp4
        self.audio = AudioSegment.from_file(file_path)

    def get_audio_chunks(self, min_silence_len=700, silence_thresh=-45, keep_silence=300):
        """
        Splits the audio into chunks based on silence. This is a form of
        Voice Activity Detection.

        Args:
            min_silence_len (int): Min length of silence to split on (in ms).
            silence_thresh (int): The silence threshold in dBFS (decibels relative to full scale).
            keep_silence (int): Amount of silence to keep at the beginning/end of chunks (in ms).

        Returns:
            list: A list of audio chunks (pydub.AudioSegment).
        """
        print("Splitting audio into chunks based on silence...")
        return split_on_silence(
            self.audio,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
            keep_silence=keep_silence
        )