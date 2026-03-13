from pyannote.audio import Pipeline

class Diarizer:
    """
    Performs speaker diarization on an audio file.
    """
    def __init__(self, auth_token):
        """
        Initializes the Diarizer with a Hugging Face authentication token.

        Args:
            auth_token (str): Your Hugging Face authentication token.
        """
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-cambridge-1.0",
            use_auth_token=auth_token
        )

    def diarize(self, file_path):
        """
        Performs speaker diarization on the given audio file.

        Args:
            file_path (str): The path to the audio file.

        Returns:
            pyannote.core.Annotation: An object containing the speaker segments.
        """
        return self.pipeline(file_path)