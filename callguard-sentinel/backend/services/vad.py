"""
Voice Activity Detection (VAD)
Lightweight wrapper to mark audio chunks as speech/non-speech.
In production, replace with WebRTC VAD or pyannote's VAD for higher accuracy.
"""

import numpy as np
import base64
import io
import soundfile as sf

class SimpleEnergyVAD:
    def __init__(self, energy_threshold: float = 0.01):
        self.energy_threshold = energy_threshold

    def is_speech_wav_bytes(self, wav_bytes: bytes) -> bool:
        try:
            data, sr = sf.read(io.BytesIO(wav_bytes))
            if data.size == 0:
                return False
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            energy = float(np.mean(np.square(data)))
            return energy > self.energy_threshold
        except Exception:
            return True  # fail-open to avoid dropping all audio

    def is_speech_base64_wav(self, b64: str) -> bool:
        try:
            wav_bytes = base64.b64decode(b64)
            return self.is_speech_wav_bytes(wav_bytes)
        except Exception:
            return True

vad = SimpleEnergyVAD()


