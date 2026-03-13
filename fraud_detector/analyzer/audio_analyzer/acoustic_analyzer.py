import librosa
import numpy as np
from scipy import stats
from scipy.signal import find_peaks

class AcousticAnalyzer:
    """
    Enhanced acoustic analyzer for fraud detection with comprehensive feature extraction.
    Analyzes audio characteristics that may indicate fraudulent behavior.
    """
    
    def __init__(self):
        """Initialize the acoustic analyzer with optimized parameters"""
        # Thresholds for fraud detection (calibrated for accuracy)
        self.ENERGY_SPIKE_THRESHOLD = 1.8  # Higher threshold for energy spikes
        self.PITCH_VARIANCE_THRESHOLD = 2000  # Threshold for pitch variance
        self.BACKGROUND_NOISE_THRESHOLD = 0.01  # Threshold for background noise
        self.SPEECH_RATE_THRESHOLD = 0.3  # Threshold for speech rate analysis
        
    def analyze_chunk(self, audio_chunk):
        """
        Comprehensive analysis of acoustic features for fraud detection.

        Args:
            audio_chunk (pydub.AudioSegment): The audio chunk to analyze.

        Returns:
            dict: A dictionary of acoustic features with fraud indicators.
        """
        try:
            # Convert pydub audio segment to a numpy array for librosa
            samples = np.array(audio_chunk.get_array_of_samples())
            
            # Ensure the samples are floating-point for librosa
            if audio_chunk.sample_width == 2:
                samples = samples.astype(np.float32) / 32768.0

                # Normalize audio
                if np.max(np.abs(samples)) > 0:
                    samples = samples / np.max(np.abs(samples))

                # Extract comprehensive features
            features = {
                    # Basic energy features
                    "rms_energy": self._get_rms_energy(samples),
                    "max_amplitude": self._get_max_amplitude(samples),
                    "energy_spikes": self._get_energy_spikes(samples),
                    "energy_variance": self._get_energy_variance(samples),
                    
                    # Pitch and frequency features
                "pitch_variance": self._get_pitch_variance(samples, audio_chunk.frame_rate),
                    "pitch_mean": self._get_pitch_mean(samples, audio_chunk.frame_rate),
                    "spectral_centroid": self._get_spectral_centroid(samples, audio_chunk.frame_rate),
                    "spectral_rolloff": self._get_spectral_rolloff(samples, audio_chunk.frame_rate),
                    
                    # Temporal features
                    "zero_crossing_rate": self._get_zero_crossing_rate(samples),
                    "speech_rate": self._get_speech_rate(samples, audio_chunk.frame_rate),
                    "pause_ratio": self._get_pause_ratio(samples),
                    
                    # Background noise and quality
                    "background_noise": self._get_background_noise(samples),
                    "signal_to_noise_ratio": self._get_signal_to_noise_ratio(samples),
                    "spectral_bandwidth": self._get_spectral_bandwidth(samples, audio_chunk.frame_rate),
                    
                    # Fraud-specific indicators
                    "stress_indicators": self._get_stress_indicators(samples, audio_chunk.frame_rate),
                    "voice_quality": self._get_voice_quality(samples, audio_chunk.frame_rate),
                    "rhythm_irregularity": self._get_rhythm_irregularity(samples, audio_chunk.frame_rate)
                }
                
            # Calculate fraud risk score based on acoustic features
            features["acoustic_fraud_score"] = self._calculate_acoustic_fraud_score(features)
            
            return features

        except Exception as e:
            print(f"Error in acoustic analysis: {e}")
            return self._get_default_features()

    def _get_rms_energy(self, y):
        """Calculate RMS energy"""
        return float(np.sqrt(np.mean(y**2)))

    def _get_max_amplitude(self, y):
        """Calculate maximum amplitude"""
        return float(np.max(np.abs(y)))

    def _get_energy_spikes(self, y, threshold=None):
        """Detect energy spikes that may indicate stress or urgency"""
        if threshold is None:
            threshold = self.ENERGY_SPIKE_THRESHOLD
            
        rms = librosa.feature.rms(y=y)[0]
        mean_energy = np.mean(rms)
        std_energy = np.std(rms)
        
        # Count spikes above threshold
        spike_threshold = mean_energy + threshold * std_energy
        spikes = np.sum(rms > spike_threshold)
        
        return int(spikes)

    def _get_energy_variance(self, y):
        """Calculate energy variance"""
        rms = librosa.feature.rms(y=y)[0]
        return float(np.var(rms))

    def _get_pitch_variance(self, y, sr):
        """Calculate pitch variance with improved robustness"""
        try:
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr, threshold=0.1)
            non_zero_pitches = pitches[pitches > 0]
            
            if len(non_zero_pitches) > 10:  # Need sufficient data
                return float(np.var(non_zero_pitches))
            return 0.0
        except:
            return 0.0

    def _get_pitch_mean(self, y, sr):
        """Calculate mean pitch"""
        try:
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr, threshold=0.1)
            non_zero_pitches = pitches[pitches > 0]
            
            if len(non_zero_pitches) > 0:
                return float(np.mean(non_zero_pitches))
            return 0.0
        except:
            return 0.0

    def _get_spectral_centroid(self, y, sr):
        """Calculate spectral centroid"""
        try:
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            return float(np.mean(spectral_centroids))
        except:
            return 0.0

    def _get_spectral_rolloff(self, y, sr):
        """Calculate spectral rolloff"""
        try:
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            return float(np.mean(spectral_rolloff))
        except:
            return 0.0

    def _get_zero_crossing_rate(self, y):
        """Calculate zero crossing rate"""
        try:
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            return float(np.mean(zcr))
        except:
            return 0.0

    def _get_speech_rate(self, y, sr):
        """Estimate speech rate based on energy patterns"""
        try:
            # Use energy-based voice activity detection
            rms = librosa.feature.rms(y=y)[0]
            energy_threshold = np.percentile(rms, 30)  # Bottom 30% as silence
            
            # Count speech segments
            speech_segments = rms > energy_threshold
            speech_rate = np.sum(speech_segments) / len(speech_segments)
            
            return float(speech_rate)
        except:
            return 0.0

    def _get_pause_ratio(self, y):
        """Calculate ratio of pauses in speech"""
        try:
            rms = librosa.feature.rms(y=y)[0]
            energy_threshold = np.percentile(rms, 20)  # Bottom 20% as silence
            
            pauses = rms <= energy_threshold
            pause_ratio = np.sum(pauses) / len(pauses)
            
            return float(pause_ratio)
        except:
            return 0.0

    def _get_background_noise(self, y):
        """Estimate background noise level"""
        try:
            rms = librosa.feature.rms(y=y)[0]
            # Use bottom 10% as noise floor
            noise_level = np.percentile(rms, 10)
            return float(noise_level)
        except:
            return 0.0

    def _get_signal_to_noise_ratio(self, y):
        """Calculate signal-to-noise ratio"""
        try:
            rms = librosa.feature.rms(y=y)[0]
            signal_level = np.mean(rms)
            noise_level = np.percentile(rms, 10)
            
            if noise_level > 0:
                snr = 20 * np.log10(signal_level / noise_level)
                return float(snr)
            return 0.0
        except:
            return 0.0

    def _get_spectral_bandwidth(self, y, sr):
        """Calculate spectral bandwidth"""
        try:
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            return float(np.mean(spectral_bandwidth))
        except:
            return 0.0

    def _get_stress_indicators(self, y, sr):
        """Detect stress indicators in voice"""
        try:
            # High pitch variance and energy spikes indicate stress
            pitch_var = self._get_pitch_variance(y, sr)
            energy_spikes = self._get_energy_spikes(y)
            
            # Normalize and combine
            stress_score = min(1.0, (pitch_var / 5000) + (energy_spikes / 20))
            return float(stress_score)
        except:
            return 0.0

    def _get_voice_quality(self, y, sr):
        """Assess voice quality (lower values may indicate poor quality or manipulation)"""
        try:
            # Use spectral centroid and bandwidth as quality indicators
            spectral_centroid = self._get_spectral_centroid(y, sr)
            spectral_bandwidth = self._get_spectral_bandwidth(y, sr)
            
            # Normalize quality score
            quality_score = min(1.0, (spectral_centroid / 4000) + (spectral_bandwidth / 2000))
            return float(quality_score)
        except:
            return 0.0

    def _get_rhythm_irregularity(self, y, sr):
        """Detect irregular rhythm patterns"""
        try:
            # Use energy-based rhythm analysis
            rms = librosa.feature.rms(y=y)[0]
            
            # Find peaks in energy (rhythm markers)
            peaks, _ = find_peaks(rms, height=np.mean(rms))
            
            if len(peaks) > 2:
                # Calculate intervals between peaks
                intervals = np.diff(peaks)
                # High variance in intervals indicates irregular rhythm
                rhythm_irregularity = np.var(intervals) / (np.mean(intervals) + 1e-8)
                return float(min(1.0, rhythm_irregularity / 10))
            
            return 0.0
        except:
            return 0.0

    def _calculate_acoustic_fraud_score(self, features):
        """Calculate overall acoustic fraud score based on all features"""
        try:
            score = 0.0
            
            # Energy-based indicators (weight: 0.3)
            if features["energy_spikes"] > 5:
                score += 0.3 * min(1.0, features["energy_spikes"] / 20)
            
            # Pitch-based indicators (weight: 0.25)
            if features["pitch_variance"] > self.PITCH_VARIANCE_THRESHOLD:
                score += 0.25 * min(1.0, features["pitch_variance"] / 10000)
            
            # Stress indicators (weight: 0.2)
            score += 0.2 * features["stress_indicators"]
            
            # Voice quality indicators (weight: 0.15)
            if features["voice_quality"] < 0.3:  # Poor voice quality
                score += 0.15 * (1.0 - features["voice_quality"])
            
            # Rhythm irregularity (weight: 0.1)
            score += 0.1 * features["rhythm_irregularity"]
            
            return float(min(1.0, score))
        except:
            return 0.0

    def _get_default_features(self):
        """Return default features in case of error"""
        return {
            "rms_energy": 0.0,
            "max_amplitude": 0.0,
            "energy_spikes": 0,
            "energy_variance": 0.0,
            "pitch_variance": 0.0,
            "pitch_mean": 0.0,
            "spectral_centroid": 0.0,
            "spectral_rolloff": 0.0,
            "zero_crossing_rate": 0.0,
            "speech_rate": 0.0,
            "pause_ratio": 0.0,
            "background_noise": 0.0,
            "signal_to_noise_ratio": 0.0,
            "spectral_bandwidth": 0.0,
            "stress_indicators": 0.0,
            "voice_quality": 0.0,
            "rhythm_irregularity": 0.0,
            "acoustic_fraud_score": 0.0
        }