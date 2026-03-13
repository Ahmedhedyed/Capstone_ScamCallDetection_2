
"""
Feature Extraction Service
Extracts text-based scam indicators and forwards them to analysis.
"""

from fastapi import APIRouter
from datetime import datetime
from typing import Dict, Any
import logging
import re
import httpx
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords

from .schemas import TranscriptionToFeaturePayload
# from config import settings
from config import ANALYSIS_SERVICE_URL

try:
    nltk.download("vader_lexicon", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
except Exception:
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

feature_router = APIRouter()


class FeatureExtractionService:
    def __init__(self):
        #self.analysis_service_url = f"{settings.ANALYSIS_SERVICE_URL}/api/analysis/predict"
        self.analysis_service_url = ANALYSIS_SERVICE_URL
        try:
            self.sia = SentimentIntensityAnalyzer()
        except Exception as e:
            logger.error(f"Failed to initialize SentimentIntensityAnalyzer: {e}")
            self.sia = None
        try:
            self.stop_words = set(stopwords.words("english"))
        except Exception as e:
            logger.error(f"Failed to load NLTK stopwords: {e}")
            self.stop_words = set()

        self.scam_keywords = {
            "authority": [
                "bank", "government", "police", "court", "irs", "fbi", "official"
            ],
            "urgency": [
                "urgent", "immediately", "now", "asap", "right now", "hurry", "quickly", "emergency"
            ],
            "threat": [
                "arrest", "warrant", "jail", "prison", "fine", "penalty", "suspended", "blocked", "frozen"
            ],
            "bait": [
                "prize", "winner", "congratulations", "free", "gift", "bonus", "reward", "lottery"
            ],
            "sensitivity": [
                "ssn", "social security", "credit card", "bank account", "password", "pin", "otp", "personal"
            ],
        }

    async def extract_features(self, payload: TranscriptionToFeaturePayload) -> Dict[str, Any]:
        text = payload.text.lower()
        sentiment = self.sia.polarity_scores(text)
        tokens = re.findall(r"\b\w+\b", text)

        keyword_hits = {
            category: sum(text.count(term) for term in terms)
            for category, terms in self.scam_keywords.items()
        }

        features = {
            "text_length": len(text),
            "word_count": len(tokens),
            "negative_sentiment": sentiment["neg"],
            "neutral_sentiment": sentiment["neu"],
            "positive_sentiment": sentiment["pos"],
            "compound_sentiment": sentiment["compound"],
            "authority_claims": keyword_hits["authority"],
            "urgency_language": keyword_hits["urgency"],
            "threat_language": keyword_hits["threat"],
            "bait_language": keyword_hits["bait"],
            "sensitive_info_requests": keyword_hits["sensitivity"],
            "pause_length": 0.0,
            "background_noise": 0.0,
            "energy_spikes": 0.0,
            "pitch_raising": 0.0,
            "is_known_contact": 0.0,
        }

        return features

    async def forward_to_analysis(
        self,
        payload: TranscriptionToFeaturePayload,
        features: Dict[str, Any]
    ):
        request_body = {
            "call_id": payload.call_id,
            "user_id": payload.user_id,
            "timestamp": payload.timestamp,
            "text": payload.text,
            "features": features,
        }

        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.post(self.analysis_service_url, json=request_body)
                response.raise_for_status()
                return response.json()
        except Exception as exc:
            logger.exception("Failed forwarding to analysis for call_id=%s: %s", payload.call_id, exc)
            return {"status": "failed", "error": str(exc)}


feature_service = FeatureExtractionService()


@feature_router.post("/extract")
async def extract_features(payload: TranscriptionToFeaturePayload):
    features = await feature_service.extract_features(payload)
    analysis_result = await feature_service.forward_to_analysis(payload, features)

    return {
        "status": "extracted",
        "call_id": payload.call_id,
        "features": features,
        "analysis_result": analysis_result,
    }



# """
# Feature Extraction Service
# Extracts linguistic, conversational, and agnostic features from transcribed text and audio
# """

# from fastapi import APIRouter, HTTPException
# import asyncio
# import json
# import logging
# import numpy as np
# import pandas as pd
# from datetime import datetime
# from typing import Dict, List, Optional
# import httpx
# import re
# from collections import Counter
# import nltk
# from nltk.sentiment import SentimentIntensityAnalyzer
# from nltk.tokenize import word_tokenize, sent_tokenize
# from nltk.corpus import stopwords
# import librosa
# import soundfile as sf
# import tempfile
# import os

# # Download required NLTK data
# try:
#     nltk.download('vader_lexicon', quiet=True)
#     nltk.download('punkt', quiet=True)
#     nltk.download('stopwords', quiet=True)
# except:
#     pass

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# feature_router = APIRouter()

# class FeatureExtractionService:
#     def __init__(self):
#         self.analysis_service_url = "http://localhost:8003/api/analysis/predict"
#         self.sia = SentimentIntensityAnalyzer()
#         self.stop_words = set(stopwords.words('english'))
        
#         # Scam-related keywords and patterns
#         self.scam_keywords = {
#             'authority': ['bank', 'government', 'police', 'court', 'irs', 'fbi', 'official', 'urgent', 'immediately'],
#             'urgency': ['urgent', 'immediately', 'now', 'asap', 'right now', 'hurry', 'quickly', 'emergency'],
#             'threat': ['arrest', 'warrant', 'jail', 'prison', 'fine', 'penalty', 'suspended', 'blocked', 'frozen'],
#             'bait': ['prize', 'winner', 'congratulations', 'free', 'gift', 'bonus', 'reward', 'lottery'],
#             'sensitivity': ['ssn', 'social security', 'credit card', 'bank account', 'password', 'pin', 'personal'],
#             'repetition': ['repeat', 'again', 'same', 'similar', 'duplicate']
#         }
    
#     async def extract_features(self, call_id: str, text: str, user_id: str, timestamp: str):
#         """Extract all features from transcribed text"""
#         try:
#             features = {}
            
#             # Extract linguistic features
#             linguistic_features = await self.extract_linguistic_features(text)
#             features['linguistic'] = linguistic_features
            
#             # Extract conversational features (would need audio data for full implementation)
#             conversational_features = await self.extract_conversational_features(text)
#             features['conversational'] = conversational_features
            
#             # Extract agnostic features (would need audio data for full implementation)
#             agnostic_features = await self.extract_agnostic_features(text)
#             features['agnostic'] = agnostic_features
            
#             # Context features (temporal/contextual metadata placeholder)
#             context_features = {
#                 "call_time_hour": datetime.utcnow().hour / 24,
#                 "is_known_contact": 0.0,  # TODO: populate from DB
#             }

#             # Combine all features for model
#             all_features = {
#                 **linguistic_features,
#                 **conversational_features,
#                 **agnostic_features,
#                 **context_features,
#             }
            
#             logger.info(f"Extracted features for call {call_id}: {len(all_features)} features")
            
#             # Forward to analysis service
#             await self.forward_to_analysis(call_id, text, all_features, user_id)
            
#             return features
            
#         except Exception as e:
#             logger.error(f"Error extracting features for call {call_id}: {e}")
#             raise HTTPException(status_code=500, detail=str(e))
    
#     async def extract_linguistic_features(self, text: str) -> Dict[str, float]:
#         """Extract linguistic features from text"""
#         try:
#             # Clean and tokenize text
#             text_lower = text.lower()
#             words = word_tokenize(text_lower)
#             sentences = sent_tokenize(text)
            
#             # Remove stopwords
#             words_filtered = [word for word in words if word not in self.stop_words and word.isalpha()]
            
#             features = {}
            
#             # Authority score
#             authority_words = sum(1 for word in words if word in self.scam_keywords['authority'])
#             features['authority'] = min(authority_words / max(len(words), 1), 1.0)
            
#             # Urgency score
#             urgency_words = sum(1 for word in words if word in self.scam_keywords['urgency'])
#             features['urgency'] = min(urgency_words / max(len(words), 1), 1.0)
            
#             # Threat score
#             threat_words = sum(1 for word in words if word in self.scam_keywords['threat'])
#             features['threat'] = min(threat_words / max(len(words), 1), 1.0)
            
#             # Bait score
#             bait_words = sum(1 for word in words if word in self.scam_keywords['bait'])
#             features['bait'] = min(bait_words / max(len(words), 1), 1.0)
            
#             # Sensitivity score
#             sensitivity_words = sum(1 for word in words if word in self.scam_keywords['sensitivity'])
#             features['sensitivity'] = min(sensitivity_words / max(len(words), 1), 1.0)
            
#             # Repetition score
#             word_counts = Counter(words)
#             repeated_words = sum(1 for count in word_counts.values() if count > 1)
#             features['repetition'] = min(repeated_words / max(len(word_counts), 1), 1.0)
            
#             # Language switching (simplified - would need more sophisticated detection)
#             features['language_switching'] = 0.0  # Placeholder
            
#             # Sentiment analysis
#             sentiment = self.sia.polarity_scores(text)
#             features['sentiment_negative'] = sentiment['neg']
#             features['sentiment_neutral'] = sentiment['neu']
#             features['sentiment_positive'] = sentiment['pos']
#             features['sentiment_compound'] = sentiment['compound']
            
#             # Text statistics
#             features['word_count'] = len(words)
#             features['sentence_count'] = len(sentences)
#             features['avg_word_length'] = np.mean([len(word) for word in words]) if words else 0
#             features['avg_sentence_length'] = np.mean([len(sent.split()) for sent in sentences]) if sentences else 0
            
#             # Question marks and exclamation marks
#             features['question_marks'] = text.count('?') / max(len(sentences), 1)
#             features['exclamation_marks'] = text.count('!') / max(len(sentences), 1)
            
#             return features
            
#         except Exception as e:
#             logger.error(f"Error extracting linguistic features: {e}")
#             return {}
    
#     async def extract_conversational_features(self, text: str) -> Dict[str, float]:
#         """Extract conversational features (simplified without audio)"""
#         try:
#             features = {}
            
#             # Turn taking (simplified - would need speaker diarization)
#             sentences = sent_tokenize(text)
#             features['turn_taking'] = min(len(sentences) / 10, 1.0)  # Normalized
            
#             # Pause length (simplified - would need audio analysis)
#             features['pause_length'] = 0.5  # Placeholder
            
#             # Speech rate (simplified - would need audio analysis)
#             words = word_tokenize(text)
#             features['speech_rate'] = min(len(words) / 100, 1.0)  # Normalized
            
#             return features
            
#         except Exception as e:
#             logger.error(f"Error extracting conversational features: {e}")
#             return {}
    
#     async def extract_agnostic_features(self, text: str) -> Dict[str, float]:
#         """Extract agnostic features (simplified without audio)"""
#         try:
#             features = {}
            
#             # Background noise (would need audio analysis)
#             features['background_noise'] = 0.0  # Placeholder
            
#             # Energy spikes (would need audio analysis)
#             features['energy_spikes'] = 0.0  # Placeholder
            
#             # Pitch raising (would need audio analysis)
#             features['pitch_raising'] = 0.0  # Placeholder
            
#             # Text-based agnostic features
#             words = word_tokenize(text.lower())
            
#             # Caps lock usage
#             caps_words = sum(1 for word in words if word.isupper() and len(word) > 1)
#             features['caps_usage'] = caps_words / max(len(words), 1)
            
#             # Number usage
#             numbers = sum(1 for word in words if word.isdigit())
#             features['number_usage'] = numbers / max(len(words), 1)
            
#             # Special characters
#             special_chars = sum(1 for char in text if not char.isalnum() and not char.isspace())
#             features['special_char_usage'] = special_chars / max(len(text), 1)
            
#             return features
            
#         except Exception as e:
#             logger.error(f"Error extracting agnostic features: {e}")
#             return {}
    
#     async def forward_to_analysis(self, call_id: str, text: str, features: Dict, user_id: str):
#         """Forward features to analysis service"""
#         try:
#             payload = {
#                 "call_id": call_id,
#                 "text": text,
#                 "features": features,
#                 "user_id": user_id,
#                 "timestamp": datetime.utcnow().isoformat()
#             }
            
#             async with httpx.AsyncClient() as client:
#                 response = await client.post(
#                     self.analysis_service_url,
#                     json=payload,
#                     timeout=10.0
#                 )
                
#                 if response.status_code != 200:
#                     logger.error(f"Analysis service error: {response.text}")
                    
#         except Exception as e:
#             logger.error(f"Error forwarding to analysis service: {e}")

# # Initialize service
# feature_service = FeatureExtractionService()

# @feature_router.post("/extract")
# async def extract_features(
#     call_id: str,
#     text: str,
#     user_id: str,
#     timestamp: str
# ):
#     """Extract features from transcribed text"""
#     features = await feature_service.extract_features(call_id, text, user_id, timestamp)
#     return {"features": features, "status": "extracted"}

# @feature_router.post("/extract-audio")
# async def extract_audio_features(
#     call_id: str,
#     audio_data: str,  # Base64 encoded audio
#     user_id: str
# ):
#     """Extract features from audio data (future implementation)"""
#     # This would implement audio-based feature extraction using librosa, openSMILE, etc.
#     return {"message": "Audio feature extraction not yet implemented"}

# @feature_router.get("/features/{call_id}")
# async def get_call_features(call_id: str):
#     """Get extracted features for a call"""
#     # This would retrieve stored features from database
#     return {"call_id": call_id, "features": {}}
