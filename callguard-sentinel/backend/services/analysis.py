"""
Analysis Service
Computes scam probability and returns one final decision.
"""

from fastapi import APIRouter
from typing import Dict, Any, List
import logging
import os
import joblib
import numpy as np
import httpx
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from datetime import datetime

from .schemas import FeaturePayload
#from config import settings
from config import (
    ALERT_BROADCAST_URL,
    ALERT_SERVICE_URL,
    SAFE_THRESHOLD,
    WARNING_THRESHOLD,
    CRITICAL_THRESHOLD,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

analysis_router = APIRouter()


class ScamDetectionModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_order = [
            "authority_claims",
            "urgency_language",
            "threat_language",
            "bait_language",
            "sensitive_info_requests",
            "negative_sentiment",
            "compound_sentiment",
            "text_length",
            "word_count",
        ]
        self.load_or_initialize_model()

    def load_or_initialize_model(self):
        model_path = "models/scam_detection_model.pkl"
        scaler_path = "models/scam_scaler.pkl"

        if os.path.exists(model_path) and os.path.exists(scaler_path):
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.is_trained = True
            return

        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42
        )
        self.is_trained = False

    def rule_based_score(self, features: Dict[str, Any]) -> float:
        score = 0.0
        score += min(features.get("authority_claims", 0) * 0.18, 0.36)
        score += min(features.get("urgency_language", 0) * 0.20, 0.40)
        score += min(features.get("threat_language", 0) * 0.22, 0.44)
        score += min(features.get("bait_language", 0) * 0.12, 0.24)
        score += min(features.get("sensitive_info_requests", 0) * 0.25, 0.50)

        compound = features.get("compound_sentiment", 0.0)
        negative = features.get("negative_sentiment", 0.0)

        if compound < -0.4:
            score += 0.08
        if negative > 0.4:
            score += 0.08

        return min(score, 1.0)

    def model_score(self, features: Dict[str, Any]) -> float:
        if not self.is_trained:
            return self.rule_based_score(features)

        row = np.array([[features.get(name, 0.0) for name in self.feature_order]])
        row_scaled = self.scaler.transform(row)
        proba = self.model.predict_proba(row_scaled)[0][1]
        return float(proba)

    def explain_reasons(self, features: Dict[str, Any]) -> List[str]:
        reasons = []
        if features.get("authority_claims", 0) > 0:
            reasons.append("Authority indicators detected")
        if features.get("urgency_language", 0) > 0:
            reasons.append("Urgency patterns identified")
        if features.get("threat_language", 0) > 0:
            reasons.append("Threat language detected")
        if features.get("sensitive_info_requests", 0) > 0:
            reasons.append("Request for sensitive information")
        if features.get("bait_language", 0) > 0:
            reasons.append("Prize or reward bait language detected")
        return reasons

    def label_from_score(self, score: float) -> str:
        #if score >= settings.CRITICAL_THRESHOLD:
        if score >= CRITICAL_THRESHOLD:

            return "critical"
        #if score >= settings.WARNING_THRESHOLD:
        if score >= WARNING_THRESHOLD:

            return "warning"
        return "safe"


analysis_model = ScamDetectionModel()


# async def send_alert_if_needed(
#     call_id: str,
#     user_id: str,
#     final_label: str,
#     final_score: float,
#     reasons: List[str],
#     features: Dict[str, Any],
# ):
#     if final_label == "safe":
#         return

#     payload = {
#         "call_id": call_id,
#         "user_id": user_id,
#         "level": final_label,
#         "reason": "; ".join(reasons) if reasons else "Suspicious call indicators detected",
#         "confidence": final_score,
#         "features": features,
#     }

#     try:
#         async with httpx.AsyncClient(timeout=10.0) as client:
#             #await client.post(f"{settings.ALERT_SERVICE_URL}/api/alerts/send", json=payload)
#             await client.post(ALERT_SERVICE_URL, json=payload)
#     except Exception as exc:
#         logger.exception("Failed to send alert for call_id=%s: %s", call_id, exc)


async def send_alert_if_needed(
    call_id: str,
    user_id: str,
    final_label: str,
    final_score: float,
    reasons: List[str],
    features: Dict[str, Any],
):
    if final_label == "safe":
        return

    payload = {
        "call_id": call_id,
        "user_id": user_id,
        "level": final_label,
        "reason": "; ".join(reasons) if reasons else "Suspicious call indicators detected",
        "confidence": final_score,
        "features": features,
        "timestamp": datetime.utcnow().isoformat(),
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(ALERT_SERVICE_URL, json=payload)
    except Exception as exc:
        logger.exception("Failed to send alert for call_id=%s: %s", call_id, exc)

@analysis_router.post("/predict")
async def predict_scam(payload: FeaturePayload):
    realtime_score = analysis_model.rule_based_score(payload.features)
    model_score = analysis_model.model_score(payload.features)

    final_score = max(realtime_score, model_score)
    final_label = analysis_model.label_from_score(final_score)
    reasons = analysis_model.explain_reasons(payload.features)

    await send_alert_if_needed(
        call_id=payload.call_id,
        user_id=payload.user_id,
        final_label=final_label,
        final_score=final_score,
        reasons=reasons,
        features=payload.features,
    )

    return {
        "status": "predicted",
        "call_id": payload.call_id,
        "realtime_score": round(realtime_score, 4),
        "model_score": round(model_score, 4),
        "final_score": round(final_score, 4),
        "final_label": final_label,
        "is_scam": final_label in {"warning", "critical"},
        "confidence": round(final_score, 4),
        "reasons": reasons,
        "features": payload.features,
    }


# """
# Analysis Service
# ML model for scam detection based on extracted features
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
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import StandardScaler
# import joblib
# import os

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# analysis_router = APIRouter()

# class ScamDetectionModel:
#     def __init__(self):
#         self.model = None
#         self.scaler = StandardScaler()
#         self.feature_names = []
#         self.is_trained = False
        
#         # Load or initialize model
#         self.load_or_initialize_model()
    
#     def load_or_initialize_model(self):
#         """Load existing model or initialize new one"""
#         model_path = "models/scam_detection_model.pkl"
#         scaler_path = "models/scam_scaler.pkl"
        
#         if os.path.exists(model_path) and os.path.exists(scaler_path):
#             try:
#                 self.model = joblib.load(model_path)
#                 self.scaler = joblib.load(scaler_path)
#                 self.is_trained = True
#                 logger.info("Loaded existing scam detection model")
#             except Exception as e:
#                 logger.error(f"Error loading model: {e}")
#                 self.initialize_new_model()
#         else:
#             self.initialize_new_model()
    
#     def initialize_new_model(self):
#         """Initialize a new model with default parameters"""
#         self.model = RandomForestClassifier(
#             n_estimators=100,
#             max_depth=10,
#             random_state=42,
#             class_weight='balanced'
#         )
#         self.is_trained = False
#         logger.info("Initialized new scam detection model")
    
#     def prepare_features(self, features: Dict) -> np.ndarray:
#         """Prepare features for model prediction"""
#         # Define expected feature names
#         expected_features = [
#             'authority', 'urgency', 'threat', 'bait', 'sensitivity', 'repetition',
#             'language_switching', 'sentiment_negative', 'sentiment_neutral', 
#             'sentiment_positive', 'sentiment_compound', 'word_count', 'sentence_count',
#             'avg_word_length', 'avg_sentence_length', 'question_marks', 'exclamation_marks',
#             'turn_taking', 'pause_length', 'speech_rate', 'background_noise',
#             'energy_spikes', 'pitch_raising', 'caps_usage', 'number_usage', 'special_char_usage',
#             'ctx_call_time_hour', 'ctx_is_known_contact'
#         ]
        
#         # Create feature vector
#         feature_vector = []
#         for feature in expected_features:
#             if feature in features:
#                 feature_vector.append(features[feature])
#             else:
#                 feature_vector.append(0.0)  # Default value for missing features
        
#         return np.array(feature_vector).reshape(1, -1)
    
#     def predict(self, features: Dict) -> Dict:
#         """Make prediction on features"""
#         if not self.is_trained:
#             # Use rule-based prediction as fallback
#             return self.rule_based_prediction(features)
        
#         try:
#             # Prepare features
#             X = self.prepare_features(features)
            
#             # Scale features
#             X_scaled = self.scaler.transform(X)
            
#             # Make prediction
#             prediction_proba = self.model.predict_proba(X_scaled)[0]
#             prediction = self.model.predict(X_scaled)[0]
            
#             # Get feature importance
#             feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
            
#             return {
#                 'prediction': int(prediction),
#                 'probability': float(prediction_proba[1]),  # Probability of being a scam
#                 'confidence': float(max(prediction_proba)),
#                 'feature_importance': feature_importance
#             }
            
#         except Exception as e:
#             logger.error(f"Error making prediction: {e}")
#             return self.rule_based_prediction(features)
    
#     def rule_based_prediction(self, features: Dict) -> Dict:
#         """Rule-based prediction as fallback"""
#         # Simple rule-based scoring
#         score = 0.0
        
#         # High-risk features
#         if features.get('authority', 0) > 0.3:
#             score += 0.2
#         if features.get('urgency', 0) > 0.3:
#             score += 0.2
#         if features.get('threat', 0) > 0.2:
#             score += 0.3
#         if features.get('bait', 0) > 0.2:
#             score += 0.2
#         if features.get('sensitivity', 0) > 0.2:
#             score += 0.3
#         if features.get('repetition', 0) > 0.4:
#             score += 0.1
        
#         # Sentiment-based scoring
#         if features.get('sentiment_negative', 0) > 0.5:
#             score += 0.1
#         if features.get('sentiment_compound', 0) < -0.5:
#             score += 0.1
        
#         # Text pattern scoring
#         if features.get('caps_usage', 0) > 0.1:
#             score += 0.1
#         if features.get('exclamation_marks', 0) > 0.2:
#             score += 0.1
        
#         # Normalize score
#         score = min(score, 1.0)
        
#         return {
#             'prediction': 1 if score > 0.5 else 0,
#             'probability': score,
#             'confidence': 0.8 if score > 0.7 or score < 0.3 else 0.6,
#             'feature_importance': {}
#         }
    
#     def train_model(self, X: np.ndarray, y: np.ndarray):
#         """Train the model with new data"""
#         try:
#             # Scale features
#             X_scaled = self.scaler.fit_transform(X)
            
#             # Train model
#             self.model.fit(X_scaled, y)
#             self.is_trained = True
            
#             # Save model
#             os.makedirs("models", exist_ok=True)
#             joblib.dump(self.model, "models/scam_detection_model.pkl")
#             joblib.dump(self.scaler, "models/scam_scaler.pkl")
            
#             logger.info("Model trained and saved successfully")
#             return True
            
#         except Exception as e:
#             logger.error(f"Error training model: {e}")
#             return False

# class AnalysisService:
#     def __init__(self):
#         self.model = ScamDetectionModel()
#         self.alert_service_url = "http://localhost:8004/api/alerts/send"
#         self.threshold = 0.6  # Scam detection threshold
    
#     async def analyze_call(self, call_id: str, text: str, features: Dict, user_id: str, timestamp: str):
#         """Analyze call for scam indicators"""
#         try:
#             # Make prediction
#             prediction_result = self.model.predict(features)
            
#             is_scam = prediction_result['prediction'] == 1
#             scam_probability = prediction_result['probability']
#             confidence = prediction_result['confidence']
            
#             # Determine alert level
#             if is_scam and scam_probability > 0.8:
#                 alert_level = "critical"
#             elif is_scam and scam_probability > 0.6:
#                 alert_level = "warning"
#             else:
#                 alert_level = "safe"
            
#             # Generate reason
#             reason = self.generate_scam_reason(features, prediction_result)
            
#             # Store prediction
#             await self.store_prediction(call_id, text, features, scam_probability)
            
#             # Send alert if scam detected
#             if is_scam and scam_probability > self.threshold:
#                 await self.send_alert(call_id, user_id, alert_level, reason, scam_probability, features)

#             # Stream rolling score to frontend for visualization
#             await self.stream_score_update(user_id, call_id, scam_probability, alert_level)
            
#             return {
#                 'is_scam': is_scam,
#                 'probability': scam_probability,
#                 'confidence': confidence,
#                 'alert_level': alert_level,
#                 'reason': reason,
#                 'features': features
#             }
            
#         except Exception as e:
#             logger.error(f"Error analyzing call {call_id}: {e}")
#             raise HTTPException(status_code=500, detail=str(e))
    
#     def generate_scam_reason(self, features: Dict, prediction_result: Dict) -> str:
#         """Generate human-readable reason for scam detection"""
#         reasons = []
        
#         if features.get('authority', 0) > 0.3:
#             reasons.append("Authority claims detected")
#         if features.get('urgency', 0) > 0.3:
#             reasons.append("Urgent language used")
#         if features.get('threat', 0) > 0.2:
#             reasons.append("Threatening language detected")
#         if features.get('bait', 0) > 0.2:
#             reasons.append("Baiting tactics used")
#         if features.get('sensitivity', 0) > 0.2:
#             reasons.append("Request for sensitive information")
#         if features.get('repetition', 0) > 0.4:
#             reasons.append("Excessive repetition detected")
        
#         if not reasons:
#             reasons.append("Suspicious communication patterns")
        
#         return "; ".join(reasons)
    
#     async def store_prediction(self, call_id: str, text: str, features: Dict, probability: float):
#         """Store prediction in database"""
#         # This would store in the database
#         logger.info(f"Stored prediction for call {call_id}: {probability}")
    
#     async def send_alert(self, call_id: str, user_id: str, level: str, reason: str, confidence: float, features: Dict):
#         """Send alert to user"""
#         try:
#             payload = {
#                 "call_id": call_id,
#                 "user_id": user_id,
#                 "level": level,
#                 "reason": reason,
#                 "confidence": confidence,
#                 "features": features,
#                 "timestamp": datetime.utcnow().isoformat()
#             }
            
#             async with httpx.AsyncClient() as client:
#                 response = await client.post(
#                     self.alert_service_url,
#                     json=payload,
#                     timeout=5.0
#                 )
                
#                 if response.status_code != 200:
#                     logger.error(f"Alert service error: {response.text}")
                    
#         except Exception as e:
#             logger.error(f"Error sending alert: {e}")

#     async def stream_score_update(self, user_id: str, call_id: str, score: float, level: str):
#         try:
#             import httpx
#             payload = {
#                 "type": "call_analysis_update",
#                 "call_id": call_id,
#                 "user_id": user_id,
#                 "score": score,
#                 "level": level,
#                 "timestamp": datetime.utcnow().isoformat(),
#             }
#             async with httpx.AsyncClient() as client:
#                 await client.post("http://localhost:8004/api/alerts/broadcast", json=payload, timeout=3.0)
#         except Exception:
#             pass

# # Initialize service
# analysis_service = AnalysisService()

# @analysis_router.post("/predict")
# async def predict_scam(
#     call_id: str,
#     text: str,
#     features: Dict,
#     user_id: str,
#     timestamp: str
# ):
#     """Analyze call for scam indicators"""
#     result = await analysis_service.analyze_call(call_id, text, features, user_id, timestamp)
#     return result

# @analysis_router.post("/train")
# async def train_model(training_data: List[Dict]):
#     """Train the model with new data"""
#     try:
#         X = np.array([item['features'] for item in training_data])
#         y = np.array([item['label'] for item in training_data])
        
#         success = analysis_service.model.train_model(X, y)
        
#         if success:
#             return {"status": "trained", "message": "Model trained successfully"}
#         else:
#             raise HTTPException(status_code=500, detail="Training failed")
            
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @analysis_router.get("/model/status")
# async def get_model_status():
#     """Get model status and statistics"""
#     return {
#         "is_trained": analysis_service.model.is_trained,
#         "threshold": analysis_service.threshold,
#         "feature_count": len(analysis_service.model.feature_names)
#     }
