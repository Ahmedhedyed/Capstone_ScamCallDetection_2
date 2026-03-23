"""
Analysis Service
================
ML model for scam detection.

REPLACED:  sklearn RandomForestClassifier  (was never trained; always fell
           back to rule-based scoring anyway)
WITH:      TinyBERT semantic classifier  (understands call context without
           hand-crafted lexicons)

All existing API endpoints, request/response shapes, and fallback behaviour
are fully preserved.  The rule-based scorer is kept as the real-time fallback
when TinyBERT is unavailable or the text field is empty.
"""

from fastapi import APIRouter
from typing import Dict, Any, List
import logging
import httpx
from datetime import datetime

from .schemas import FeaturePayload
from .tinybert_classifier import TinyBERTClassifier
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


# ---------------------------------------------------------------------------
# ScamDetectionModel
# ---------------------------------------------------------------------------
class ScamDetectionModel:
    """
    Unified scam detection model.

    Scoring strategy:
      1. TinyBERT semantic score  (on raw transcript text)
      2. Rule-based score         (on hand-crafted features — real-time fallback)
      Final score = max(tinybert_score, rule_based_score)
      This keeps the rule-based floor as a safety net.
    """

    def __init__(self):
        # Feature order kept for reference / future use
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

        # --- TinyBERT (replaces RandomForestClassifier) ---
        logger.info("Initialising TinyBERT classifier...")
        self._tinybert  = TinyBERTClassifier()
        self.is_trained = self._tinybert.is_ready   # mirrors old RF API

        if self._tinybert.is_ready:
            logger.info(
                f"✓ TinyBERT ready in '{self._tinybert._mode}' mode "
                "(replaces RandomForestClassifier)."
            )
        else:
            logger.warning(
                "TinyBERT unavailable. Falling back to rule-based scoring. "
                "Install 'torch' and 'transformers' for full functionality."
            )

    # ------------------------------------------------------------------
    # Scoring methods
    # ------------------------------------------------------------------
    def rule_based_score(self, features: Dict[str, Any]) -> float:
        """
        Real-time rule-based scorer — identical to the original implementation.
        Used as:
          • Primary scorer when TinyBERT is unavailable
          • Safety floor when TinyBERT is active
        """
        score = 0.0
        score += min(features.get("authority_claims",        0) * 0.18, 0.36)
        score += min(features.get("urgency_language",        0) * 0.20, 0.40)
        score += min(features.get("threat_language",         0) * 0.22, 0.44)
        score += min(features.get("bait_language",           0) * 0.12, 0.24)
        score += min(features.get("sensitive_info_requests", 0) * 0.25, 0.50)

        compound = features.get("compound_sentiment", 0.0)
        negative = features.get("negative_sentiment", 0.0)
        if compound < -0.4:
            score += 0.08
        if negative > 0.4:
            score += 0.08

        return min(score, 1.0)

    def model_score(self, features: Dict[str, Any], text: str = "") -> float:
        """
        TinyBERT semantic score.
        Falls back to rule_based_score when TinyBERT is unavailable or text
        is empty (preserves identical behaviour to the old RF fallback path).
        """
        if self._tinybert.is_ready and text.strip():
            result = self._tinybert.predict(text)
            return result["fraud_probability"]
        # Graceful fallback — behaviour identical to original unfitted RF
        return self.rule_based_score(features)

    # ------------------------------------------------------------------
    # Helpers (unchanged from original)
    # ------------------------------------------------------------------
    def explain_reasons(self, features: Dict[str, Any]) -> List[str]:
        reasons = []
        if features.get("authority_claims",        0) > 0:
            reasons.append("Authority indicators detected")
        if features.get("urgency_language",        0) > 0:
            reasons.append("Urgency patterns identified")
        if features.get("threat_language",         0) > 0:
            reasons.append("Threat language detected")
        if features.get("sensitive_info_requests", 0) > 0:
            reasons.append("Request for sensitive information")
        if features.get("bait_language",           0) > 0:
            reasons.append("Prize or reward bait language detected")
        return reasons

    def label_from_score(self, score: float) -> str:
        if score >= CRITICAL_THRESHOLD:
            return "critical"
        if score >= WARNING_THRESHOLD:
            return "warning"
        return "safe"

    def tinybert_status(self) -> dict:
        return self._tinybert.status()


# ---------------------------------------------------------------------------
# Module-level singleton (mirrors original pattern)
# ---------------------------------------------------------------------------
analysis_model = ScamDetectionModel()


# ---------------------------------------------------------------------------
# Alert helper (unchanged from original)
# ---------------------------------------------------------------------------
async def send_alert_if_needed(
    call_id:     str,
    user_id:     str,
    final_label: str,
    final_score: float,
    reasons:     List[str],
    features:    Dict[str, Any],
):
    if final_label == "safe":
        return

    payload = {
        "call_id":   call_id,
        "user_id":   user_id,
        "level":     final_label,
        "reason":    "; ".join(reasons) if reasons else "Suspicious call indicators detected",
        "confidence": final_score,
        "features":  features,
        "timestamp": datetime.utcnow().isoformat(),
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(ALERT_SERVICE_URL, json=payload)
    except Exception as exc:
        logger.exception("Failed to send alert for call_id=%s: %s", call_id, exc)


# ---------------------------------------------------------------------------
# Endpoint  /api/analysis/predict
# Response shape is IDENTICAL to the original — no frontend changes required.
# New fields added (tinybert_score, tinybert_mode) are additive / non-breaking.
# ---------------------------------------------------------------------------
@analysis_router.post("/predict")
async def predict_scam(payload: FeaturePayload):
    """
    Analyse extracted features + raw transcript text for scam probability.

    The FeaturePayload already contains a `text` field (the transcript), which
    TinyBERT uses directly.  Feature dict is still used by the rule-based layer.
    """
    realtime_score = analysis_model.rule_based_score(payload.features)
    # Pass the transcript text so TinyBERT can score it semantically
    bert_score     = analysis_model.model_score(payload.features, text=payload.text)

    # Keep the same "take the max" combination as the original implementation
    final_score  = max(realtime_score, bert_score)
    final_label  = analysis_model.label_from_score(final_score)
    reasons      = analysis_model.explain_reasons(payload.features)

    await send_alert_if_needed(
        call_id     = payload.call_id,
        user_id     = payload.user_id,
        final_label = final_label,
        final_score = final_score,
        reasons     = reasons,
        features    = payload.features,
    )

    tinybert_info = analysis_model.tinybert_status()

    return {
        # --- Original fields (unchanged) ---
        "status":        "predicted",
        "call_id":       payload.call_id,
        "realtime_score": round(realtime_score, 4),
        "model_score":   round(bert_score,      4),
        "final_score":   round(final_score,     4),
        "final_label":   final_label,
        "is_scam":       final_label in {"warning", "critical"},
        "confidence":    round(final_score, 4),
        "reasons":       reasons,
        "features":      payload.features,
        # --- New additive fields (non-breaking) ---
        "tinybert_score": round(bert_score, 4),
        "tinybert_mode":  tinybert_info.get("mode", "unavailable"),
    }


# ---------------------------------------------------------------------------
# Model status endpoint  /api/analysis/model/status
# ---------------------------------------------------------------------------
@analysis_router.get("/model/status")
async def get_model_status():
    """Return current model status including TinyBERT availability."""
    tinybert_info = analysis_model.tinybert_status()
    return {
        "model_type":    "TinyBERT",
        "is_trained":    analysis_model.is_trained,
        "tinybert":      tinybert_info,
        "thresholds": {
            "safe":     SAFE_THRESHOLD,
            "warning":  WARNING_THRESHOLD,
            "critical": CRITICAL_THRESHOLD,
        },
    }
