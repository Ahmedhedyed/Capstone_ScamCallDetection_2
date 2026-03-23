"""
MasterModel — Hybrid TinyBERT + Rule-Based Fraud Scorer
=========================================================
Architecture:
  1. TinyBERT Classifier   — semantic understanding of the full transcript
  2. Rule-Based Override    — safety net for clear-cut knockout patterns
     (critical PII demands, direct threats, urgency+authority combos)

If TinyBERT is unavailable (transformers not installed / model not yet
downloaded), the system falls back seamlessly to the legacy weighted
rule-based scoring so the API never goes down.
"""

import datetime
import logging

from .tinybert_classifier import TinyBERTClassifier

logger = logging.getLogger(__name__)


class MasterModel:
    """
    Hybrid fraud detection model.

    predict(text_features, acoustic_features, raw_text="")
      - raw_text: the full English transcription.  When provided, TinyBERT
                  scores this directly for semantic fraud detection.
      - text_features / acoustic_features: still used by the rule-based
        override layer and as a fallback when TinyBERT is unavailable.
    """

    def __init__(self):
        # --- Thresholds ---
        self.FRAUD_THRESHOLD          = 0.40
        self.LLM_ESCALATION_THRESHOLD = 0.40   # kept in sync with server.py

        # --- Fallback weighted scoring (used when TinyBERT is unavailable) ---
        self.TEXT_WEIGHT  = 0.7
        self.AUDIO_WEIGHT = 0.3

        self.TEXT_FEATURE_WEIGHTS = {
            "authority":      0.15,
            "urgency":        0.18,
            "threats":        0.23,
            "pii_requests":   0.23,
            "scam_lexicon":   0.10,
            "action_demands": 0.05,
            "repetition":     0.06,
            # Previously extracted but accidentally excluded — now included:
            "evasiveness":        0.00,   # reserved; add weight once validated
            "false_reassurance":  0.00,   # reserved; add weight once validated
        }

        # --- TinyBERT ---
        self._tinybert = TinyBERTClassifier()

        if self._tinybert.is_ready:
            logger.info(
                f"✓ MasterModel: TinyBERT active in '{self._tinybert._mode}' mode. "
                "Rule-based override layer enabled as safety net."
            )
        else:
            logger.warning(
                "MasterModel: TinyBERT unavailable. "
                "Running in legacy rule-based mode."
            )

        logger.info(f"  Fraud threshold: {self.FRAUD_THRESHOLD}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict(self, text_features: dict, acoustic_features: dict,
                raw_text: str = "") -> dict:
        """
        Run the full fraud detection pipeline.

        Args:
            text_features     (dict): output of TextFeatureExtractor
            acoustic_features (dict): acoustic features (pitch variance, etc.)
            raw_text          (str):  full English transcription for TinyBERT

        Returns:
            dict with keys: fraud_score, is_fraud, confidence, explanation,
                            triggered_features, tinybert_score, text_score,
                            audio_score, timestamp
        """
        try:
            # ---- Stage 1: primary scoring --------------------------------
            tinybert_result = None
            tinybert_score  = None

            if self._tinybert.is_ready and raw_text.strip():
                tinybert_result = self._tinybert.predict(raw_text)
                tinybert_score  = tinybert_result["fraud_probability"]

            # Fallback weighted score (always computed for the override layer)
            base_text_score  = self._calc_weighted_score(text_features, self.TEXT_FEATURE_WEIGHTS)
            base_audio_score = acoustic_features.get("acoustic_fraud_score", 0.0)
            base_combined    = (self.TEXT_WEIGHT * base_text_score) + \
                               (self.AUDIO_WEIGHT * base_audio_score)

            # ---- Stage 2: rule-based knockout check (always runs first) ----
            # Run rules on base_combined to detect knockout triggers BEFORE
            # we decide which score to use as primary.
            _, rule_notes, triggered_rules = self._apply_fraud_rules(base_combined, text_features)
            knockout_fired = len(triggered_rules) > 0

            # ---- Smart score fusion ----------------------------------------
            # When a knockout rule fires (OTP/threat/urgency+authority) the
            # rule override is authoritative — use it directly.
            # When NO knockout fires and the rule-based score is low (<0.20),
            # the call likely lacks all scam vocabulary.  In that case blend
            # TinyBERT with the rule-based score heavily toward rules to reduce
            # false positives on innocent conversational speech.
            if tinybert_score is not None:
                if knockout_fired:
                    # Hard evidence found — rules are authoritative
                    primary_score = tinybert_score   # rules will override below
                elif base_combined < 0.20:
                    # No scam vocabulary at all — dampen an uncertain TinyBERT
                    primary_score = 0.40 * tinybert_score + 0.60 * base_combined
                elif base_combined < 0.35:
                    # Weak scam vocabulary — blend evenly
                    primary_score = 0.60 * tinybert_score + 0.40 * base_combined
                else:
                    # Strong scam vocabulary — trust TinyBERT fully
                    primary_score = tinybert_score
            else:
                primary_score = base_combined

            # ---- Stage 3: rule-based override on the fused score -----------
            final_score, rule_notes, _ = self._apply_fraud_rules(primary_score, text_features)

            # ---- Stage 3: final determination ---------------------------
            is_fraud   = final_score >= self.FRAUD_THRESHOLD
            confidence = self._calc_confidence(final_score)
            explanation, triggered_features = self._build_output(
                final_score, is_fraud, rule_notes,
                text_features, base_text_score, base_audio_score
            )

            return {
                "fraud_score":        round(float(final_score), 4),
                "is_fraud":           is_fraud,
                "confidence":         confidence,
                "explanation":        explanation,
                "triggered_features": triggered_features,
                # Extra transparency fields
                "tinybert_score":     round(float(tinybert_score), 4) if tinybert_score is not None else None,
                "tinybert_mode":      tinybert_result["mode"] if tinybert_result else "unavailable",
                "text_score":         round(float(base_text_score), 4),
                "audio_score":        round(float(base_audio_score), 4),
                "threshold_used":     self.FRAUD_THRESHOLD,
                "timestamp":          datetime.datetime.now().isoformat(),
            }

        except Exception as exc:
            logger.exception(f"Error in MasterModel.predict: {exc}")
            return self._default_result()

    # ------------------------------------------------------------------
    # Rule-based override layer  (unchanged from original)
    # ------------------------------------------------------------------
    def _apply_fraud_rules(self, base_score: float, text_features: dict):
        """
        Apply knockout rules that can escalate the score regardless of the
        primary scorer.  These guard against edge cases where TinyBERT may
        be uncertain but the signal is unambiguous (e.g. OTP demand).
        """
        score    = base_score
        notes    = []
        triggered = []

        # Rule 1 — Critical PII request (OTP / CVV / PIN / password)
        pii          = text_features.get("pii_requests", {})
        critical_pii = {"otp", "password", "pin", "cvv", "security code"}
        if pii.get("score", 0) > 0.5 and \
                any(item in pii.get("evidence", []) for item in critical_pii):
            score = max(score, 0.95)
            notes.append(
                f"CRITICAL: Agent demanded high-risk credentials "
                f"({', '.join(pii['evidence'])})."
            )
            triggered.append("critical_pii_request")

        # Rule 2 — Direct threat (arrest warrant, prosecution, etc.)
        threats = text_features.get("threats", {})
        if threats.get("score", 0) > 0.6:
            score = max(score, 0.85)
            notes.append(
                f"CRITICAL: Agent made direct threats "
                f"({', '.join(threats.get('evidence', []))})."
            )
            triggered.append("direct_threat")

        # Rule 3 — Urgency + authority combination
        urgency   = text_features.get("urgency", {})
        authority = text_features.get("authority", {})
        if urgency.get("score", 0) > 0.4 and authority.get("score", 0) > 0.4:
            score = max(score, 0.70)
            notes.append("SUSPICIOUS TACTIC: High urgency combined with authority claims.")
            triggered.append("urgency_plus_authority")

        # Rule 4 — Risky action demand under urgency pressure
        actions = text_features.get("action_demands", {})
        if actions.get("score", 0) > 0.3 and urgency.get("score", 0) > 0.4:
            score = max(score, 0.65)
            notes.append("SUSPICIOUS TACTIC: Agent demanded a risky action under time pressure.")
            triggered.append("risky_action_under_pressure")

        return min(1.0, score), notes, triggered

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _calc_weighted_score(self, features: dict, weights: dict) -> float:
        if not features:
            return 0.0
        score = sum(
            weights.get(name, 0) * data.get("score", 0)
            for name, data in features.items()
        )
        return min(1.0, score)

    def _calc_confidence(self, score: float) -> str:
        if score > 0.8 or score < 0.2:
            return "high"
        if score > 0.6 or score < 0.3:
            return "medium"
        return "low"

    def _build_output(self, score, is_fraud, notes, text_features,
                      text_score, audio_score):
        if is_fraud:
            if notes:
                bullets = "\n• ".join(notes)
                explanation = (
                    f"🚨 FRAUD LIKELY (Score: {score:.3f})\n\n"
                    f"Critical indicators:\n• {bullets}\n\n"
                    "⚠️  RECOMMENDATION: End the call immediately. "
                    "Do not provide any personal information."
                )
            else:
                explanation = (
                    f"⚠️  SUSPICIOUS CALL (Score: {score:.3f})\n\n"
                    "TinyBERT semantic analysis detected patterns consistent with "
                    "scam calls. No specific rule-based triggers were found, but "
                    "the overall language pattern is suspicious.\n\n"
                    "RECOMMENDATION: Exercise caution. Do not share personal or "
                    "financial information."
                )
        else:
            explanation = (
                f"✅ LIKELY NORMAL CALL (Score: {score:.3f})\n\n"
                "This call lacks the critical indicators and suspicious tactic "
                "combinations associated with fraud."
            )

        contributing = [
            {"name": name, "score": data["score"], "evidence": data.get("evidence", [])}
            for name, data in text_features.items()
            if data.get("score", 0) > 0.1
        ]
        return explanation, {
            "rule_based_triggers":   notes,
            "contributing_features": contributing,
        }

    def _default_result(self) -> dict:
        return {
            "fraud_score":        0.0,
            "is_fraud":           False,
            "confidence":         "low",
            "explanation":        "Error during analysis.",
            "triggered_features": {},
            "tinybert_score":     None,
            "tinybert_mode":      "unavailable",
            "text_score":         0.0,
            "audio_score":        0.0,
            "threshold_used":     self.FRAUD_THRESHOLD,
            "timestamp":          datetime.datetime.now().isoformat(),
        }

    def status(self) -> dict:
        return {
            "tinybert": self._tinybert.status(),
            "fraud_threshold": self.FRAUD_THRESHOLD,
        }
