import datetime

class MasterModel:
    """
    An advanced, hybrid master model for fraud detection. It combines a weighted
    scoring system with a rule-based engine to detect critical "knockout" fraud
    indicators and suspicious tactic combinations for superior accuracy.
    """
    
    def __init__(self):
        """Initialize the master model with calibrated weights and thresholds."""
        self.FRAUD_THRESHOLD = 0.40 # A more robust threshold for the new logic
        
        # Base weights for general suspicion level
        self.TEXT_WEIGHT = 0.7
        self.AUDIO_WEIGHT = 0.3
        
        # Individual feature weights for calculating the base score
        self.TEXT_FEATURE_WEIGHTS = {
            "authority": 0.15,
            "urgency": 0.20,
            "threats": 0.25,
            "pii_requests": 0.25,
            "scam_lexicon": 0.10,
            "action_demands": 0.05,
        }
        
        print("✓ Hybrid MasterModel initialized.")
        print(f"  - Base Fraud Threshold: {self.FRAUD_THRESHOLD}")

    def predict(self, text_features, acoustic_features):
        """
        Predicts fraud using a hybrid of weighted scoring and a rule-based engine.

        Args:
            text_features (dict): Features from the text analyzer.
            acoustic_features (dict): Features from the acoustic analyzer.

        Returns:
            dict: A comprehensive fraud analysis result.
        """
        try:
            # --- 1. Calculate Base Scores ---
            base_text_score = self._calculate_weighted_score(text_features, self.TEXT_FEATURE_WEIGHTS)
            # (Assuming a simple audio score calculation for now)
            base_audio_score = acoustic_features.get("acoustic_fraud_score", 0.0)
            
            base_combined_score = (self.TEXT_WEIGHT * base_text_score) + (self.AUDIO_WEIGHT * base_audio_score)
            
            # --- 2. Apply Rule-Based Engine for Score Adjustment ---
            final_score, rule_based_notes, triggered_rules = self._apply_fraud_rules(base_combined_score, text_features)

            # --- 3. Final Determination ---
            is_fraud = final_score >= self.FRAUD_THRESHOLD
            confidence = self._calculate_confidence(final_score)
            
            explanation, triggered_features = self._generate_final_output(
                final_score, is_fraud, rule_based_notes, text_features, base_text_score, base_audio_score
            )
            
            return {
                "fraud_score": float(final_score),
                "is_fraud": is_fraud,
                "confidence": confidence,
                "explanation": explanation,
                "triggered_features": triggered_features,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error in fraud prediction: {e}")
            return self._get_default_result()

    def _calculate_weighted_score(self, features, weights):
        """Generic function to calculate a weighted score from features."""
        if not features: return 0.0
        
        score = sum(
            weights.get(name, 0) * data.get('score', 0)
            for name, data in features.items()
        )
        return min(1.0, score)

    def _apply_fraud_rules(self, base_score, text_features):
        """
        Applies a set of rules to adjust the score based on critical indicators.
        This is the core of the hybrid logic.
        """
        adjusted_score = base_score
        notes = []
        triggered = []

        # --- Rule 1: "Knockout" PII Request ---
        # Demanding critical credentials is an almost certain sign of fraud.
        pii = text_features.get('pii_requests', {})
        critical_pii = {'otp', 'password', 'pin', 'cvv', 'security code'}
        if pii.get('score', 0) > 0.5 and any(item in pii.get('evidence', []) for item in critical_pii):
            adjusted_score = max(adjusted_score, 0.95) # Immediately escalate score
            notes.append(f"CRITICAL: Agent demanded high-risk credentials ({', '.join(pii['evidence'])}).")
            triggered.append("critical_pii_request")

        # --- Rule 2: "Knockout" Threat ---
        # The presence of a direct, severe threat is a massive red flag.
        threats = text_features.get('threats', {})
        if threats.get('score', 0) > 0.6:
            adjusted_score = max(adjusted_score, 0.85)
            notes.append(f"CRITICAL: Agent made direct threats ({', '.join(threats['evidence'])}).")
            triggered.append("direct_threat")
            
        # --- Rule 3: Suspicious Combination - Urgency + Authority ---
        # This classic pincer movement is a strong indicator of manipulation.
        urgency = text_features.get('urgency', {})
        authority = text_features.get('authority', {})
        if urgency.get('score', 0) > 0.4 and authority.get('score', 0) > 0.4:
            adjusted_score = max(adjusted_score, 0.70)
            notes.append("SUSPICIOUS TACTIC: Agent combined high urgency with a claim of authority.")
            triggered.append("urgency_plus_authority")

        # --- Rule 4: Suspicious Combination - High-Risk Command + Urgency ---
        # E.g., "You must transfer the money right now."
        actions = text_features.get('action_demands', {})
        if actions.get('score', 0) > 0.3 and urgency.get('score', 0) > 0.4:
             adjusted_score = max(adjusted_score, 0.65)
             notes.append("SUSPICIOUS TACTIC: Agent demanded a risky action under pressure.")
             triggered.append("risky_action_under_pressure")

        return min(1.0, adjusted_score), notes, triggered

    def _calculate_confidence(self, score):
        """Calculate confidence based on how decisive the score is."""
        if score > 0.8 or score < 0.2:
            return "high"
        elif score > 0.6 or score < 0.3:
            return "medium"
        else:
            return "low"

    def _generate_final_output(self, score, is_fraud, notes, text_features, text_score, audio_score):
        """Generates the final explanation and triggered features list."""
        if is_fraud:
            explanation = f"🚨 FRAUD LIKELY (Score: {score:.3f})\n\nThis call was flagged as fraudulent based on the following critical indicators:\n• " + "\n• ".join(notes)
            recommendation = "\n\n⚠️ RECOMMENDATION: End the call immediately. Do not provide any personal information."
            explanation += recommendation
        else:
            explanation = f"✅ LIKELY NORMAL CALL (Score: {score:.3f})\n\nThis call appears legitimate because it lacks the critical indicators and suspicious tactic combinations associated with fraud."

        # Identify all features that contributed to the base score
        triggered_base_features = [
            {"name": name, "score": data['score'], "evidence": data.get('evidence', [])}
            for name, data in text_features.items() if data.get('score', 0) > 0.1
        ]
        
        return explanation, {"rule_based_triggers": notes, "contributing_features": triggered_base_features}

    def _get_default_result(self):
        """Return a default error result."""
        return {
            "fraud_score": 0.0, "is_fraud": False, "confidence": "low",
            "explanation": "Error during analysis.", "triggered_features": {},
            "timestamp": datetime.datetime.now().isoformat()
        }