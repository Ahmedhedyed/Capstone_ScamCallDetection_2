"""
TinyBERT Scam Detection Classifier
====================================
Replaces the Random Forest / rule-based MasterModel with a lightweight
transformer-based classifier suitable for both server and mobile deployment.

Model loading hierarchy (highest to lowest priority):
  1. Fine-tuned TinyBERT loaded from LOCAL_MODEL_PATH   → best accuracy
  2. Base TinyBERT with prototype-embedding zero-shot    → good accuracy, no training needed
  3. is_ready = False  →  caller falls back to rule-based scoring

Mobile export:
  Run  python export_tinybert_tflite.py  after fine-tuning to produce a
  .tflite file for on-device inference.
"""

import os
import logging
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TINYBERT_BASE_MODEL = "huawei-noah/TinyBERT_General_4L_312D"

# After fine-tuning, save the model here and it will be auto-loaded.
LOCAL_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "tinybert_scam_classifier"
)

# ---------------------------------------------------------------------------
# Prototype sentences for zero-shot scoring (used when no fine-tuned model)
# These represent the core linguistic patterns of scam and legitimate calls.
# ---------------------------------------------------------------------------
SCAM_PROTOTYPES = [
    "Please provide your OTP password immediately or your account will be suspended.",
    "This is the IRS. You owe back taxes. Pay now to avoid an arrest warrant.",
    "Transfer money right now to avoid prosecution. Give me your CVV and PIN.",
    "Congratulations, you have won a prize. Send your credit card number to claim it.",
    "Your bank account has been hacked. Verify your PIN immediately to secure it.",
    "This is Microsoft security department. We detected a virus. Allow remote access now.",
    "Final notice: pay the fine immediately or face criminal charges and jail.",
    "You must buy gift cards right now to clear your outstanding government debt.",
    "Do not tell anyone about this call. Your social security number has been compromised.",
    "We will issue an arrest warrant if you do not pay within the next 30 minutes.",
]

LEGIT_PROTOTYPES = [
    "Hello, I am calling to confirm your appointment scheduled for tomorrow.",
    "Hi, this is a reminder that your package will be delivered today between 2 and 4 PM.",
    "I wanted to follow up on the proposal we discussed in our last meeting.",
    "Can we reschedule our meeting to Thursday afternoon? I have a conflict on Wednesday.",
    "Thank you for your order. Your confirmation number is 12345.",
    "I am calling from customer service regarding your recent support request.",
    "Just checking in to see how you are doing and whether you need any assistance.",
    "Your prescription is ready for pickup at the pharmacy.",
    "This is a courtesy call to remind you about your upcoming dental appointment.",
    "We wanted to let you know your car service is complete and ready for collection.",
]


# ---------------------------------------------------------------------------
# Main classifier class
# ---------------------------------------------------------------------------
class TinyBERTClassifier:
    """
    Lightweight BERT-based scam call classifier built on TinyBERT.

    Usage:
        clf = TinyBERTClassifier()
        if clf.is_ready:
            result = clf.predict("Your account has been hacked. Give me your OTP now.")
            # result = {"fraud_probability": 0.92, "label": "scam",
            #           "confidence": "high", "mode": "finetuned"}
    """

    def __init__(self):
        self.tokenizer    = None
        self.model        = None
        self._base_model  = None
        self._torch       = None
        self.is_ready     = False
        self._mode        = "unavailable"   # "finetuned" | "prototype" | "unavailable"
        self._scam_proto  = None
        self._legit_proto = None
        self._load_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _load_model(self):
        """
        Try to load the model.  Tries fine-tuned first, then base+prototypes,
        then logs a warning and leaves is_ready=False.
        """
        # Check for torch / transformers
        try:
            import torch
            self._torch = torch
        except ImportError:
            logger.warning(
                "⚠  'torch' is not installed. "
                "TinyBERT is unavailable — falling back to rule-based scoring."
            )
            return

        try:
            from transformers import AutoTokenizer  # noqa: just check import
        except ImportError:
            logger.warning(
                "⚠  'transformers' is not installed. "
                "TinyBERT is unavailable — falling back to rule-based scoring."
            )
            return

        # --- Option 1: fine-tuned classification model ---
        if os.path.isdir(LOCAL_MODEL_PATH):
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                self.tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
                self.model = AutoModelForSequenceClassification.from_pretrained(LOCAL_MODEL_PATH)
                self.model.eval()
                self._mode    = "finetuned"
                self.is_ready = True
                logger.info(f"✓ TinyBERT: loaded fine-tuned model from '{LOCAL_MODEL_PATH}'")
                return
            except Exception as exc:
                logger.warning(
                    f"Fine-tuned TinyBERT failed to load ({exc}). "
                    "Falling back to base model with prototype embeddings."
                )

        # --- Option 2: base TinyBERT + prototype zero-shot ---
        try:
            from transformers import AutoTokenizer, AutoModel
            logger.info(f"Loading base TinyBERT: '{TINYBERT_BASE_MODEL}' ...")
            self.tokenizer   = AutoTokenizer.from_pretrained(TINYBERT_BASE_MODEL)
            self._base_model = AutoModel.from_pretrained(TINYBERT_BASE_MODEL)
            self._base_model.eval()
            logger.info("✓ TinyBERT base model loaded. Building prototype embeddings...")
            self._build_prototypes()
            self._mode    = "prototype"
            self.is_ready = True
            logger.info("✓ TinyBERT prototype-based zero-shot scorer is ready.")
        except Exception as exc:
            logger.warning(
                f"Could not load base TinyBERT ({exc}). "
                "TinyBERT is fully unavailable — using rule-based fallback."
            )

    # ------------------------------------------------------------------
    # Embedding helpers (used in prototype mode)
    # ------------------------------------------------------------------
    def _encode(self, text: str) -> np.ndarray:
        """Return a mean-pooled sentence embedding for `text`."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True,
        )
        with self._torch.no_grad():
            outputs = self._base_model(**inputs)
        hidden = outputs.last_hidden_state           # (1, seq_len, hidden_dim)
        mask   = inputs["attention_mask"].unsqueeze(-1).float()
        emb    = (hidden * mask).sum(1) / mask.sum(1)
        return emb.squeeze(0).cpu().numpy()

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def _build_prototypes(self):
        """Pre-compute average embeddings for the scam and legitimate prototypes."""
        self._scam_proto  = np.mean([self._encode(t) for t in SCAM_PROTOTYPES],  axis=0)
        self._legit_proto = np.mean([self._encode(t) for t in LEGIT_PROTOTYPES], axis=0)
        logger.info("✓ Prototype embeddings built.")

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, text: str) -> dict:
        """
        Predict the fraud probability of a call transcript.

        Returns:
            dict with keys:
              - fraud_probability (float 0–1)
              - label             ("scam" | "legitimate" | "unknown")
              - confidence        ("high" | "medium" | "low")
              - mode              ("finetuned" | "prototype" | "unavailable")
        """
        if not self.is_ready or not text or not text.strip():
            return {
                "fraud_probability": 0.0,
                "label":             "unknown",
                "confidence":        "low",
                "mode":              "unavailable",
            }

        if self._mode == "finetuned":
            return self._predict_finetuned(text)
        return self._predict_prototype(text)

    def _predict_finetuned(self, text: str) -> dict:
        """Use the fine-tuned sequence classification head."""
        import torch.nn.functional as F
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True,
        )
        with self._torch.no_grad():
            logits = self.model(**inputs).logits
        probs      = F.softmax(logits, dim=-1).squeeze()
        # Convention: label index 1 = scam (matches standard binary classification)
        fraud_prob = float(probs[1]) if probs.dim() > 0 and probs.shape[0] > 1 else float(probs)
        return {
            "fraud_probability": round(fraud_prob, 4),
            "label":             "scam" if fraud_prob >= 0.5 else "legitimate",
            "confidence":        self._confidence_label(fraud_prob),
            "mode":              "finetuned",
        }

    def _predict_prototype(self, text: str) -> dict:
        """Zero-shot scoring: cosine similarity to scam vs. legitimate prototypes."""
        emb       = self._encode(text)
        scam_sim  = max(0.0, self._cosine(emb, self._scam_proto))
        legit_sim = max(0.0, self._cosine(emb, self._legit_proto))
        total     = scam_sim + legit_sim + 1e-8
        fraud_prob = scam_sim / total
        return {
            "fraud_probability": round(float(fraud_prob), 4),
            "label":             "scam" if fraud_prob >= 0.5 else "legitimate",
            "confidence":        self._confidence_label(fraud_prob),
            "mode":              "prototype",
        }

    @staticmethod
    def _confidence_label(p: float) -> str:
        if p > 0.80 or p < 0.20:
            return "high"
        if p > 0.65 or p < 0.35:
            return "medium"
        return "low"

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------
    def status(self) -> dict:
        return {
            "is_ready":    self.is_ready,
            "mode":        self._mode,
            "model_path":  LOCAL_MODEL_PATH if self._mode == "finetuned" else TINYBERT_BASE_MODEL,
        }
