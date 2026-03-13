import spacy
import re
from collections import Counter

class TextFeatureExtractor:
    """
    An advanced feature extractor that identifies linguistic tactics used in fraud.
    It focuses on high-risk signals and reduces noise from common conversation
    for superior accuracy.
    """
    def __init__(self):
        """
        Initializes the extractor with enhanced lexicons and word lists
        to detect nuanced fraud patterns.
        """
        self.nlp = spacy.load("en_core_web_sm")
        
        # --- Expanded and Refined Lexicons ---
        self.authority_lexicon = {"irs agent": 1.0, "federal officer": 1.0, "social security administration": 1.0, "security department": 0.8, "bank security": 0.8, "official": 0.6, "badge number": 0.9, "case number": 0.9, "microsoft": 0.8, "amazon": 0.7}
        self.urgency_lexicon = {"right now": 1.0, "immediately": 1.0, "within minutes": 0.9, "act now": 0.9, "final notice": 1.0, "last chance": 1.0, "account will be closed": 0.9, "expires": 0.8}
        self.threat_lexicon = {"arrest warrant": 1.0, "prosecution": 1.0, "criminal charges": 1.0, "lawsuit": 0.9, "legal action": 0.9, "police": 0.8, "account suspended": 0.9, "locked": 0.8}
        self.scam_lexicon = {"virus": 0.8, "infected": 0.8, "hacked": 0.8, "remote access": 1.0, "unusual transaction": 0.9, "fraudulent activity": 0.9, "gift card": 1.0, "processing fee": 1.0}
        
        # PII Lexicon: Critical credentials have max weight. Common PII is weighted lower.
        self.pii_lexicon = {
            # Critical Credentials
            "password": 1.0, "pin": 1.0, "security code": 1.0, "cvv": 1.0, "otp": 1.0, "one time password": 1.0,
            # Sensitive Identifiers
            "social security": 1.0, "ssn": 1.0, "mother's maiden name": 0.9, "date of birth": 0.7,
            # Commonly requested (but less critical) PII
            "address": 0.2, "email": 0.2
        }

        # --- NEW: Lexicons for Linguistic Tactics ---
        self.evasive_lexicon = {
            "don't worry about that": 0.9, "that's not important": 0.9, "just listen to me": 0.8,
            "i am not authorized to": 0.7, "let's focus on": 0.6, "you don't need to know": 1.0
        }
        self.reassurance_lexicon = {
            "this is a secure line": 0.8, "rest assured": 0.7, "i'm here to help": 0.5,
            "trust me": 0.9, "this is legitimate": 0.8
        }

        # --- NEW: Sets for Filtering Noise ---
        # Only score commands that are inherently risky.
        self.high_risk_commands = {'install', 'download', 'transfer', 'buy', 'send', 'verify', 'provide', 'give', 'allow', 'confirm', 'share'}
        # Ignore common conversational words in repetition analysis.
        self.filler_words = {'yes', 'okay', 'no', 'sir', 'maam', 'hello', 'right', 'please', 'thank', 'you', 'i', 'the', 'a'}

        # --- Regex (Unchanged) ---
        self.pii_regex = {"SSN": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'), "CREDIT_CARD": re.compile(r'\b(?:\d[ -]*?){13,16}\b')}

    def _score_from_lexicon(self, text, lexicon):
        """Generic function to score text based on a weighted lexicon."""
        score = 0.0
        evidence = [phrase for phrase in lexicon if phrase in text]
        if evidence:
            score = sum(lexicon[phrase] for phrase in evidence)
        return min(1.0, score), evidence

    def _analyze_action_demands(self, doc):
        """
        IMPROVED: Analyzes text for HIGH-RISK imperative commands, ignoring benign
        ones like 'tell' or 'let' for higher accuracy.
        """
        commands = []
        for token in doc:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                has_subject = any(child.dep_.startswith("nsubj") for child in token.children)
                # Only flag the command if it's on our high-risk list.
                if not has_subject and token.lemma_ in self.high_risk_commands:
                    commands.append(token.lemma_)
        
        score = min(1.0, len(commands) * 0.3) # Each risky command contributes significantly
        return score, commands

    def _format_output(self, score, evidence):
        """Standardizes the output format for each feature."""
        return {"score": round(score, 2), "confidence": round(score, 2), "evidence": evidence}

    def extract_features(self, text):
        """
        Main method to orchestrate the extraction of all features, now including
        linguistic tactics and noise reduction.
        """
        text = text.lower()
        doc = self.nlp(text)

        # --- 1. Lexical and Tactical Analysis ---
        authority = self._score_from_lexicon(text, self.authority_lexicon)
        urgency = self._score_from_lexicon(text, self.urgency_lexicon)
        threats = self._score_from_lexicon(text, self.threat_lexicon)
        scam_lexicon = self._score_from_lexicon(text, self.scam_lexicon)
        pii_requests = self._score_from_lexicon(text, self.pii_lexicon) # Simplified PII for clarity, regex can be added back if needed
        
        # NEW Tactical Features
        evasiveness = self._score_from_lexicon(text, self.evasive_lexicon)
        reassurance = self._score_from_lexicon(text, self.reassurance_lexicon)

        # --- 2. Syntactic Analysis (Noise Reduced) ---
        action_demands = self._analyze_action_demands(doc)
        
        # --- 3. Repetition Analysis (Noise Reduced) ---
        words = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop and token.lemma_ not in self.filler_words]
        word_counts = Counter(words)
        repeated_words = [word for word, count in word_counts.items() if count >= 3]
        repetition_score = min(1.0, len(repeated_words) * 0.2)
        
        # --- 4. Assemble Final Features ---
        features = {
            "authority": self._format_output(authority[0], authority[1]),
            "urgency": self._format_output(urgency[0], urgency[1]),
            "threats": self._format_output(threats[0], threats[1]),
            "pii_requests": self._format_output(pii_requests[0], pii_requests[1]),
            "scam_lexicon": self._format_output(scam_lexicon[0], scam_lexicon[1]),
            "action_demands": self._format_output(action_demands[0], action_demands[1]),
            "repetition": self._format_output(repetition_score, repeated_words),
            "evasiveness": self._format_output(evasiveness[0], evasiveness[1]),
            "false_reassurance": self._format_output(reassurance[0], reassurance[1])
        }
        
        return features