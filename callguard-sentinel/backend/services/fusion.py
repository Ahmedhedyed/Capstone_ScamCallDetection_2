"""
Feature Synchronization and Fusion
Merges linguistic, conversational, acoustic, and context features by timestamp/speaker.
"""

from typing import Dict, Any

def merge_features(
    linguistic: Dict[str, Any],
    conversational: Dict[str, Any],
    acoustic: Dict[str, Any],
    context: Dict[str, Any],
) -> Dict[str, Any]:
    merged = {}
    merged.update(linguistic or {})
    merged.update(conversational or {})
    merged.update(acoustic or {})
    merged.update({f"ctx_{k}": v for k, v in (context or {}).items()})
    return merged


