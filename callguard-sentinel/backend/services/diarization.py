"""
Speaker Diarization (2-speaker heuristic placeholder)
Replace with pyannote.audio for production diarization.
"""

from typing import List, Dict

class TwoSpeakerHeuristic:
    def __init__(self):
        # simple alternating speaker heuristic placeholder
        self.call_turns: Dict[str, int] = {}

    def label_chunk(self, call_id: str, chunk_index: int) -> str:
        turn = self.call_turns.get(call_id, 0)
        speaker = "A" if turn % 2 == 0 else "B"
        self.call_turns[call_id] = turn + 1
        return speaker

diarizer = TwoSpeakerHeuristic()


