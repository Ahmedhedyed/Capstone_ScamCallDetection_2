from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

class StartCallBody(BaseModel):
    user_id: str
    phone_number: str

class AudioChunkPayload(BaseModel):
    call_id: str
    audio_chunk: str
    timestamp: str
    user_id: str

class TranscriptionToFeaturePayload(BaseModel):
    call_id: str
    text: str
    user_id: str
    timestamp: str

class FeaturePayload(BaseModel):
    call_id: str
    user_id: str
    timestamp: str
    text: str
    features: Dict[str, Any]

class AlertPayload(BaseModel):
    call_id: str
    user_id: str
    level: str
    reason: str
    confidence: float
    features: Dict[str, Any] = Field(default_factory=dict)

class FinalDecisionPayload(BaseModel):
    call_id: str
    user_id: str
    phone_number: Optional[str] = None
    duration: int = 0
    transcript: str = ""
    realtime_score: float = 0.0
    model_score: float = 0.0
    realtime_reasons: List[str] = Field(default_factory=list)
    model_reasons: List[str] = Field(default_factory=list)