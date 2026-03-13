"""
Audio Ingestion Service
Receives live audio streams from CPaaS providers and forwards to transcription service
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse
from .schemas import StartCallBody
import asyncio
import json
import base64
import logging
from typing import Dict, List
from datetime import datetime
import uuid
from .vad import vad
from .diarization import diarizer

class SimpleVAD:
    @staticmethod
    def is_speech_base64_wav(audio_chunk: str) -> bool:
        return bool(audio_chunk and audio_chunk.strip())


class SimpleDiarizer:
    @staticmethod
    def label_chunk(call_id: str, chunk_index: int) -> str:
        return "speaker_1"


# vad = SimpleVAD()
# diarizer = SimpleDiarizer()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

audio_router = APIRouter()

# Store active WebSocket connections
active_connections: Dict[str, WebSocket] = {}
call_sessions: Dict[str, Dict] = {}

class AudioIngestionService:
    def __init__(self):
        self.transcription_service_url = "http://localhost:8001/api/transcription/process"
    
    async def connect_websocket(self, websocket: WebSocket, call_id: str, user_id: str):
        """Handle new WebSocket connection for audio stream"""
        await websocket.accept()
        active_connections[call_id] = websocket
        
        # Initialize call session
        call_sessions[call_id] = {
            "user_id": user_id,
            "start_time": datetime.utcnow(),
            "audio_chunks": [],
            "is_active": True
        }
        
        logger.info(f"Audio connection established for call {call_id}")
        
        try:
            while True:
                # Receive audio data from CPaaS
                data = await websocket.receive_text()
                audio_data = json.loads(data)
                
                # Process audio chunk
                await self.process_audio_chunk(call_id, audio_data)
                
        except WebSocketDisconnect:
            logger.info(f"Audio connection closed for call {call_id}")
            await self.disconnect_call(call_id)
    
    async def process_audio_chunk(self, call_id: str, audio_data: dict):
        """Process incoming audio chunk and forward to transcription service"""
        try:
            # Extract audio data (assuming base64 encoded mulaw)
            audio_chunk = audio_data.get("audio", "")
            timestamp = audio_data.get("timestamp", datetime.utcnow().isoformat())
            
            if not audio_chunk:
                return
            
            # VAD gate: skip non-speech chunks
            is_speech = vad.is_speech_base64_wav(audio_chunk)

            # Assign speaker label (heuristic until true diarization)
            chunk_index = len(call_sessions[call_id]["audio_chunks"]) or 0
            speaker = diarizer.label_chunk(call_id, chunk_index)

            # Store audio chunk with metadata
            call_sessions[call_id]["audio_chunks"].append({
                "data": audio_chunk,
                "timestamp": timestamp,
                "is_speech": is_speech,
                "speaker": speaker,
            })

            # Forward to transcription service only if speech
            if is_speech:
                await self.forward_to_transcription(call_id, audio_chunk, timestamp)
            
        except Exception as e:
            logger.error(f"Error processing audio chunk for call {call_id}: {e}")
    
    async def forward_to_transcription(self, call_id: str, audio_chunk: str, timestamp: str):
        """Forward audio chunk to transcription service"""
        try:
            import httpx
            
            payload = {
                "call_id": call_id,
                "audio_chunk": audio_chunk,
                "timestamp": timestamp,
                "user_id": call_sessions[call_id]["user_id"]
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.transcription_service_url,
                    json=payload,
                    timeout=5.0
                )
                
                if response.status_code != 200:
                    logger.error(f"Transcription service error: {response.text}")
                    
        except Exception as e:
            logger.error(f"Error forwarding to transcription service: {e}")
    
    async def disconnect_call(self, call_id: str):
        """Clean up call session"""
        if call_id in active_connections:
            del active_connections[call_id]
        
        if call_id in call_sessions:
            call_sessions[call_id]["is_active"] = False
            # Store call summary for analysis
            await self.store_call_summary(call_id)
    
    async def store_call_summary(self, call_id: str):
        """Store call summary for later analysis"""
        if call_id not in call_sessions:
            return
        
        session = call_sessions[call_id]
        summary = {
            "call_id": call_id,
            "user_id": session["user_id"],
            "duration": (datetime.utcnow() - session["start_time"]).total_seconds(),
            "chunk_count": len(session["audio_chunks"]),
            "is_active": session["is_active"]
        }
        
        logger.info(f"Call summary stored: {summary}")

# Initialize service
audio_service = AudioIngestionService()

@audio_router.websocket("/stream/{call_id}")
async def audio_stream_endpoint(websocket: WebSocket, call_id: str, user_id: str = ""):
    """WebSocket endpoint for receiving audio streams"""
    await audio_service.connect_websocket(websocket, call_id, user_id)

@audio_router.post("/start-call")
async def start_call(payload: StartCallBody):
    """Start a new call session"""
    call_id = str(uuid.uuid4())
    
    # Initialize call session
    call_sessions[call_id] = {
        "user_id": payload.user_id,
        "phone_number": payload.phone_number,
        "start_time": datetime.utcnow(),
        "audio_chunks": [],
        "is_active": True
    }
    
    return JSONResponse({
        "call_id": call_id,
        "websocket_url": f"ws://localhost:8000/api/audio/stream/{call_id}?user_id={payload.user_id}",
        "status": "ready"
    })

@audio_router.post("/end-call/{call_id}")
async def end_call(call_id: str):
    """End a call session"""
    if call_id in call_sessions:
        call_sessions[call_id]["is_active"] = False
        await audio_service.store_call_summary(call_id)
        
        return JSONResponse({
            "call_id": call_id,
            "status": "ended"
        })
    
    raise HTTPException(status_code=404, detail="Call not found")

@audio_router.get("/active-calls")
async def get_active_calls():
    """Get list of active calls"""
    active = {call_id: session for call_id, session in call_sessions.items() if session["is_active"]}
    return JSONResponse({"active_calls": active})

@audio_router.get("/call/{call_id}/status")
async def get_call_status(call_id: str):
    """Get status of a specific call"""
    if call_id not in call_sessions:
        raise HTTPException(status_code=404, detail="Call not found")
    
    session = call_sessions[call_id]
    return JSONResponse({
        "call_id": call_id,
        "user_id": session["user_id"],
        "phone_number": session.get("phone_number"),
        "is_active": session["is_active"],
        "duration": (datetime.utcnow() - session["start_time"]).total_seconds(),
        "chunk_count": len(session["audio_chunks"])
    })
