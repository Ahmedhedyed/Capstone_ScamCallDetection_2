"""
Twilio Integration Service
Handles CPaaS functionality for making and receiving calls
"""

from fastapi import APIRouter, HTTPException
from twilio.rest import Client
from twilio.twiml import VoiceResponse
import logging
import os
from typing import Dict, Optional
import asyncio
import httpx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

twilio_router = APIRouter()

class TwilioService:
    def __init__(self):
        self.account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        self.auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        self.phone_number = os.getenv("TWILIO_PHONE_NUMBER")
        
        if not all([self.account_sid, self.auth_token, self.phone_number]):
            logger.warning("Twilio credentials not configured")
            self.client = None
        else:
            self.client = Client(self.account_sid, self.auth_token)
        
        self.audio_service_url = os.getenv("AUDIO_SERVICE_URL", "http://localhost:8000/api/audio")
    
    async def make_call(self, to_number: str, from_number: str, user_id: str) -> Dict:
        """Initiate an outbound call"""
        if not self.client:
            raise HTTPException(status_code=500, detail="Twilio not configured")
        
        try:
            # Create call with streaming enabled
            call = self.client.calls.create(
                to=to_number,
                from_=from_number,
                url=f"{self.audio_service_url}/twilio/voice",
                method='POST',
                status_callback=f"{self.audio_service_url}/twilio/status",
                status_callback_event=['initiated', 'ringing', 'answered', 'completed'],
                status_callback_method='POST'
            )
            
            logger.info(f"Call initiated: {call.sid}")
            
            return {
                "call_sid": call.sid,
                "status": call.status,
                "to": to_number,
                "from": from_number,
                "user_id": user_id
            }
            
        except Exception as e:
            logger.error(f"Error making call: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def handle_incoming_call(self, call_sid: str, from_number: str, to_number: str) -> str:
        """Handle incoming call and return TwiML response"""
        try:
            # Create TwiML response
            response = VoiceResponse()
            
            # Say greeting
            response.say("Hello, this is CallGuard Sentinel. Your call is being monitored for security.")
            
            # Start streaming to our audio service
            response.start().stream(
                url=f"wss://{self.audio_service_url}/stream/{call_sid}",
                track='both_tracks'
            )
            
            # Keep the call active
            response.pause(length=3600)  # 1 hour max
            
            return str(response)
            
        except Exception as e:
            logger.error(f"Error handling incoming call: {e}")
            # Return error TwiML
            response = VoiceResponse()
            response.say("Sorry, there was an error processing your call.")
            response.hangup()
            return str(response)
    
    async def handle_call_status(self, call_sid: str, status: str, user_id: str):
        """Handle call status updates"""
        try:
            # Notify audio service of status change
            payload = {
                "call_sid": call_sid,
                "status": status,
                "user_id": user_id
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.audio_service_url}/twilio/status-update",
                    json=payload,
                    timeout=5.0
                )
                
                if response.status_code != 200:
                    logger.error(f"Error notifying audio service: {response.text}")
            
            logger.info(f"Call {call_sid} status updated to {status}")
            
        except Exception as e:
            logger.error(f"Error handling call status: {e}")
    
    async def end_call(self, call_sid: str) -> bool:
        """End an active call"""
        if not self.client:
            return False
        
        try:
            call = self.client.calls(call_sid).update(status='completed')
            logger.info(f"Call {call_sid} ended")
            return True
            
        except Exception as e:
            logger.error(f"Error ending call {call_sid}: {e}")
            return False
    
    def get_call_info(self, call_sid: str) -> Optional[Dict]:
        """Get information about a call"""
        if not self.client:
            return None
        
        try:
            call = self.client.calls(call_sid).fetch()
            return {
                "sid": call.sid,
                "status": call.status,
                "direction": call.direction,
                "from": call.from_,
                "to": call.to,
                "start_time": call.start_time,
                "end_time": call.end_time,
                "duration": call.duration
            }
            
        except Exception as e:
            logger.error(f"Error getting call info: {e}")
            return None

# Initialize service
twilio_service = TwilioService()

@twilio_router.post("/make-call")
async def make_call_endpoint(
    to_number: str,
    from_number: str,
    user_id: str
):
    """Make an outbound call"""
    result = await twilio_service.make_call(to_number, from_number, user_id)
    return result

@twilio_router.post("/twilio/voice")
async def handle_voice_webhook(
    call_sid: str,
    from_number: str,
    to_number: str
):
    """Twilio webhook for incoming calls"""
    twiml = await twilio_service.handle_incoming_call(call_sid, from_number, to_number)
    return {"twiml": twiml}

@twilio_router.post("/twilio/status")
async def handle_status_webhook(
    call_sid: str,
    call_status: str,
    user_id: str = None
):
    """Twilio webhook for call status updates"""
    await twilio_service.handle_call_status(call_sid, call_status, user_id)
    return {"status": "received"}

@twilio_router.post("/end-call/{call_sid}")
async def end_call_endpoint(call_sid: str):
    """End a call"""
    success = await twilio_service.end_call(call_sid)
    return {"success": success}

@twilio_router.get("/call/{call_sid}")
async def get_call_info_endpoint(call_sid: str):
    """Get call information"""
    info = twilio_service.get_call_info(call_sid)
    if info is None:
        raise HTTPException(status_code=404, detail="Call not found")
    return info

@twilio_router.get("/health")
async def health_check():
    """Health check for Twilio service"""
    return {
        "status": "healthy",
        "twilio_configured": twilio_service.client is not None
    }
