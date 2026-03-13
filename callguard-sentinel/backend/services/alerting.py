"""
Alerting Service
Handles scam alerts and broadcasts them to connected dashboards.
"""

from datetime import datetime
import logging
from typing import Dict, List

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

alerting_router = APIRouter()

dashboard_connections: List[WebSocket] = []


class AlertPayload(BaseModel):
    call_id: str
    user_id: str
    level: str
    reason: str
    confidence: float
    features: Dict
    timestamp: str


class BroadcastPayload(BaseModel):
    type: str
    call_id: str
    user_id: str
    score: float
    level: str
    reason: str
    timestamp: str


async def broadcast_to_dashboards(message: Dict):
    dead = []
    for ws in dashboard_connections:
        try:
            await ws.send_json(message)
        except Exception:
            dead.append(ws)

    for ws in dead:
        if ws in dashboard_connections:
            dashboard_connections.remove(ws)


@alerting_router.websocket("/dashboard")
async def dashboard_socket(websocket: WebSocket):
    await websocket.accept()
    dashboard_connections.append(websocket)
    logger.info("Dashboard connected")

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        logger.info("Dashboard disconnected")
        if websocket in dashboard_connections:
            dashboard_connections.remove(websocket)


@alerting_router.post("/send")
async def send_alert(payload: AlertPayload):
    logger.warning(
        "SCAM ALERT: %s level=%s confidence=%.2f",
        payload.call_id,
        payload.level,
        payload.confidence,
    )

    message = {
        "type": "scam_alert",
        "call_id": payload.call_id,
        "user_id": payload.user_id,
        "level": payload.level,
        "reason": payload.reason,
        "confidence": payload.confidence,
        "features": payload.features,
        "timestamp": payload.timestamp or datetime.utcnow().isoformat(),
    }

    await broadcast_to_dashboards(message)
    return {"status": "alert_sent"}


@alerting_router.post("/broadcast")
async def broadcast(payload: BroadcastPayload):
    message = {
        "type": payload.type,
        "call_id": payload.call_id,
        "user_id": payload.user_id,
        "score": payload.score,
        "level": payload.level,
        "reason": payload.reason,
        "timestamp": payload.timestamp or datetime.utcnow().isoformat(),
    }

    await broadcast_to_dashboards(message)
    return {"status": "broadcast_sent"}


# """
# Alerting Service
# Manages WebSocket connections and sends real-time alerts to users
# """

# from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
# from fastapi.responses import JSONResponse
# import asyncio
# import json
# import logging
# from datetime import datetime
# from typing import Dict, List, Optional
# import uuid

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# alerting_router = APIRouter()

# class AlertingService:
#     def __init__(self):
#         # Store active WebSocket connections by user_id
#         self.active_connections: Dict[str, WebSocket] = {}
#         # Store user sessions
#         self.user_sessions: Dict[str, Dict] = {}
#         # Store pending alerts
#         self.pending_alerts: Dict[str, List[Dict]] = {}
    
#     async def connect_user(self, websocket: WebSocket, user_id: str):
#         """Handle new user WebSocket connection"""
#         await websocket.accept()
#         self.active_connections[user_id] = websocket
        
#         # Initialize user session
#         self.user_sessions[user_id] = {
#             "connected_at": datetime.utcnow(),
#             "is_active": True,
#             "call_count": 0
#         }
        
#         # Send any pending alerts
#         if user_id in self.pending_alerts:
#             for alert in self.pending_alerts[user_id]:
#                 await self.send_alert_to_user(user_id, alert)
#             self.pending_alerts[user_id] = []
        
#         logger.info(f"User {user_id} connected to alerting service")
        
#         try:
#             while True:
#                 # Keep connection alive and handle any incoming messages
#                 data = await websocket.receive_text()
#                 message = json.loads(data)
#                 await self.handle_user_message(user_id, message)
                
#         except WebSocketDisconnect:
#             logger.info(f"User {user_id} disconnected from alerting service")
#             await self.disconnect_user(user_id)
    
#     async def disconnect_user(self, user_id: str):
#         """Handle user disconnection"""
#         if user_id in self.active_connections:
#             del self.active_connections[user_id]
        
#         if user_id in self.user_sessions:
#             self.user_sessions[user_id]["is_active"] = False
    
#     async def handle_user_message(self, user_id: str, message: Dict):
#         """Handle incoming messages from user"""
#         message_type = message.get("type")
        
#         if message_type == "ping":
#             await self.send_to_user(user_id, {"type": "pong", "timestamp": datetime.utcnow().isoformat()})
#         elif message_type == "alert_dismissed":
#             alert_id = message.get("alert_id")
#             logger.info(f"User {user_id} dismissed alert {alert_id}")
#         elif message_type == "call_started":
#             call_id = message.get("call_id")
#             await self.handle_call_started(user_id, call_id)
#         elif message_type == "call_ended":
#             call_id = message.get("call_id")
#             await self.handle_call_ended(user_id, call_id)
    
#     async def handle_call_started(self, user_id: str, call_id: str):
#         """Handle call started event"""
#         if user_id in self.user_sessions:
#             self.user_sessions[user_id]["call_count"] += 1
#             self.user_sessions[user_id]["current_call"] = call_id
        
#         logger.info(f"Call {call_id} started for user {user_id}")
    
#     async def handle_call_ended(self, user_id: str, call_id: str):
#         """Handle call ended event"""
#         if user_id in self.user_sessions:
#             self.user_sessions[user_id]["current_call"] = None
        
#         logger.info(f"Call {call_id} ended for user {user_id}")
    
#     async def send_alert(self, call_id: str, user_id: str, level: str, reason: str, confidence: float, features: Dict):
#         """Send scam alert to user"""
#         alert_id = str(uuid.uuid4())
        
#         alert_data = {
#             "id": alert_id,
#             "call_id": call_id,
#             "level": level,
#             "reason": reason,
#             "confidence": confidence,
#             "features": features,
#             "timestamp": datetime.utcnow().isoformat()
#         }
        
#         # Try to send immediately if user is connected
#         if user_id in self.active_connections:
#             await self.send_alert_to_user(user_id, alert_data)
#         else:
#             # Store as pending alert
#             if user_id not in self.pending_alerts:
#                 self.pending_alerts[user_id] = []
#             self.pending_alerts[user_id].append(alert_data)
#             logger.info(f"Stored pending alert for user {user_id}")
    
#     async def send_alert_to_user(self, user_id: str, alert_data: Dict):
#         """Send alert directly to user's WebSocket"""
#         try:
#             if user_id in self.active_connections:
#                 websocket = self.active_connections[user_id]
#                 await websocket.send_text(json.dumps({
#                     "type": "scam_alert",
#                     "data": alert_data
#                 }))
#                 logger.info(f"Alert sent to user {user_id}")
#             else:
#                 logger.warning(f"User {user_id} not connected, cannot send alert")
#         except Exception as e:
#             logger.error(f"Error sending alert to user {user_id}: {e}")
    
#     async def send_to_user(self, user_id: str, message: Dict):
#         """Send generic message to user"""
#         try:
#             if user_id in self.active_connections:
#                 websocket = self.active_connections[user_id]
#                 await websocket.send_text(json.dumps(message))
#         except Exception as e:
#             logger.error(f"Error sending message to user {user_id}: {e}")
    
#     async def broadcast_to_all(self, message: Dict):
#         """Broadcast message to all connected users"""
#         for user_id in self.active_connections:
#             await self.send_to_user(user_id, message)
    
#     def get_connected_users(self) -> List[str]:
#         """Get list of connected user IDs"""
#         return list(self.active_connections.keys())
    
#     def get_user_status(self, user_id: str) -> Optional[Dict]:
#         """Get status of a specific user"""
#         if user_id in self.user_sessions:
#             return {
#                 **self.user_sessions[user_id],
#                 "is_connected": user_id in self.active_connections
#             }
#         return None

# # Initialize service
# alerting_service = AlertingService()

# @alerting_router.websocket("/connect/{user_id}")
# async def connect_user_endpoint(websocket: WebSocket, user_id: str):
#     """WebSocket endpoint for user connections"""
#     await alerting_service.connect_user(websocket, user_id)

# @alerting_router.post("/send")
# async def send_alert_endpoint(
#     call_id: str,
#     user_id: str,
#     level: str,
#     reason: str,
#     confidence: float,
#     features: Dict
# ):
#     """Send scam alert to user"""
#     await alerting_service.send_alert(call_id, user_id, level, reason, confidence, features)
#     return {"status": "sent"}

# @alerting_router.get("/users")
# async def get_connected_users():
#     """Get list of connected users"""
#     users = alerting_service.get_connected_users()
#     return {"connected_users": users, "count": len(users)}

# @alerting_router.get("/users/{user_id}/status")
# async def get_user_status(user_id: str):
#     """Get status of a specific user"""
#     status = alerting_service.get_user_status(user_id)
#     if status is None:
#         raise HTTPException(status_code=404, detail="User not found")
#     return status

# @alerting_router.post("/broadcast")
# async def broadcast_message(message: Dict):
#     """Broadcast message to all connected users"""
#     await alerting_service.broadcast_to_all(message)
#     return {"status": "broadcasted"}

# @alerting_router.get("/health")
# async def health_check():
#     """Health check for alerting service"""
#     return {
#         "status": "healthy",
#         "connected_users": len(alerting_service.active_connections),
#         "active_sessions": len(alerting_service.user_sessions)
#     }
