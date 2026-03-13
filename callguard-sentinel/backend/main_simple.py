"""
CallGuard Sentinel - Simplified Backend Service
Working demo backend with local WebSocket job updates.
"""

import asyncio
import json
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from config import API_HOST, API_PORT, CORS_ORIGINS

load_dotenv()

analysis_jobs: Dict[str, Dict] = {}
websocket_connections: Dict[str, List[WebSocket]] = {}

app = FastAPI(
    title="CallGuard Sentinel API",
    description="Real-time scam detection and call analysis system",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def connect_websocket(websocket: WebSocket, job_id: str):
    await websocket.accept()
    websocket_connections.setdefault(job_id, []).append(websocket)


async def disconnect_websocket(websocket: WebSocket, job_id: str):
    if job_id in websocket_connections and websocket in websocket_connections[job_id]:
        websocket_connections[job_id].remove(websocket)
        if not websocket_connections[job_id]:
            del websocket_connections[job_id]


async def broadcast_to_job(job_id: str, message: dict):
    if job_id not in websocket_connections:
        return

    dead = []
    for ws in websocket_connections[job_id]:
        try:
            await ws.send_text(json.dumps(message))
        except Exception:
            dead.append(ws)

    for ws in dead:
        await disconnect_websocket(ws, job_id)


async def update_job_status(job_id: str, status: str, progress: int, data: Optional[dict] = None):
    if job_id not in analysis_jobs:
        return

    analysis_jobs[job_id]["status"] = status
    analysis_jobs[job_id]["progress"] = progress
    analysis_jobs[job_id]["updated_at"] = datetime.utcnow().isoformat()

    if data is not None:
        analysis_jobs[job_id]["data"] = data

    await broadcast_to_job(
        job_id,
        {
            "job_id": job_id,
            "status": status,
            "progress": progress,
            "data": data,
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


async def analyze_call_recording(job_id: str, file_path: str):
    try:
        await update_job_status(job_id, "processing", 10, {"step": "Processing audio file"})

        if not os.path.exists(file_path):
            raise RuntimeError("Uploaded file not found")
        if os.path.getsize(file_path) == 0:
            raise RuntimeError("Uploaded file is empty")

        await asyncio.sleep(1)
        await update_job_status(job_id, "processing", 30, {"step": "Transcribing audio"})
        await asyncio.sleep(1)
        await update_job_status(job_id, "processing", 55, {"step": "Extracting scam features"})
        await asyncio.sleep(1)
        await update_job_status(job_id, "processing", 80, {"step": "Running classifier"})
        await asyncio.sleep(1)

        result = {
            "is_scam": False,
            "final_score": 0.12,
            "alert_level": "safe",
            "confidence": 0.88,
            "features": {
                "authority": 0.0,
                "urgency": 0.0,
                "threat": 0.0,
                "sensitivity": 0.0,
            },
            "transcription": "Hello, this is a normal call.",
            "analysis_summary": "No major scam indicators detected.",
        }

        await update_job_status(
            job_id,
            "completed",
            100,
            {"step": "Analysis complete", "result": result},
        )

    except Exception as exc:
        await update_job_status(job_id, "failed", 0, {"step": "Analysis failed", "error": str(exc)})
    finally:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass


@app.get("/")
async def root():
    return {"message": "CallGuard Sentinel API", "status": "running"}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "services": {
            "audio_ingestion": "running",
            "transcription": "running",
            "feature_extraction": "running",
            "analysis": "running",
            "alerting": "running",
        },
    }


@app.post("/api/audio/start-call")
async def start_call():
    return JSONResponse(
        {
            "call_id": "test-call-123",
            "websocket_url": "ws://localhost:8000/api/audio/stream/test-call-123",
            "status": "ready",
        }
    )


@app.get("/api/audio/active-calls")
async def get_active_calls():
    return JSONResponse({"active_calls": {}})


@app.post("/api/transcription/process")
async def process_transcription():
    return {"status": "processed"}


@app.post("/api/features/extract")
async def extract_features():
    return {"status": "extracted"}


@app.post("/api/analysis/predict")
async def predict_scam():
    return {"status": "predicted", "is_scam": False, "confidence": 0.10}


@app.post("/api/alerts/send")
async def send_alert():
    return {"status": "sent"}


@app.post("/analyze/fast/")
async def analyze_call_fast(file: UploadFile = File(...)):
    """Fast analysis endpoint — returns result directly without a job queue."""
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    # Simulate analysis (replace with real ML pipeline when ready)
    return {
        "is_fraud": False,
        "fraud_score": 0.08,
        "alert_level": "safe",
        "confidence": 0.92,
        "explanation": "No scam indicators detected in this call.",
        "features": {
            "authority": 0.0,
            "urgency": 0.0,
            "threat": 0.0,
            "sensitivity": 0.0,
        },
        "transcription": "This is a demo transcription of the incoming call.",
    }


@app.post("/analyze/")
async def analyze_call(file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    uploads_dir = "uploads"
    os.makedirs(uploads_dir, exist_ok=True)

    file_extension = file.filename.split(".")[-1] if "." in file.filename else "mp4"
    file_path = os.path.join(uploads_dir, f"{job_id}_call_recording.{file_extension}")

    try:
        with open(file_path, "wb") as buffer:
            content = await file.read()
            if not content:
                raise HTTPException(status_code=400, detail="Uploaded file is empty")
            buffer.write(content)

        analysis_jobs[job_id] = {
            "id": job_id,
            "status": "processing",
            "progress": 0,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "file_path": file_path,
        }

        asyncio.create_task(analyze_call_recording(job_id, file_path))
        return {"job_id": job_id, "status": "processing"}

    except Exception as exc:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Error processing file: {exc}")


@app.websocket("/ws/status/{job_id}")
async def websocket_status(websocket: WebSocket, job_id: str):
    await connect_websocket(websocket, job_id)

    try:
        if job_id in analysis_jobs:
            await websocket.send_text(json.dumps(analysis_jobs[job_id]))

        while True:
            await websocket.receive_text()

    except WebSocketDisconnect:
        await disconnect_websocket(websocket, job_id)
    except Exception:
        await disconnect_websocket(websocket, job_id)
        await websocket.close(code=1011)


@app.get("/analysis/{job_id}")
async def get_analysis_status(job_id: str):
    if job_id not in analysis_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return analysis_jobs[job_id]


@app.get("/analysis/")
async def list_analysis_jobs():
    return {"jobs": list(analysis_jobs.values())}


if __name__ == "__main__":
    uvicorn.run("main_simple:app", host=API_HOST, port=API_PORT, reload=True)


# """
# CallGuard Sentinel - Simplified Backend Service
# Basic version to get the project running
# """

# from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File, Request
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# import uvicorn
# import os
# import uuid
# import asyncio
# import json
# import tempfile
# import shutil
# import httpx
# from datetime import datetime
# from typing import Dict, List
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# # Global storage for analysis jobs and WebSocket connections
# analysis_jobs: Dict[str, Dict] = {}
# websocket_connections: Dict[str, List[WebSocket]] = {}

# app = FastAPI(
#     title="CallGuard Sentinel API",
#     description="Real-time scam detection and call analysis system",
#     version="1.0.0"
# )

# # CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Configure appropriately for production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # WebSocket connection management
# async def connect_websocket(websocket: WebSocket, job_id: str):
#     """Connect a WebSocket to a job"""
#     await websocket.accept()
#     if job_id not in websocket_connections:
#         websocket_connections[job_id] = []
#     websocket_connections[job_id].append(websocket)

# async def disconnect_websocket(websocket: WebSocket, job_id: str):
#     """Disconnect a WebSocket from a job"""
#     if job_id in websocket_connections:
#         websocket_connections[job_id].remove(websocket)
#         if not websocket_connections[job_id]:
#             del websocket_connections[job_id]

# async def broadcast_to_job(job_id: str, message: dict):
#     """Broadcast a message to all WebSocket connections for a job"""
#     if job_id in websocket_connections:
#         disconnected = []
#         for websocket in websocket_connections[job_id]:
#             try:
#                 await websocket.send_text(json.dumps(message))
#             except:
#                 disconnected.append(websocket)
        
#         # Remove disconnected websockets
#         for ws in disconnected:
#             websocket_connections[job_id].remove(ws)

# # Analysis pipeline functions
# async def update_job_status(job_id: str, status: str, progress: int = 0, data: dict = None):
#     """Update job status and broadcast to WebSocket connections"""
#     if job_id in analysis_jobs:
#         analysis_jobs[job_id]["status"] = status
#         analysis_jobs[job_id]["progress"] = progress
#         analysis_jobs[job_id]["updated_at"] = datetime.utcnow().isoformat()
        
#         if data:
#             analysis_jobs[job_id]["data"] = data
        
#         # Broadcast update
#         await broadcast_to_job(job_id, {
#             "job_id": job_id,
#             "status": status,
#             "progress": progress,
#             "data": data,
#             "timestamp": datetime.utcnow().isoformat()
#         })

# async def analyze_call_recording(job_id: str, file_path: str):
#     """Simulate the analysis pipeline with status updates"""
#     try:
#         # Step 1: File processing
#         await update_job_status(job_id, "processing", 10, {"step": "Processing audio file"})
        
#         # Check if file exists and has content
#         if not os.path.exists(file_path):
#             raise Exception("File not found")
        
#         file_size = os.path.getsize(file_path)
#         if file_size == 0:
#             raise Exception("Uploaded file is empty")
        
#         await asyncio.sleep(1)
        
#         # Step 2: Audio extraction
#         await update_job_status(job_id, "processing", 25, {"step": "Extracting audio from video"})
#         await asyncio.sleep(1)
        
#         # Step 3: Transcription
#         await update_job_status(job_id, "processing", 40, {"step": "Transcribing audio"})
#         await asyncio.sleep(2)
        
#         # Step 4: Feature extraction
#         await update_job_status(job_id, "processing", 60, {"step": "Extracting linguistic features"})
#         await asyncio.sleep(1)
        
#         # Step 5: Analysis
#         await update_job_status(job_id, "processing", 80, {"step": "Running scam detection analysis"})
#         await asyncio.sleep(1)
        
#         # Step 6: Complete
#         result = {
#             "is_scam": False,
#             "confidence": 0.15,
#             "risk_level": "low",
#             "features": {
#                 "authority_claims": 0,
#                 "urgency_language": 0,
#                 "threat_language": 0,
#                 "sensitive_info_requests": 0
#             },
#             "transcription": "Hello, this is a test call. How are you doing today?",
#             "analysis_summary": "No scam indicators detected in this call."
#         }
        
#         await update_job_status(job_id, "completed", 100, {
#             "step": "Analysis complete",
#             "result": result
#         })
        
#     except Exception as e:
#         await update_job_status(job_id, "failed", 0, {
#             "step": "Analysis failed",
#             "error": str(e)
#         })
#     finally:
#         # Clean up file
#         try:
#             if os.path.exists(file_path):
#                 os.remove(file_path)
#         except Exception as cleanup_error:
#             print(f"Error cleaning up file {file_path}: {cleanup_error}")

# @app.get("/")
# async def root():
#     return {"message": "CallGuard Sentinel API", "status": "running"}

# @app.get("/health")
# async def health_check():
#     """Health check endpoint"""
#     return {
#         "status": "healthy",
#         "services": {
#             "audio_ingestion": "running",
#             "transcription": "running", 
#             "feature_extraction": "running",
#             "analysis": "running",
#             "alerting": "running"
#         }
#     }

# @app.post("/api/audio/start-call")
# async def start_call():
#     """Start a new call session"""
#     return JSONResponse({
#         "call_id": "test-call-123",
#         "websocket_url": "ws://localhost:8000/api/audio/stream/test-call-123",
#         "status": "ready"
#     })

# @app.get("/api/audio/active-calls")
# async def get_active_calls():
#     """Get list of active calls"""
#     return JSONResponse({"active_calls": {}})

# @app.post("/api/transcription/process")
# async def process_transcription():
#     """Process transcription"""
#     return {"status": "processed"}

# @app.post("/api/features/extract")
# async def extract_features():
#     """Extract features"""
#     return {"status": "extracted"}

# @app.post("/api/analysis/predict")
# async def predict_scam():
#     """Predict scam"""
#     return {"status": "predicted", "is_scam": False, "confidence": 0.1}

# @app.post("/api/alerts/send")
# async def send_alert():
#     """Send alert"""
#     return {"status": "sent"}

# @app.post("/analyze/")
# async def analyze_call(file: UploadFile = File(None), call_data: str = None):
#     """Handle call analysis directly"""
#     if not file:
#         raise HTTPException(status_code=400, detail="No file provided")
    
#     # Generate job ID
#     job_id = str(uuid.uuid4())
    
#     # Create uploads directory if it doesn't exist
#     uploads_dir = "uploads"
#     os.makedirs(uploads_dir, exist_ok=True)
    
#     # Save uploaded file
#     file_extension = file.filename.split('.')[-1] if '.' in file.filename else 'mp4'
#     file_path = os.path.join(uploads_dir, f"{job_id}_call_recording.{file_extension}")
    
#     try:
#         # Save file to disk
#         with open(file_path, "wb") as buffer:
#             content = await file.read()
#             if len(content) == 0:
#                 raise HTTPException(status_code=400, detail="Uploaded file is empty")
#             buffer.write(content)
        
#         # Initialize job
#         analysis_jobs[job_id] = {
#             "id": job_id,
#             "status": "processing",
#             "progress": 0,
#             "created_at": datetime.utcnow().isoformat(),
#             "updated_at": datetime.utcnow().isoformat(),
#             "file_path": file_path
#         }
        
#         # Start analysis in background
#         asyncio.create_task(analyze_call_recording(job_id, file_path))
        
#         return {"job_id": job_id, "status": "processing"}
        
#     except Exception as e:
#         # Clean up file if it exists
#         if os.path.exists(file_path):
#             os.remove(file_path)
#         raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

# @app.websocket("/ws/status/{job_id}")
# async def websocket_status(websocket: WebSocket, job_id: str):
#     """Proxy WebSocket connection to analysis service"""
#     try:
#         # Connect to analysis service WebSocket
#         analysis_ws_url = f"ws://localhost:8001/ws/status/{job_id}"
        
#         async with httpx.AsyncClient() as client:
#             # For WebSocket proxying, we'll need to handle this differently
#             # For now, we'll create a simple proxy
#             pass
            
#     except Exception as e:
#         await websocket.close(code=1011, reason=f"Analysis service error: {str(e)}")

# @app.get("/analysis/{job_id}")
# async def get_analysis_status(job_id: str):
#     """Get analysis status for a specific job"""
#     if job_id not in analysis_jobs:
#         raise HTTPException(status_code=404, detail="Job not found")
    
#     return analysis_jobs[job_id]

# @app.get("/analysis/")
# async def list_analysis_jobs():
#     """List all analysis jobs"""
#     return {"jobs": list(analysis_jobs.values())}

# if __name__ == "__main__":
#     uvicorn.run(
#         "main_simple:app",
#         host="0.0.0.0",
#         port=8000,
#         reload=True
#     )
