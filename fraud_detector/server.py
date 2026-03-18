import os
import uuid
import asyncio
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import aiofiles

# --- Import Your Existing Fraud Detection Components ---
from audio_ingestion.audio_ingester import AudioIngester
from analyzer.word_analyzer.transcriber import Transcriber
from analyzer.word_analyzer.text_feature_extractor import TextFeatureExtractor
from fusion_and_decision.master_model import MasterModel
from fusion_and_decision.llm_verifier import LLMVerifier

# --- 1. Initialize FastAPI and Load Models ONCE on Startup ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

UPLOADS_DIR = "uploads"
os.makedirs(UPLOADS_DIR, exist_ok=True)

print("Initializing AI models... This may take a moment.")
try:
    TRANSCRIBER = Transcriber(model_size="small")
    print("✓ Transcriber loaded.")
except Exception as e:
    print(f"✗ Failed to load Transcriber: {e}")
    TRANSCRIBER = None

try:
    TEXT_EXTRACTOR = TextFeatureExtractor()
    print("✓ TextFeatureExtractor loaded.")
except Exception as e:
    print(f"✗ Failed to load TextFeatureExtractor: {e}")
    TEXT_EXTRACTOR = None

try:
    INITIAL_MODEL = MasterModel()
    print("✓ MasterModel loaded.")
except Exception as e:
    print(f"✗ Failed to load MasterModel: {e}")
    INITIAL_MODEL = None

try:
    LLM_VERIFIER = LLMVerifier()
    print("✓ LLMVerifier loaded.")
except Exception as e:
    print(f"✗ Failed to load LLMVerifier (GPT4All server may not be running): {e}")
    LLM_VERIFIER = None

print("✓ AI model initialization complete.")


# --- Health Check Endpoint ---
@app.get("/health")
async def health_check():
    """Simple health check endpoint for the frontend to confirm the backend is live."""
    return {
        "status": "ok",
        "models": {
            "transcriber": TRANSCRIBER is not None,
            "text_extractor": TEXT_EXTRACTOR is not None,
            "master_model": INITIAL_MODEL is not None,
            "llm_verifier": LLM_VERIFIER is not None,
        }
    }


# --- 2. OPTIMIZED Core Analysis Function (for /analyze/fast/) ---
async def perform_full_analysis_optimized(file_path: str) -> dict:
    """
    Runs the entire fraud detection pipeline with a focus on maximum speed.
    Transcribes the entire file at once, skipping chunk-by-chunk logging.
    """
    # File Integrity Check
    await asyncio.sleep(0.2)
    if not os.path.exists(file_path) or os.path.getsize(file_path) < 1024:
        raise ValueError("Uploaded file is empty or invalid.")

    # PERFORMANCE OPTIMIZATION: Transcribe the entire file at once
    print("Starting optimized transcription of the entire file...")
    English_Fraud_Prompt = "bank, account, OTP, one-time password, transaction, credit card, debit card, CVV, security, verify, reverse, payment, fraud, alert, KYC, customer support, computer, virus."
    
    transcription_result = TRANSCRIBER.translate_entire_file(
        file_path, initial_prompt=English_Fraud_Prompt
    )
    
    if not transcription_result or not transcription_result["full_text"].strip():
        return {"status": "error", "message": "No speech could be transcribed from the audio."}
    
    full_english_transcription = transcription_result["full_text"]
    print("Transcription complete.")

    # Lexical Analysis
    textual_features = TEXT_EXTRACTOR.extract_features(full_english_transcription)
    preliminary_result = INITIAL_MODEL.predict(textual_features, {})
    preliminary_score = preliminary_result['fraud_score']

    # LLM Verification (if needed)
    final_result = preliminary_result.copy()
    LLM_VERIFICATION_THRESHOLD = 0.4
    
    if preliminary_score >= LLM_VERIFICATION_THRESHOLD:
        print("Preliminary score is high, escalating to LLM...")
        llm_response = LLM_VERIFIER.verify(full_english_transcription, textual_features)
        if llm_response and 'probability' in llm_response:
            final_result.update({
                'fraud_score': llm_response['probability'],
                'explanation': llm_response['reasoning'],
                'confidence': 'high' if llm_response['probability'] > 0.7 else 'medium',
                'is_fraud': llm_response['probability'] >= 0.5
            })
            print("LLM verification complete.")
        else:
            print("LLM verification failed.")
    
    # Add the full transcript to the final result for the frontend
    final_result['full_transcription'] = full_english_transcription
    return final_result


# --- 3. WebSocket Connection Manager and Pipeline (for real-time updates) ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, job_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[job_id] = websocket

    def disconnect(self, job_id: str):
        if job_id in self.active_connections:
            del self.active_connections[job_id]

    async def send_json(self, job_id: str, data: dict):
        if job_id in self.active_connections:
            try:
                await self.active_connections[job_id].send_json(data)
            except (WebSocketDisconnect, ConnectionResetError):
                print(f"Client for job {job_id} disconnected before message could be sent.")
                self.disconnect(job_id)

manager = ConnectionManager()

async def run_fraud_analysis_with_updates(job_id: str, file_path: str):
    """
    The original, detailed analysis pipeline that sends step-by-step updates
    over a WebSocket connection.
    """
    try:
        await asyncio.sleep(0.5)
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            await manager.send_json(job_id, {"status": "error", "message": "Uploaded file is empty."})
            return

        await manager.send_json(job_id, {"status": "initializing", "message": "File received. Starting analysis..."})
        audio_ingester = AudioIngester(file_path)
        audio_chunks = audio_ingester.get_audio_chunks()
        total_chunks = len(audio_chunks)
        await manager.send_json(job_id, {"status": "transcribing", "message": f"Found {total_chunks} speech chunks."})
        
        full_english_transcription = ""
        for i, chunk in enumerate(audio_chunks):
            english_text = TRANSCRIBER.transcribe_and_translate_chunk(chunk)
            if english_text:
                full_english_transcription += english_text + " "
                await manager.send_json(job_id, {"status": "progress", "step": "transcription", "chunk_number": i + 1, "total_chunks": total_chunks, "text": english_text})

        await manager.send_json(job_id, {"status": "analyzing", "message": "Transcription complete. Analyzing text..."})
        textual_features = TEXT_EXTRACTOR.extract_features(full_english_transcription)
        preliminary_result = INITIAL_MODEL.predict(textual_features, {})
        preliminary_score = preliminary_result['fraud_score']
        await manager.send_json(job_id, {"status": "analyzing", "step": "preliminary_analysis", "message": f"Preliminary score: {preliminary_score:.2f}", "data": preliminary_result})

        final_result = preliminary_result.copy()
        LLM_VERIFICATION_THRESHOLD = 0.4
        if preliminary_score >= LLM_VERIFICATION_THRESHOLD:
            await manager.send_json(job_id, {"status": "verifying", "message": "Escalating to local LLM..."})
            llm_response = LLM_VERIFIER.verify(full_english_transcription, textual_features)
            if llm_response and 'probability' in llm_response:
                final_result.update({'fraud_score': llm_response['probability'], 'explanation': llm_response['reasoning'], 'confidence': 'high' if llm_response['probability'] > 0.7 else 'medium', 'is_fraud': llm_response['probability'] >= 0.5})
                await manager.send_json(job_id, {"status": "verifying", "message": "LLM verification complete."})
            else:
                await manager.send_json(job_id, {"status": "error", "message": "LLM verification failed."})

        await manager.send_json(job_id, {"status": "complete", "result": final_result})

    except Exception as e:
        await manager.send_json(job_id, {"status": "error", "message": str(e)})
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Cleaned up WebSocket job file for job {job_id}.")


# --- 4. API ENDPOINTS ---

# --- NEW: High-performance endpoint for a single, final result ---
@app.post("/analyze/fast/")
async def analyze_full_call_fast(file: UploadFile = File(...)):
    """
    Accepts an audio file, runs the OPTIMIZED full analysis pipeline, and returns
    a single JSON object with the final result. Prioritizes speed.
    """
    job_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOADS_DIR, f"{job_id}.tmp") # Use a temporary, generic name

    try:
        async with aiofiles.open(file_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
        
        result = await perform_full_analysis_optimized(file_path)
        return result

    except Exception as e:
        print(f"Fast analysis job {job_id} failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Cleaned up temporary file for fast analysis job {job_id}.")


# --- Existing Endpoints for Real-Time WebSocket Updates ---
@app.post("/analyze/")
async def create_realtime_analysis_job(file: UploadFile = File(...)):
    """
    Accepts a file and returns a job_id for a real-time analysis
    session via WebSocket.
    """
    job_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOADS_DIR, f"{job_id}_{file.filename}")
    async with aiofiles.open(file_path, 'wb') as out_file:
        content = await file.read()
        await out_file.write(content)
    return {"job_id": job_id}


@app.websocket("/ws/status/{job_id}")
async def websocket_status_endpoint(websocket: WebSocket, job_id: str):
    """

    Handles the WebSocket connection for a real-time analysis job.
    """
    await manager.connect(job_id, websocket)
    
    file_path = None
    for _ in range(5): # Retry finding the file to give it time to save
        for f in os.listdir(UPLOADS_DIR):
            if f.startswith(job_id):
                file_path = os.path.join(UPLOADS_DIR, f)
                break
        if file_path: break
        await asyncio.sleep(0.1)

    if not file_path:
        await manager.send_json(job_id, {"status": "error", "message": "File for job ID not found."})
        manager.disconnect(job_id)
        return

    try:
        asyncio.create_task(run_fraud_analysis_with_updates(job_id, file_path))
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        print(f"Client for job {job_id} disconnected.")
        manager.disconnect(job_id)