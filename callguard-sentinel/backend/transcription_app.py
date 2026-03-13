from fastapi import FastAPI
from services.transcription import transcription_router

app = FastAPI(title="Transcription Service")
app.include_router(transcription_router, prefix="/api/transcription")