from fastapi import FastAPI
from services.audio_ingestion import audio_router

app = FastAPI(title="Audio Service")
app.include_router(audio_router, prefix="/api/audio")