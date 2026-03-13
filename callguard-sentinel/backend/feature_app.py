from fastapi import FastAPI
from services.feature_extraction import feature_router

app = FastAPI(title="Feature Service")
app.include_router(feature_router, prefix="/api/features")