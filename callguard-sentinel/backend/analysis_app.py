from fastapi import FastAPI
from services.analysis import analysis_router

app = FastAPI(title="Analysis Service")
app.include_router(analysis_router, prefix="/api/analysis")