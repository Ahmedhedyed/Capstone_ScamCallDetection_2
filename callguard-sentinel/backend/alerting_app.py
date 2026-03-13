from fastapi import FastAPI
from services.alerting import alerting_router

app = FastAPI(title="Alerting Service")
app.include_router(alerting_router, prefix="/api/alerts")