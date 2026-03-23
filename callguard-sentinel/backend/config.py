import os
from dotenv import load_dotenv

load_dotenv()


def _csv_env(name: str, default: str) -> list[str]:
    raw = os.getenv(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

CORS_ORIGINS = _csv_env(
    "CORS_ORIGINS",
    "http://localhost:8080,http://127.0.0.1:8080,http://localhost:8081,http://127.0.0.1:8081,http://localhost:5173,http://127.0.0.1:5173",
)

AUDIO_SERVICE_URL = os.getenv(
    "AUDIO_SERVICE_URL",
    "http://audio-service:8001/api/audio",
)

TRANSCRIPTION_SERVICE_URL = os.getenv(
    "TRANSCRIPTION_SERVICE_URL",
    "http://transcription-service:8002/api/transcription/process",
)

FEATURE_SERVICE_URL = os.getenv(
    "FEATURE_SERVICE_URL",
    "http://feature-service:8003/api/features/extract",
)

ANALYSIS_SERVICE_URL = os.getenv(
    "ANALYSIS_SERVICE_URL",
    "http://analysis-service:8004/api/analysis/predict",
)

ALERT_SERVICE_URL = os.getenv(
    "ALERT_SERVICE_URL",
    "http://alerting-service:8005/api/alerts/send",
)

ALERT_BROADCAST_URL = os.getenv(
    "ALERT_BROADCAST_URL",
    "http://alerting-service:8005/api/alerts/broadcast",
)

SAFE_THRESHOLD = float(os.getenv("SAFE_THRESHOLD", "0.30"))
WARNING_THRESHOLD = float(os.getenv("WARNING_THRESHOLD", "0.60"))
CRITICAL_THRESHOLD = float(os.getenv("CRITICAL_THRESHOLD", "0.85"))


# import os
# from dotenv import load_dotenv

# load_dotenv()

# class Settings:
#     APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
#     APP_PORT = int(os.getenv("APP_PORT", "8000"))

#     API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

#     DATABASE_URL = os.getenv(
#         "DATABASE_URL",
#         "postgresql://user:password@localhost:5432/callguard"
#     )
#     REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

#     AUDIO_SERVICE_URL = os.getenv(
#         "AUDIO_SERVICE_URL",
#         "http://audio-service:8001"
#     )
#     TRANSCRIPTION_SERVICE_URL = os.getenv(
#         "TRANSCRIPTION_SERVICE_URL",
#         "http://transcription-service:8002"
#     )
#     FEATURE_SERVICE_URL = os.getenv(
#         "FEATURE_SERVICE_URL",
#         "http://feature-service:8003"
#     )
#     ANALYSIS_SERVICE_URL = os.getenv(
#         "ANALYSIS_SERVICE_URL",
#         "http://analysis-service:8004"
#     )
#     ALERT_SERVICE_URL = os.getenv(
#         "ALERT_SERVICE_URL",
#         "http://alerting-service:8005"
#     )

#     SAFE_THRESHOLD = float(os.getenv("SAFE_THRESHOLD", "0.30"))
#     WARNING_THRESHOLD = float(os.getenv("WARNING_THRESHOLD", "0.60"))
#     CRITICAL_THRESHOLD = float(os.getenv("CRITICAL_THRESHOLD", "0.85"))

#     FRONTEND_ORIGINS = [
#         origin.strip()
#         for origin in os.getenv(
#             "FRONTEND_ORIGINS",
#             "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000"
#         ).split(",")
#         if origin.strip()
#     ]

# settings = Settings()