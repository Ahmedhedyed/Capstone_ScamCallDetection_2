
from datetime import datetime
import os

from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, Boolean, Text, JSON
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://callguard_user:callguard_password@localhost:5432/callguard"
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class CallRecord(Base):
    __tablename__ = "call_records"

    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, index=True)
    phone_number = Column(String, index=True)
    contact_name = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    duration = Column(Integer, default=0)
    call_type = Column(String, default="incoming")
    is_scam = Column(Boolean, default=False)
    scam_score = Column(Float, nullable=True)
    scam_reason = Column(Text, nullable=True)
    recording_url = Column(String, nullable=True)
    features = Column(JSON, nullable=True)


class Contact(Base):
    __tablename__ = "contacts"

    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, index=True)
    name = Column(String)
    phone_number = Column(String, index=True)
    email = Column(String, nullable=True)
    is_scam = Column(Boolean, default=False)
    scam_score = Column(Float, nullable=True)
    last_call_date = Column(DateTime, nullable=True)
    call_count = Column(Integer, default=0)
    notes = Column(Text, nullable=True)


class ScamAlert(Base):
    __tablename__ = "scam_alerts"

    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, index=True)
    call_id = Column(String, index=True)
    alert_level = Column(String)
    reason = Column(Text)
    confidence = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    phone_number = Column(String, nullable=True)
    contact_name = Column(String, nullable=True)
    features = Column(JSON, nullable=True)


class ModelPrediction(Base):
    __tablename__ = "model_predictions"

    id = Column(String, primary_key=True, index=True)
    call_id = Column(String, index=True)
    text = Column(Text)
    prediction = Column(Float)
    features = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def init_database():
    Base.metadata.create_all(bind=engine)
    print("Database initialized successfully")

# """
# Database configuration and models
# """

# from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, Boolean, Text, JSON
# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.orm import sessionmaker
# from datetime import datetime
# import os

# # Database URL
# DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/callguard")

# # Create engine
# engine = create_engine(DATABASE_URL)
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# Base = declarative_base()

# # Database Models
# class CallRecord(Base):
#     __tablename__ = "call_records"
    
#     id = Column(String, primary_key=True, index=True)
#     user_id = Column(String, index=True)
#     phone_number = Column(String, index=True)
#     contact_name = Column(String, nullable=True)
#     timestamp = Column(DateTime, default=datetime.utcnow)
#     duration = Column(Integer, default=0)  # in seconds
#     call_type = Column(String)  # incoming, outgoing, missed
#     is_scam = Column(Boolean, default=False)
#     scam_score = Column(Float, nullable=True)
#     scam_reason = Column(Text, nullable=True)
#     recording_url = Column(String, nullable=True)
#     features = Column(JSON, nullable=True)  # Store extracted features

# class Contact(Base):
#     __tablename__ = "contacts"
    
#     id = Column(String, primary_key=True, index=True)
#     user_id = Column(String, index=True)
#     name = Column(String)
#     phone_number = Column(String, index=True)
#     email = Column(String, nullable=True)
#     is_scam = Column(Boolean, default=False)
#     scam_score = Column(Float, nullable=True)
#     last_call_date = Column(DateTime, nullable=True)
#     call_count = Column(Integer, default=0)
#     notes = Column(Text, nullable=True)

# class ScamAlert(Base):
#     __tablename__ = "scam_alerts"
    
#     id = Column(String, primary_key=True, index=True)
#     user_id = Column(String, index=True)
#     call_id = Column(String, index=True)
#     alert_level = Column(String)  # safe, warning, critical
#     reason = Column(Text)
#     confidence = Column(Float)
#     timestamp = Column(DateTime, default=datetime.utcnow)
#     phone_number = Column(String)
#     contact_name = Column(String, nullable=True)
#     features = Column(JSON, nullable=True)

# class ModelPrediction(Base):
#     __tablename__ = "model_predictions"
    
#     id = Column(String, primary_key=True, index=True)
#     call_id = Column(String, index=True)
#     text = Column(Text)
#     prediction = Column(Float)  # Scam probability
#     features = Column(JSON)  # All extracted features
#     timestamp = Column(DateTime, default=datetime.utcnow)

# # Dependency to get database session
# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()

# # Initialize database
# async def init_database():
#     """Create all tables"""
#     Base.metadata.create_all(bind=engine)
#     print("Database initialized successfully")
