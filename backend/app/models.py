from sqlalchemy import Column, Integer, String, DateTime, Float, JSON, Enum, ForeignKey, Boolean, Text
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from .database import Base

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    email = Column(String(100), unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    gait_patterns = relationship("GaitPattern", back_populates="user")
    training_sessions = relationship("TrainingSession", back_populates="user")
    predictions = relationship("PredictionLog", back_populates="predicted_user")

class GaitPattern(Base):
    __tablename__ = "gait_patterns"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'))
    pattern_data = Column(JSON, nullable=False)
    confidence_score = Column(Float, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())

    user = relationship("User", back_populates="gait_patterns")
    metrics = relationship("GaitMetrics", back_populates="pattern")

class TrainingSession(Base):
    __tablename__ = "training_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'))
    video_path = Column(String(255), nullable=False)
    metrics = Column(JSON)
    status = Column(Enum('pending', 'processing', 'completed', 'failed', name='session_status'))
    created_at = Column(DateTime, default=func.now())
    completed_at = Column(DateTime)
    error_message = Column(Text)

    user = relationship("User", back_populates="training_sessions")
    parameters = relationship("TrainingParameters", back_populates="session")

class PredictionLog(Base):
    __tablename__ = "prediction_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    input_type = Column(Enum('live', 'video', name='input_type'))
    predicted_user_id = Column(Integer, ForeignKey('users.id', ondelete='SET NULL'))
    confidence_score = Column(Float, nullable=False)
    prediction_data = Column(JSON, nullable=False)
    video_path = Column(String(255))
    created_at = Column(DateTime, default=func.now())

    predicted_user = relationship("User", back_populates="predictions")

class GaitMetrics(Base):
    __tablename__ = "gait_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    pattern_id = Column(Integer, ForeignKey('gait_patterns.id', ondelete='CASCADE'))
    stride_length = Column(Float)
    step_width = Column(Float)
    cadence = Column(Float)
    gait_speed = Column(Float)
    symmetry_score = Column(Float)
    created_at = Column(DateTime, default=func.now())

    pattern = relationship("GaitPattern", back_populates="metrics")

class TrainingParameters(Base):
    __tablename__ = "training_parameters"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey('training_sessions.id', ondelete='CASCADE'))
    parameter_name = Column(String(100), nullable=False)
    parameter_value = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=func.now())

    session = relationship("TrainingSession", back_populates="parameters")