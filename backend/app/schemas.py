# app/schemas.py
from pydantic import BaseModel, EmailStr, field_validator
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

class SessionStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"

class InputType(str, Enum):
    live = "live"
    video = "video"

class UserBase(BaseModel):
    name: str
    email: str

class UserCreate(UserBase):
    pass

class User(UserBase):
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class GaitPatternBase(BaseModel):
    user_id: int
    pattern_data: Dict[str, Any]
    confidence_score: float

    @field_validator('confidence_score')
    def validate_confidence_score(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Confidence score must be between 0 and 1')
        return v

class GaitPatternCreate(GaitPatternBase):
    pass

class GaitPattern(GaitPatternBase):
    id: int
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True

class GaitMetricsBase(BaseModel):
    pattern_id: int
    stride_length: Optional[float] = None
    step_width: Optional[float] = None
    cadence: Optional[float] = None
    gait_speed: Optional[float] = None
    symmetry_score: Optional[float] = None

class GaitMetricsCreate(GaitMetricsBase):
    pass

class GaitMetrics(GaitMetricsBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True

class TrainingSessionBase(BaseModel):
    user_id: int
    video_path: str
    metrics: Optional[Dict[str, Any]] = None
    status: SessionStatus = SessionStatus.pending
    error_message: Optional[str] = None

class TrainingSessionCreate(TrainingSessionBase):
    pass

class TrainingSession(TrainingSessionBase):
    id: int
    created_at: datetime
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class PredictionLogBase(BaseModel):
    input_type: InputType
    predicted_user_id: Optional[int] = None
    confidence_score: float
    prediction_data: Dict[str, Any]
    video_path: Optional[str] = None

class PredictionLogCreate(PredictionLogBase):
    pass

class PredictionLog(PredictionLogBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True

class TrainingParametersBase(BaseModel):
    session_id: int
    parameter_name: str
    parameter_value: Dict[str, Any]

class TrainingParametersCreate(TrainingParametersBase):
    pass

class TrainingParameters(TrainingParametersBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True