# app/crud.py
from sqlalchemy.orm import Session
from . import models, schemas
from datetime import datetime
from typing import List, Optional
import os

# User operations
def get_user(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.id == user_id).first()

def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()

def get_users(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.User).offset(skip).limit(limit).all()

def create_user(db: Session, user: schemas.UserCreate):
    try:
        existing_user = get_user_by_email(db, email=user.email)
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already registered")
            
        db_user = models.User(**user.dict())
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        return db_user
    except Exception:
        db.rollback()
        raise

# Training session operations
def create_training_session(db: Session, session: schemas.TrainingSessionCreate):
    db_session = models.TrainingSession(**session.dict())
    db.add(db_session)
    db.commit()
    db.refresh(db_session)
    return db_session

def get_user_training_sessions(db: Session, user_id: int, skip: int = 0, limit: int = 100):
    return db.query(models.TrainingSession)\
             .filter(models.TrainingSession.user_id == user_id)\
             .offset(skip)\
             .limit(limit)\
             .all()

def update_training_session_status(db: Session, session_id: int, 
                                 status: schemas.SessionStatus, 
                                 error_message: Optional[str] = None):
    db_session = db.query(models.TrainingSession).filter(models.TrainingSession.id == session_id).first()
    if db_session:
        db_session.status = status
        if status == schemas.SessionStatus.completed:
            db_session.completed_at = datetime.utcnow()
        if error_message:
            db_session.error_message = error_message
        db.commit()
        db.refresh(db_session)
    return db_session

# Gait pattern operations
def create_gait_pattern(db: Session, pattern: schemas.GaitPatternCreate):
    db_pattern = models.GaitPattern(**pattern.dict())
    db.add(db_pattern)
    db.commit()
    db.refresh(db_pattern)
    return db_pattern

def get_user_gait_patterns(db: Session, user_id: int, skip: int = 0, limit: int = 100):
    return db.query(models.GaitPattern)\
             .filter(models.GaitPattern.user_id == user_id)\
             .filter(models.GaitPattern.is_active == True)\
             .offset(skip)\
             .limit(limit)\
             .all()

# Prediction operations
def create_prediction_log(db: Session, prediction: schemas.PredictionLogCreate):
    db_prediction = models.PredictionLog(**prediction.dict())
    db.add(db_prediction)
    db.commit()
    db.refresh(db_prediction)
    return db_prediction

def get_user_predictions(db: Session, user_id: int, skip: int = 0, limit: int = 100):
    return db.query(models.PredictionLog)\
             .filter(models.PredictionLog.predicted_user_id == user_id)\
             .offset(skip)\
             .limit(limit)\
             .all()

# Gait metrics operations
def create_gait_metrics(db: Session, metrics: schemas.GaitMetricsCreate):
    db_metrics = models.GaitMetrics(**metrics.dict())
    db.add(db_metrics)
    db.commit()
    db.refresh(db_metrics)
    return db_metrics

# Training parameters operations
def create_training_parameters(db: Session, parameters: schemas.TrainingParametersCreate):
    db_params = models.TrainingParameters(**parameters.dict())
    db.add(db_params)
    db.commit()
    db.refresh(db_params)
    return db_params

def get_session_parameters(db: Session, session_id: int):
    return db.query(models.TrainingParameters)\
             .filter(models.TrainingParameters.session_id == session_id)\
             .all()