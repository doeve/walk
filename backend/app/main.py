# app/main.py
from fastapi import FastAPI, Depends, HTTPException, File, UploadFile, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from . import crud, models, schemas
from .database import engine, get_db, init_db
from typing import List
import os
import shutil
from datetime import datetime
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    print("Starting up...")
    try:
        init_db()
        print("Database initialized successfully.")
    except Exception as e:
        print(f"Error during database initialization: {e}")

# Ensure upload directories exist
os.makedirs("uploads/training", exist_ok=True)
os.makedirs("uploads/prediction", exist_ok=True)

# User endpoints
@app.post("/users/", response_model=schemas.User)
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = crud.get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    return crud.create_user(db=db, user=user)

@app.get("/users/", response_model=List[schemas.User])
def read_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    users = crud.get_users(db, skip=skip, limit=limit)
    return users

@app.get("/users/{user_id}", response_model=schemas.User)
def read_user(user_id: int, db: Session = Depends(get_db)):
    db_user = crud.get_user(db, user_id=user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user

# Training endpoints
async def process_training_video(training_session_id: int, video_path: str, db: Session):
    """Background task to process training video"""
    try:
        print(f"Processing video for training session {training_session_id}")
        # Update session status to processing
        crud.update_training_session_status(db, training_session_id, schemas.SessionStatus.processing)
        
        # TODO: Implement actual video processing logic here
        # This would include:
        # 1. Extract gait features from the video
        # 2. Create gait pattern
        # 3. Calculate metrics
        
        # For now, we'll create dummy pattern data
        pattern_data = {
            "features": {
                "stride_length": 120.5,
                "step_width": 30.2,
                "cadence": 110.0
            }
        }
        
        # Create gait pattern
        pattern = schemas.GaitPatternCreate(
            user_id=db.query(models.TrainingSession).get(training_session_id).user_id,
            pattern_data=pattern_data,
            confidence_score=0.95
        )
        db_pattern = crud.create_gait_pattern(db, pattern)
        
        # Update session status to completed
        crud.update_training_session_status(db, training_session_id, schemas.SessionStatus.completed)
        
    except Exception as e:
        crud.update_training_session_status(
            db, 
            training_session_id, 
            schemas.SessionStatus.failed,
            str(e)
        )

@app.post("/training/upload/", response_model=schemas.TrainingSession)
async def create_training_session(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    user_id: int = Form(...),
    db: Session = Depends(get_db)
):
    print(f"Received video for user {user_id}")
    # Verify user exists
    db_user = crud.get_user(db, user_id=user_id)
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Save video file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"training_{user_id}_{timestamp}.mp4"
    file_location = f"uploads/training/{filename}"
    
    try:
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(video.file, file_object)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Could not save video file")
    
    # Create training session
    training_session = schemas.TrainingSessionCreate(
        user_id=user_id,
        video_path=file_location,
        status=schemas.SessionStatus.pending
    )
    
    db_session = crud.create_training_session(db, training_session)
    
    # Start background processing
    background_tasks.add_task(
        process_training_video,
        db_session.id,
        file_location,
        db
    )
    
    return db_session

@app.get("/training/{user_id}", response_model=List[schemas.TrainingSession])
def get_user_training_sessions(
    user_id: int,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    db_user = crud.get_user(db, user_id=user_id)
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    return crud.get_user_training_sessions(db, user_id, skip, limit)

# Prediction endpoints
async def process_prediction_video(video_path: str, db: Session) -> dict:
    """Process video for prediction"""
    # TODO: Implement actual prediction logic
    # This would include:
    # 1. Extract gait features
    # 2. Compare with stored patterns
    # 3. Return best matches with confidence scores
    
    # Dummy prediction results
    return {
        "predictions": [
            {"user_id": 1, "confidence": 0.85},
            {"user_id": 2, "confidence": 0.65}
        ]
    }

@app.post("/predict", response_model=schemas.PredictionLog)
async def create_prediction(
    video: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # Save video file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"prediction_{timestamp}.mp4"
    file_location = f"uploads/prediction/{filename}"
    
    try:
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(video.file, file_object)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Could not save video file")
    
    # Process video for prediction
    results = await process_prediction_video(file_location, db)
    
    # Get highest confidence prediction
    best_prediction = max(results["predictions"], key=lambda x: x["confidence"])
    
    # Create prediction log
    prediction_log = schemas.PredictionLogCreate(
        input_type=schemas.InputType.video,
        predicted_user_id=best_prediction["user_id"],
        confidence_score=best_prediction["confidence"],
        prediction_data=results,
        video_path=file_location
    )
    
    return crud.create_prediction_log(db, prediction_log)

@app.get("/predictions/{user_id}", response_model=List[schemas.PredictionLog])
def get_user_predictions(
    user_id: int,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    db_user = crud.get_user(db, user_id=user_id)
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    return crud.get_user_predictions(db, user_id, skip, limit)

# Gait pattern endpoints
@app.get("/gait-patterns/{user_id}", response_model=List[schemas.GaitPattern])
def get_user_gait_patterns(
    user_id: int,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    db_user = crud.get_user(db, user_id=user_id)
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    return crud.get_user_gait_patterns(db, user_id, skip, limit)

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}