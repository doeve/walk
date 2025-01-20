from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List
from . import crud, models, schemas
from .database import engine, SessionLocal, get_db
from .gait_recognition import GaitFeatureExtractor, GaitRecognitionSystem
import shutil
import os
from pathlib import Path
from datetime import datetime
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create upload directories
UPLOAD_DIR = Path("uploads")
TRAINING_DIR = UPLOAD_DIR / "training"
PREDICTION_DIR = UPLOAD_DIR / "prediction"
MODEL_DIR = Path("models")

for dir in [TRAINING_DIR, PREDICTION_DIR, MODEL_DIR]:
    dir.mkdir(parents=True, exist_ok=True)

# Initialize gait recognition system
gait_system = None

@app.on_event("startup")
async def startup_event():
    """Initialize the application."""
    global gait_system
    try:
        # Initialize database
        # init_db()
        logger.info("Database tables created successfully")
        
        # Create upload directories
        for dir in [TRAINING_DIR, PREDICTION_DIR, MODEL_DIR]:
            dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize gait recognition system with a fresh database session
        db = next(get_db())
        try:
            gait_system = GaitRecognitionSystem(
                model_path=str(MODEL_DIR / "gait_model"),
                db=db
            )
            logger.info("Gait recognition system initialized")
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise

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

async def save_upload_file(upload_file: UploadFile, destination: Path) -> Path:
    """Save an uploaded file to the specified destination."""
    try:
        with destination.open("wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
        return destination
    finally:
        upload_file.file.close()

async def process_training(
    training_session_id: int,
    video_path: Path,
    db: Session
):
    """Background task to process training video."""
    try:
        # Update session status to processing
        crud.update_training_session_status(
            db, 
            training_session_id, 
            schemas.SessionStatus.processing
        )
        
        # Get training session
        session = crud.get_training_session(db, training_session_id)
        if not session:
            raise ValueError("Training session not found")
        
        # Process video and extract patterns
        results = await gait_system.process_training_video(str(video_path), session.user_id)
        
        # Create gait pattern
        pattern = schemas.GaitPatternCreate(
            user_id=session.user_id,
            pattern_data=results["pattern_data"],
            confidence_score=results["confidence_score"]
        )
        crud.create_gait_pattern(db, pattern)
        
        # Update session status to completed
        crud.update_training_session_status(
            db, 
            training_session_id, 
            schemas.SessionStatus.completed
        )
        
    except Exception as e:
        logger.error(f"Error processing training video: {e}")
        crud.update_training_session_status(
            db, 
            training_session_id, 
            schemas.SessionStatus.failed,
            str(e)
        )
        raise

@app.post("/training/upload/")
async def upload_training_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    userId: int = Form(...),
    db: Session = Depends(get_db)
):
    """Handle training video upload and processing."""
    try:
        # Verify user exists
        user = crud.get_user(db, user_id=userId)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = TRAINING_DIR / f"training_{userId}_{timestamp}.mp4"
        
        # Save uploaded file
        await save_upload_file(video, video_path)
        
        # Create training session
        session = crud.create_training_session(
            db,
            schemas.TrainingSessionCreate(
                user_id=userId,
                video_path=str(video_path),
                status=schemas.SessionStatus.pending
            )
        )
        
        # Start background processing
        background_tasks.add_task(
            process_training,
            session.id,
            video_path,
            db
        )
        
        return {
            "message": "Training video uploaded successfully",
            "sessionId": session.id,
            "status": session.status
        }
        
    except Exception as e:
        logger.error(f"Error handling training upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/gait/recognize")
async def recognize_gait(
    video: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Process video for gait recognition."""
    try:
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = PREDICTION_DIR / f"prediction_{timestamp}.mp4"
        
        # Save uploaded file
        await save_upload_file(video, video_path)
        
        # Process video for prediction
        results = await gait_system.process_prediction_video(str(video_path))
        
        # Get top prediction
        if results["predictions"]:
            top_prediction = results["predictions"][0]
            
            # Log prediction
            prediction = schemas.PredictionCreate(
                video_path=str(video_path),
                predictions=results["predictions"],
                confidence_score=top_prediction["confidence"],
            )
            crud.create_prediction(db, prediction)
            
            # Enhance response with user details
            enhanced_predictions = []
            for pred in results["predictions"]:
                user = crud.get_user(db, user_id=pred["user_id"])
                if user:
                    enhanced_predictions.append({
                        "userId": user.id,
                        "name": user.name,
                        "confidence": pred["confidence"]
                    })
            
            return {
                "predictions": enhanced_predictions,
                "timestamp": datetime.now().isoformat()
            }
            
        return {
            "predictions": [],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error processing prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    """Check API health status."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)