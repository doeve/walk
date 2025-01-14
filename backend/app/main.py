

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
import torch
import torch.distributed as dist
import yaml
import numpy as np
from modeling import models as op_models
from data import transform
import logging
from utils import get_msg_mgr





logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()



os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'

if not dist.is_initialized():
    dist.init_process_group(backend='gloo', rank=0, world_size=1)

msg_mgr = get_msg_mgr()
msg_mgr.init_manager(
    save_path="./log",
    log_to_file=False, 
    log_iter=100
)

with open('/app/OpenGait/configs/gaitbase/gaitbase_ccpg.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

if 'data_cfg' not in cfg:
    cfg['data_cfg'] = {
        'dataset_name': 'CASIA-B',
        'num_workers': 1,
        'dataset_root': '',
        'test_dataset_name': 'CASIA-B'
    }

model = op_models.Baseline(cfg, training=False)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

os.makedirs("uploads/training", exist_ok=True)
os.makedirs("uploads/prediction", exist_ok=True)

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

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.load_state_dict(torch.load('OpenGait/pretrained_models/gaitgl_CASIA-B.pt', map_location=DEVICE))

model.to(DEVICE)
model.eval()

async def process_training_video(training_session_id: int, video_path: str, db: Session):
    """Background task to process training video"""
    try:
        print(f"Processing video for training session {training_session_id}")
        crud.update_training_session_status(db, training_session_id, schemas.SessionStatus.processing)
        
        sequences = transform.extract_gait_sequences(video_path)
        with torch.no_grad():
            features = model(sequences.to(DEVICE))
            feature_vector = features.cpu().numpy()
        
        pattern_data = {
            "features": feature_vector.tolist(), 
            "metadata": {
                "model_version": "OpenGait_v1",
                "feature_dim": feature_vector.shape[-1]
            }
        }
        
        confidence_score = float(np.mean(np.abs(feature_vector)))
        
        pattern = schemas.GaitPatternCreate(
            user_id=db.query(models.TrainingSession).get(training_session_id).user_id,
            pattern_data=pattern_data,
            confidence_score=confidence_score
        )
        db_pattern = crud.create_gait_pattern(db, pattern)
        
        crud.update_training_session_status(db, training_session_id, schemas.SessionStatus.completed)
        
    except Exception as e:
        print(f"Error in training processing: {str(e)}")
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


    db_user = crud.get_user(db, user_id=user_id)
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"training_{user_id}_{timestamp}.mp4"
    file_location = f"uploads/training/{filename}"
    
    try:
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(video.file, file_object)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Could not save video file")
    


    training_session = schemas.TrainingSessionCreate(
        user_id=user_id,
        video_path=file_location,
        status=schemas.SessionStatus.pending
    )
    
    db_session = crud.create_training_session(db, training_session)
    


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



async def process_prediction_video(video_path: str, db: Session) -> dict:
    """Process video for prediction"""
    try:


        sequences = transform.extract_gait_sequences(video_path)
        with torch.no_grad():
            input_features = model(sequences.to(DEVICE))
            input_vector = input_features.cpu().numpy()
        


        stored_patterns = crud.get_all_gait_patterns(db)
        
        predictions = []
        for pattern in stored_patterns:
            stored_vector = np.array(pattern.pattern_data["features"])
            


            similarity = np.dot(input_vector.flatten(), stored_vector.flatten()) / (
                np.linalg.norm(input_vector.flatten()) * np.linalg.norm(stored_vector.flatten())
            )
            
            predictions.append({
                "user_id": pattern.user_id,
                "confidence": float(similarity)
            })
        


        predictions.sort(key=lambda x: x["confidence"], reverse=True)
        
        return {
            "predictions": predictions[:5]  # Return top 5 matches
        }
        
    except Exception as e:
        print(f"Error in prediction processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict", response_model=schemas.PredictionLog)
async def create_prediction(
    video: UploadFile = File(...),
    db: Session = Depends(get_db)
):


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"prediction_{timestamp}.mp4"
    file_location = f"uploads/prediction/{filename}"
    
    try:
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(video.file, file_object)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Could not save video file")
    


    results = await process_prediction_video(file_location, db)
    


    best_prediction = max(results["predictions"], key=lambda x: x["confidence"])
    


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



@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}