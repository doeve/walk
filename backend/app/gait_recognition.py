import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU only
import json
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models as tf_models
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import logging
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from sqlalchemy.orm import Session
from . import crud, models as db_models, schemas

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
tf.config.set_visible_devices('GPU')

class GaitFeatureExtractor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.previous_keypoints = None
        self.previous_velocity = None

    def extract_keypoints(self, frame: np.ndarray) -> np.ndarray:
        """Extract keypoints from a single frame using MediaPipe."""
        # Convert the BGR image to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        if results.pose_landmarks is None:
            return np.zeros((33, 3))  # MediaPipe returns 33 landmarks
            
        # Convert landmarks to numpy array
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
        return landmarks

    def compute_gait_features(self, keypoints: np.ndarray) -> np.ndarray:
        """Enhanced gait feature computation"""
        # Height normalization
        height = self._calculate_height(keypoints)
        normalized_keypoints = keypoints / height

        # Temporal features
        velocity = np.zeros_like(keypoints)
        acceleration = np.zeros_like(keypoints)
        if self.previous_keypoints is not None:
            velocity = keypoints - self.previous_keypoints
            if self.previous_velocity is not None:
                acceleration = velocity - self.previous_velocity
        
        # Store for next frame
        self.previous_keypoints = keypoints.copy()
        self.previous_velocity = velocity.copy()

        # Joint indices (extended)
        LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
        LEFT_ELBOW, RIGHT_ELBOW = 13, 14
        LEFT_WRIST, RIGHT_WRIST = 15, 16
        LEFT_HIP, RIGHT_HIP = 23, 24
        LEFT_KNEE, RIGHT_KNEE = 25, 26
        LEFT_ANKLE, RIGHT_ANKLE = 27, 28

        # Calculate comprehensive features
        features = []
        
        # 1. Basic step metrics (normalized)
        features.extend(self._compute_step_metrics(normalized_keypoints))
        
        # 2. Joint angles (full body)
        features.extend(self._compute_joint_angles(normalized_keypoints))
        
        # 3. Body proportions
        features.extend(self._compute_body_proportions(normalized_keypoints))
        
        # 4. Symmetry metrics
        features.extend(self._compute_symmetry_metrics(normalized_keypoints))
        
        # 5. Velocity features
        features.extend(self._compute_velocity_features(velocity))
        
        # 6. Acceleration features
        features.extend(self._compute_acceleration_features(acceleration))

        return np.array(features)
    

    def _compute_step_metrics(self, keypoints: np.ndarray) -> List[float]:
        """Compute normalized step and stride metrics."""
        LEFT_ANKLE, RIGHT_ANKLE = 27, 28
        LEFT_HIP, RIGHT_HIP = 23, 24
        
        # Get positions
        left_ankle = keypoints[LEFT_ANKLE]
        right_ankle = keypoints[RIGHT_ANKLE]
        hip_width = np.linalg.norm(keypoints[LEFT_HIP] - keypoints[RIGHT_HIP])
        
        # Calculate metrics
        step_width = np.linalg.norm(left_ankle[:2] - right_ankle[:2]) / hip_width
        step_length = abs(left_ankle[1] - right_ankle[1]) / hip_width
        stride_length = np.linalg.norm(left_ankle - right_ankle) / hip_width
        
        return [step_width, step_length, stride_length]

    def _compute_joint_angles(self, keypoints: np.ndarray) -> List[float]:
        """Compute relevant joint angles."""
        # Joint indices
        LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
        LEFT_HIP, RIGHT_HIP = 23, 24
        LEFT_KNEE, RIGHT_KNEE = 25, 26
        LEFT_ANKLE, RIGHT_ANKLE = 27, 28
        
        # Calculate angles
        left_knee_angle = self._calculate_angle(
            keypoints[LEFT_HIP], keypoints[LEFT_KNEE], keypoints[LEFT_ANKLE])
        right_knee_angle = self._calculate_angle(
            keypoints[RIGHT_HIP], keypoints[RIGHT_KNEE], keypoints[RIGHT_ANKLE])
        hip_angle = self._calculate_angle(
            keypoints[LEFT_HIP], (keypoints[LEFT_HIP] + keypoints[RIGHT_HIP])/2, 
            keypoints[RIGHT_HIP])
        
        return [left_knee_angle, right_knee_angle, hip_angle]

    def _compute_body_proportions(self, keypoints: np.ndarray) -> List[float]:
        """Calculate body segment proportions."""
        # Key points
        NOSE = 0
        LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
        LEFT_HIP, RIGHT_HIP = 23, 24
        LEFT_ANKLE, RIGHT_ANKLE = 27, 28
        
        # Calculate proportions
        torso_length = np.linalg.norm(
            (keypoints[LEFT_SHOULDER] + keypoints[RIGHT_SHOULDER])/2 - 
            (keypoints[LEFT_HIP] + keypoints[RIGHT_HIP])/2)
        leg_length = np.linalg.norm(
            (keypoints[LEFT_HIP] + keypoints[RIGHT_HIP])/2 - 
            (keypoints[LEFT_ANKLE] + keypoints[RIGHT_ANKLE])/2)
        
        torso_leg_ratio = torso_length / (leg_length + 1e-6)
        body_height = keypoints[NOSE][1] - (keypoints[LEFT_ANKLE][1] + keypoints[RIGHT_ANKLE][1])/2
        
        return [torso_leg_ratio, body_height]

    def _compute_symmetry_metrics(self, keypoints: np.ndarray) -> List[float]:
        """Calculate body symmetry metrics."""
        LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
        LEFT_HIP, RIGHT_HIP = 23, 24
        LEFT_KNEE, RIGHT_KNEE = 25, 26
        LEFT_ANKLE, RIGHT_ANKLE = 27, 28
        
        # Left-right symmetry
        leg_symmetry = abs(
            np.linalg.norm(keypoints[LEFT_HIP] - keypoints[LEFT_ANKLE]) -
            np.linalg.norm(keypoints[RIGHT_HIP] - keypoints[RIGHT_ANKLE]))
        
        knee_symmetry = abs(
            np.linalg.norm(keypoints[LEFT_KNEE] - keypoints[LEFT_ANKLE]) -
            np.linalg.norm(keypoints[RIGHT_KNEE] - keypoints[RIGHT_ANKLE]))
        
        return [leg_symmetry, knee_symmetry]

    def _compute_velocity_features(self, velocity: np.ndarray) -> List[float]:
        """Compute velocity-based features."""
        # Key joint velocities
        ankle_velocity = np.mean(np.linalg.norm(velocity[[27, 28]], axis=1))
        hip_velocity = np.mean(np.linalg.norm(velocity[[23, 24]], axis=1))
        
        return [ankle_velocity, hip_velocity]

    def _compute_acceleration_features(self, acceleration: np.ndarray) -> List[float]:
        """Compute acceleration-based features."""
        # Key joint accelerations
        ankle_acceleration = np.mean(np.linalg.norm(acceleration[[27, 28]], axis=1))
        hip_acceleration = np.mean(np.linalg.norm(acceleration[[23, 24]], axis=1))
        
        return [ankle_acceleration, hip_acceleration]

    def _calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calculate angle between three points."""
        v1 = p1[:2] - p2[:2]
        v2 = p3[:2] - p2[:2]
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def _calculate_height(self, keypoints: np.ndarray) -> float:
        """Calculate normalized height from keypoints."""
        # Get relevant keypoints
        NOSE = 0
        LEFT_ANKLE, RIGHT_ANKLE = 27, 28
        
        # Calculate height as distance from nose to average ankle position
        ankle_y = (keypoints[LEFT_ANKLE][1] + keypoints[RIGHT_ANKLE][1]) / 2
        height = abs(keypoints[NOSE][1] - ankle_y)
        
        # Add small epsilon to avoid division by zero
        return height + 1e-6
    
class GaitRecognitionSystem:
    def __init__(self, model_path: str = "models/gait_model", db: Session = None):
        self.model_path = Path(model_path)
        self.db = db
        self.feature_extractor = GaitFeatureExtractor()
        self.model = self._build_model()
        self.scaler = StandardScaler()
        
    def _build_model(self) -> tf.keras.Model:
        """Build sequence model for temporal context."""
        sequence_length = 30  # Adjust based on your needs
        feature_dim = 14  # Updated feature dimension
        
        model = tf_models.Sequential([
            layers.Input(shape=(sequence_length, feature_dim)),
            
            # Bidirectional LSTM layers
            layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
            layers.Dropout(0.3),
            
            layers.Bidirectional(layers.LSTM(64)),
            layers.Dropout(0.3),
            
            # Dense layers
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            
            layers.Dense(len(self.config['users']) or 2, activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    async def process_training_video(self, video_path: str, user_id: int) -> Dict:
        """Process a training video and extract gait patterns."""
        try:
            features = self.extract_video_features(video_path)
            if len(features) == 0:
                raise ValueError("No valid poses detected in video")
            
            # Scale features
            n_sequences, seq_len, n_features = features.shape
            features_reshaped = features.reshape(-1, n_features)
            features_scaled = self.scaler.fit_transform(features_reshaped)
            
            # Calculate confidence score based on feature quality
            confidence_score = self._calculate_confidence(features_scaled)
            
            # Prepare pattern data
            pattern_data = {
                "features": features_scaled.tolist(),
                "metadata": {
                    "video_path": video_path,
                    "timestamp": datetime.now().isoformat(),
                    "feature_dim": n_features,
                    "sequences": n_sequences
                }
            }
            
            return {
                "pattern_data": pattern_data,
                "confidence_score": confidence_score
            }
            
        except Exception as e:
            logger.error(f"Error processing training video: {str(e)}")
            raise

    async def process_prediction_video(self, video_path: str) -> Dict:
        """Process a video for prediction."""
        try:
            features = self.extract_video_features(video_path)
            if len(features) == 0:
                raise ValueError("No valid poses detected in video")
            
            n_sequences, seq_len, n_features = features.shape
            features_reshaped = features.reshape(-1, n_features)
            features_scaled = self.scaler.transform(features_reshaped)
            features_scaled = features_scaled.reshape(n_sequences, seq_len, n_features)
            
            # Get predictions
            predictions = self.model.predict(features_scaled)
            avg_prediction = np.mean(predictions, axis=0)
            
            # Format predictions for all users
            all_predictions = []
            users = crud.get_users(self.db) if self.db else []
            
            for i, confidence in enumerate(avg_prediction):
                if i < len(users):
                    all_predictions.append({
                        "user_id": users[i].id,
                        "confidence": float(confidence)
                    })
            
            # Sort by confidence
            all_predictions.sort(key=lambda x: x["confidence"], reverse=True)
            
            return {
                "predictions": all_predictions[:5]  # Return top 5 matches
            }
            
        except Exception as e:
            logger.error(f"Error processing prediction video: {str(e)}")
            raise

    def _calculate_confidence(self, features: np.ndarray) -> float:
        """Calculate confidence score based on feature quality."""
        # Implement confidence calculation based on:
        # 1. Feature stability (variance)
        # 2. Pose detection confidence
        # 3. Feature completeness
        
        # Simple implementation - can be enhanced
        feature_completeness = np.mean(np.abs(features)) # How non-zero are the features
        feature_stability = 1.0 / (1.0 + np.std(features)) # Lower variance = higher stability
        
        confidence = (feature_completeness + feature_stability) / 2.0
        return float(np.clip(confidence, 0.0, 1.0))

    def save_model(self):
        """Save the current model state."""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_weights(str(self.model_path))
        logger.info(f"Model saved to {self.model_path}")