# config/settings.py

import os
from typing import Tuple, Optional
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration settings for TrackSite Facial Recognition System"""
    
    # MySQL Database (Central Server)
    MYSQL_HOST: str = os.getenv('MYSQL_HOST', 'localhost')
    MYSQL_PORT: int = int(os.getenv('MYSQL_PORT', '3306'))
    MYSQL_USER: str = os.getenv('MYSQL_USER', 'root')
    MYSQL_PASSWORD: str = os.getenv('MYSQL_PASSWORD', '')
    MYSQL_DATABASE: str = os.getenv('MYSQL_DATABASE', 'construction_management')
    
    # SQLite Database (Local Buffer)
    SQLITE_PATH: str = 'data/local.db'
    
    # Face Recognition Settings
    FACE_RECOGNITION_TOLERANCE: float = 0.5  # Lower = stricter (0.6 default)
    FACE_DETECTION_MODEL: str = 'hog'  # 'hog' for CPU, 'cnn' for GPU
    MIN_FACE_SIZE: Tuple[int, int] = (50, 50)  # Minimum face dimensions
    
    # Attendance Logic
    DUPLICATE_TIMEOUT_SECONDS: int = 30  # Prevent duplicate scans
    AUTO_TIMEOUT_ENABLED: bool = True
    TIMEOUT_CONFIRMATION_SECONDS: int = 5
    
    # Synchronization
    SYNC_INTERVAL_SECONDS: int = 300  # 5 minutes
    MAX_RETRY_ATTEMPTS: int = 3
    RETRY_BACKOFF_MULTIPLIER: int = 2
    
    # Hardware (Raspberry Pi)
    CAMERA_RESOLUTION: Tuple[int, int] = (640, 480)
    CAMERA_FRAMERATE: int = 30
    GPIO_TIMEOUT_BUTTON: Optional[int] = None  # GPIO pin for timeout button
    GPIO_MODE_LED: Optional[int] = None  # LED indicator
    
    # Display
    DISPLAY_FEEDBACK_SECONDS: int = 3
    DISPLAY_FONT_SCALE: float = 1.5
    
    # Logging
    LOG_LEVEL: str = 'INFO'
    LOG_FILE: str = 'logs/system.log'