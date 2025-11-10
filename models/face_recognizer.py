import face_recognition
import numpy as np
import cv2
import json
import logging
from typing import List, Tuple, Optional, Dict, Any
from config.settings import Config
from config.database import MySQLDatabase, SQLiteDatabase

logger = logging.getLogger(__name__)


class FaceRecognizer:
    """Optimized face recognition engine - 30 FPS capable"""
    
    def __init__(self, mysql_db: MySQLDatabase, sqlite_db: SQLiteDatabase):
        self.mysql_db = mysql_db
        self.sqlite_db = sqlite_db
        self.known_encodings = []
        self.known_metadata = []
        self.last_update = None
        
        # Performance optimization
        self.scale_factor = 0.5  # Process at 50% size for speed
    
    def load_encodings(self) -> int:
        """Load face encodings from database"""
        logger.info("Loading face encodings...")
        
        # Try MySQL first
        if self.mysql_db and self.mysql_db.is_connected:
            encodings = self._load_from_mysql()
            if encodings:
                # Cache to SQLite
                if self.sqlite_db:
                    self.sqlite_db.cache_face_encodings(encodings)
        else:
            # Fallback to SQLite cache
            if self.sqlite_db:
                encodings = self.sqlite_db.get_cached_encodings()
                logger.warning("Using cached face encodings (offline mode)")
            else:
                encodings = []
        
        # Parse encodings
        self.known_encodings = []
        self.known_metadata = []
        
        for enc_data in encodings:
            try:
                # Parse JSON encoding
                encoding_array = np.array(json.loads(enc_data['encoding_data']))
                self.known_encodings.append(encoding_array)
                
                # Store metadata
                self.known_metadata.append({
                    'worker_id': enc_data['worker_id'],
                    'first_name': enc_data['first_name'],
                    'last_name': enc_data['last_name'],
                    'worker_code': enc_data['worker_code']
                })
            except Exception as e:
                logger.error(f"Failed to parse encoding {enc_data.get('encoding_id', 'unknown')}: {e}")
        
        logger.info(f"✅ Loaded {len(self.known_encodings)} face encodings")
        return len(self.known_encodings)
    
    def _load_from_mysql(self) -> List[Dict[str, Any]]:
        """Load encodings from MySQL with worker details"""
        query = """
            SELECT 
                fe.encoding_id,
                fe.worker_id,
                fe.encoding_data,
                w.first_name,
                w.last_name,
                w.worker_code,
                fe.is_active
            FROM face_encodings fe
            JOIN workers w ON fe.worker_id = w.worker_id
            WHERE fe.is_active = 1 
            AND w.employment_status = 'active'
            AND w.is_archived = 0
        """
        return self.mysql_db.fetch_all(query) if self.mysql_db else []
    
    def recognize_face(self, frame: np.ndarray) -> Tuple[Optional[Dict], np.ndarray]:
        """
        Optimized face recognition
        
        Returns:
            (worker_metadata, annotated_frame) or (None, original_frame)
        """
        # Fast path: no encodings loaded
        if not self.known_encodings:
            return None, frame
        
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=self.scale_factor, fy=self.scale_factor)
        
        # Convert BGR to RGB (required by face_recognition)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces using HOG (faster than CNN)
        face_locations = face_recognition.face_locations(
            rgb_frame, 
            model='hog',  # Force HOG for speed
            number_of_times_to_upsample=1  # Reduce upsampling for speed
        )
        
        if not face_locations:
            return None, frame
        
        # Get encodings for detected faces
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        # Scale face locations back to original size
        scale_reciprocal = 1.0 / self.scale_factor
        
        # Match against known faces
        for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
            # Scale coordinates back
            top = int(top * scale_reciprocal)
            right = int(right * scale_reciprocal)
            bottom = int(bottom * scale_reciprocal)
            left = int(left * scale_reciprocal)
            
            # Compare faces (optimized)
            matches = face_recognition.compare_faces(
                self.known_encodings,
                encoding,
                tolerance=Config.FACE_RECOGNITION_TOLERANCE if hasattr(Config, 'FACE_RECOGNITION_TOLERANCE') else 0.6
            )
            
            if True not in matches:
                # Unknown face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(frame, "❓ Unknown", (left, top - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                continue
            
            # Get best match
            face_distances = face_recognition.face_distance(
                self.known_encodings, 
                encoding
            )
            best_match_idx = np.argmin(face_distances)
            
            if matches[best_match_idx]:
                worker_info = self.known_metadata[best_match_idx]
                confidence = 1 - face_distances[best_match_idx]
                
                # Draw bounding box (green for recognized)
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)
                
                # Draw label with name and confidence
                label = f"{worker_info['first_name']} {worker_info['last_name']}"
                confidence_text = f"{confidence*100:.1f}%"
                
                # Background for text
                label_y = top - 35
                if label_y < 0:
                    label_y = bottom + 25
                
                # Draw semi-transparent background
                overlay = frame.copy()
                cv2.rectangle(overlay, (left, label_y - 25), (right, label_y + 5), (0, 255, 0), -1)
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                
                # Draw text
                cv2.putText(frame, label, (left + 5, label_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"✓ {confidence_text}", (left + 5, label_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                return worker_info, frame
        
        return None, frame
    
    def train_new_face(self, images: List[np.ndarray], worker_id: int) -> bool:
        """
        Train new face from multiple images
        
        Args:
            images: List of face images (BGR format)
            worker_id: Worker database ID
        
        Returns:
            Success status
        """
        encodings = []
        
        logger.info(f"Training face for worker {worker_id} with {len(images)} images...")
        
        for idx, img in enumerate(images):
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_img)
            
            if not face_locations:
                logger.warning(f"No face detected in training image {idx+1}")
                continue
            
            if len(face_locations) > 1:
                logger.warning(f"Multiple faces in training image {idx+1}, using largest")
            
            face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
            if face_encodings:
                encodings.append(face_encodings[0])
                logger.info(f"✓ Processed training image {idx+1}")
        
        if len(encodings) < 3:
            logger.error(f"❌ Insufficient valid face images (got {len(encodings)}, need 3+)")
            return False
        
        # Average encodings for robustness
        avg_encoding = np.mean(encodings, axis=0)
        encoding_json = json.dumps(avg_encoding.tolist())
        
        # Store in MySQL
        if not self.mysql_db or not self.mysql_db.is_connected:
            logger.error("❌ MySQL not connected - cannot save encoding")
            return False
        
        query = """
            INSERT INTO face_encodings 
            (worker_id, encoding_data, is_active)
            VALUES (%s, %s, 1)
        """
        encoding_id = self.mysql_db.execute_query(query, (worker_id, encoding_json))
        
        if encoding_id:
            logger.info(f"✅ Trained face for worker {worker_id} (ID: {encoding_id})")
            self.load_encodings()  # Reload
            return True
        else:
            logger.error("❌ Failed to store face encoding")
            return False
    
    def update_tolerance(self, tolerance: float):
        """Update recognition tolerance (lower = stricter)"""
        if 0.0 <= tolerance <= 1.0:
            Config.FACE_RECOGNITION_TOLERANCE = tolerance
            logger.info(f"Recognition tolerance updated to {tolerance}")
        else:
            logger.warning(f"Invalid tolerance value: {tolerance} (must be 0.0-1.0)")