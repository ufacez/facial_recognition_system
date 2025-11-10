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
    """Optimized face recognition - 30 FPS capable"""
    
    def __init__(self, mysql_db: MySQLDatabase, sqlite_db: SQLiteDatabase):
        self.mysql_db = mysql_db
        self.sqlite_db = sqlite_db
        self.known_encodings: List[np.ndarray] = []
        self.known_metadata: List[Dict[str, Any]] = []
        self.last_update: Optional[float] = None
        
        # Performance settings
        self.scale_factor = 0.5  # 50% size for speed
        self.tolerance = 0.5  # Recognition threshold
    
    def load_encodings(self) -> int:
        """Load face encodings from database"""
        logger.info("Loading face encodings...")
        
        # Try MySQL first
        encodings = []
        if self.mysql_db and self.mysql_db.is_connected:
            encodings = self._load_from_mysql()
            if encodings and self.sqlite_db:
                self.sqlite_db.cache_face_encodings(encodings)
        else:
            # Fallback to SQLite
            if self.sqlite_db:
                encodings = self.sqlite_db.get_cached_encodings()
                logger.warning("Using cached encodings (offline)")
        
        # Parse encodings
        self.known_encodings = []
        self.known_metadata = []
        
        for enc_data in encodings:
            try:
                encoding_array = np.array(json.loads(enc_data['encoding_data']))
                self.known_encodings.append(encoding_array)
                
                self.known_metadata.append({
                    'worker_id': enc_data['worker_id'],
                    'first_name': enc_data['first_name'],
                    'last_name': enc_data['last_name'],
                    'worker_code': enc_data['worker_code']
                })
            except Exception as e:
                logger.error(f"Failed to parse encoding: {e}")
        
        logger.info(f"Loaded {len(self.known_encodings)} encodings")
        return len(self.known_encodings)
    
    def _load_from_mysql(self) -> List[Dict[str, Any]]:
        """Load from MySQL"""
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
    
    def recognize_face(self, frame: np.ndarray) -> Tuple[Optional[Dict[str, Any]], np.ndarray]:
        """
        Fast face recognition
        
        Returns:
            (worker_info, annotated_frame) or (None, original_frame)
        """
        if not self.known_encodings:
            return None, frame
        
        # Resize for speed
        small_frame = cv2.resize(frame, (0, 0), fx=self.scale_factor, fy=self.scale_factor)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces (HOG is faster)
        face_locations = face_recognition.face_locations(
            rgb_frame, 
            model='hog',
            number_of_times_to_upsample=1
        )
        
        if not face_locations:
            return None, frame
        
        # Get encodings
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        # Scale back to original size
        scale_reciprocal = 1.0 / self.scale_factor
        
        # Match faces
        for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
            # Scale coordinates
            top = int(top * scale_reciprocal)
            right = int(right * scale_reciprocal)
            bottom = int(bottom * scale_reciprocal)
            left = int(left * scale_reciprocal)
            
            # Compare
            matches = face_recognition.compare_faces(
                self.known_encodings,
                encoding,
                tolerance=self.tolerance
            )
            
            if True not in matches:
                # Unknown
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 3)
                cv2.putText(frame, "Unknown", (left, top - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                continue
            
            # Best match
            face_distances = face_recognition.face_distance(
                self.known_encodings, 
                encoding
            )
            best_match_idx = np.argmin(face_distances)
            
            if matches[best_match_idx]:
                worker_info = self.known_metadata[best_match_idx]
                confidence = 1 - face_distances[best_match_idx]
                
                # Draw box (green)
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 4)
                
                # Label
                label = f"{worker_info['first_name']} {worker_info['last_name']}"
                conf_text = f"{confidence*100:.0f}%"
                
                # Background for text
                label_y = top - 40
                if label_y < 0:
                    label_y = bottom + 30
                
                # Semi-transparent background
                overlay = frame.copy()
                cv2.rectangle(overlay, (left, label_y - 30), (right, label_y + 5), (0, 255, 0), -1)
                cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
                
                # Text
                cv2.putText(frame, label, (left + 5, label_y - 12),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, f"✓ {conf_text}", (left + 5, label_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                return worker_info, frame
        
        return None, frame
    
    def train_new_face(self, images: List[np.ndarray], worker_id: int) -> bool:
        """Train new face"""
        encodings = []
        
        logger.info(f"Training face for worker {worker_id}...")
        
        for idx, img in enumerate(images):
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_img)
            
            if not face_locations:
                logger.warning(f"No face in image {idx+1}")
                continue
            
            if len(face_locations) > 1:
                logger.warning(f"Multiple faces in image {idx+1}")
            
            face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
            if face_encodings:
                encodings.append(face_encodings[0])
                logger.info(f"✓ Processed image {idx+1}")
        
        if len(encodings) < 3:
            logger.error(f"Need 3+ images (got {len(encodings)})")
            return False
        
        # Average encodings
        avg_encoding = np.mean(encodings, axis=0)
        encoding_json = json.dumps(avg_encoding.tolist())
        
        # Store
        if not self.mysql_db or not self.mysql_db.is_connected:
            logger.error("MySQL not connected")
            return False
        
        query = """
            INSERT INTO face_encodings 
            (worker_id, encoding_data, is_active)
            VALUES (%s, %s, 1)
        """
        encoding_id = self.mysql_db.execute_query(query, (worker_id, encoding_json))
        
        if encoding_id:
            logger.info(f"✅ Trained worker {worker_id}")
            self.load_encodings()
            return True
        else:
            logger.error("Failed to store encoding")
            return False