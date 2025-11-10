import cv2
import logging
from threading import Thread, Lock
from queue import Queue

logger = logging.getLogger(__name__)


class Camera:
    """Optimized camera handler with threaded frame reading for 30 FPS"""
    
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None
        self.frame = None
        self.ret = False
        self.is_running = False
        
        # Threading for non-blocking frame capture
        self.lock = Lock()
        self.thread = None
    
    def initialize(self) -> bool:
        """Initialize camera with optimal settings"""
        try:
            logger.info(f"Opening camera {self.camera_index}...")
            self.cap = cv2.VideoCapture(self.camera_index)
            
            if not self.cap.isOpened():
                logger.error("Failed to open camera")
                return False
            
            # Set optimal properties for performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Disable autofocus for stability (if supported)
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            
            # Buffer size (reduce latency)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Test read
            ret, frame = self.cap.read()
            if not ret:
                logger.error("Failed to read test frame")
                return False
            
            # Start threaded reading
            self.is_running = True
            self.thread = Thread(target=self._read_frames, daemon=True)
            self.thread.start()
            
            logger.info("Camera initialized successfully")
            logger.info(f"Resolution: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
            logger.info(f"FPS: {int(self.cap.get(cv2.CAP_PROP_FPS))}")
            
            return True
        
        except Exception as e:
            logger.exception(f"Camera initialization error: {e}")
            return False
    
    def _read_frames(self):
        """Background thread to continuously read frames"""
        while self.is_running:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                
                with self.lock:
                    self.ret = ret
                    self.frame = frame
    
    def read_frame(self):
        """Get the latest frame (non-blocking)"""
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else None
    
    def set_resolution(self, width: int, height: int):
        """Set camera resolution"""
        if self.cap:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            logger.info(f"Resolution set to {width}x{height}")
    
    def set_fps(self, fps: int):
        """Set camera FPS"""
        if self.cap:
            self.cap.set(cv2.CAP_PROP_FPS, fps)
            logger.info(f"FPS set to {fps}")
    
    def release(self):
        """Release camera resources"""
        logger.info("Releasing camera...")
        self.is_running = False
        
        if self.thread:
            self.thread.join(timeout=2)
        
        if self.cap:
            self.cap.release()
        
        logger.info("Camera released")