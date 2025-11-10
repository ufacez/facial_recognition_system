import cv2
import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class Display:
    """Optimized display handler"""
    
    def __init__(self):
        self.window_name = "TrackSite Attendance"
        self.window_created = False
    
    def create_window(self, fullscreen: bool = False):
        """Create display window"""
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            
            if fullscreen:
                cv2.setWindowProperty(
                    self.window_name, 
                    cv2.WND_PROP_FULLSCREEN, 
                    cv2.WINDOW_FULLSCREEN
                )
                logger.info("Display window created (fullscreen)")
            else:
                cv2.resizeWindow(self.window_name, 1280, 720)
                logger.info("Display window created (1280x720)")
            
            self.window_created = True
        
        except Exception as e:
            logger.error(f"Window creation failed: {e}")
    
    def show_frame(self, frame: np.ndarray):
        """Display frame"""
        if self.window_created:
            cv2.imshow(self.window_name, frame)
    
    def wait_key(self, delay: int = 1) -> int:
        """Wait for key"""
        return cv2.waitKey(delay) & 0xFF
    
    def show_message(self, message: str, duration_ms: int = 2000):
        """Show message"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        frame = self.add_overlay(
            frame, 
            message, 
            position=(320, 240), 
            color=(255, 255, 255),
            font_scale=1.5,
            centered=True
        )
        
        self.show_frame(frame)
        cv2.waitKey(duration_ms)
    
    def add_status_bar(self, frame: np.ndarray, status_text: str) -> np.ndarray:
        """Add status bar"""
        height, width = frame.shape[:2]
        
        # Semi-transparent bar
        overlay = frame.copy()
        bar_height = 50
        cv2.rectangle(overlay, (0, height - bar_height), (width, height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        text_x = 15
        text_y = height - (bar_height // 2) + 8
        
        # Shadow
        cv2.putText(frame, status_text, (text_x + 1, text_y + 1),
                   font, font_scale, (0, 0, 0), thickness + 1)
        # Main text
        cv2.putText(frame, status_text, (text_x, text_y),
                   font, font_scale, (255, 255, 255), thickness)
        
        return frame
    
    def add_overlay(self, 
                    frame: np.ndarray, 
                    text: str, 
                    position: Tuple[int, int],
                    color: Tuple[int, int, int] = (255, 255, 255),
                    font_scale: float = 1.0,
                    thickness: int = 2,
                    centered: bool = False) -> np.ndarray:
        """Add text overlay with shadow"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        x, y = position
        
        if centered:
            (text_width, text_height), _ = cv2.getTextSize(
                text, font, font_scale, thickness
            )
            x = x - (text_width // 2)
            y = y + (text_height // 2)
        
        # Shadow
        cv2.putText(frame, text, (x + 2, y + 2),
                   font, font_scale, (0, 0, 0), thickness + 2)
        
        # Main
        cv2.putText(frame, text, (x, y),
                   font, font_scale, color, thickness)
        
        return frame
    
    def destroy(self):
        """Destroy window"""
        if self.window_created:
            cv2.destroyAllWindows()
            logger.info("Display destroyed")