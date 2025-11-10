import cv2
import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class Display:
    """Enhanced display handler with better text rendering"""
    
    def __init__(self):
        self.window_name = "TrackSite Attendance System"
        self.window_created = False
    
    def create_window(self, fullscreen: bool = False):
        """Create display window"""
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            
            if fullscreen:
                cv2.setWindowProperty(self.window_name, 
                                     cv2.WND_PROP_FULLSCREEN, 
                                     cv2.WINDOW_FULLSCREEN)
            else:
                cv2.resizeWindow(self.window_name, 1280, 720)
            
            self.window_created = True
            logger.info("✅ Display window created")
        
        except Exception as e:
            logger.error(f"Failed to create window: {e}")
    
    def show_frame(self, frame: np.ndarray):
        """Display frame"""
        if self.window_created:
            cv2.imshow(self.window_name, frame)
    
    def wait_key(self, delay: int = 1) -> int:
        """Wait for key press"""
        return cv2.waitKey(delay) & 0xFF
    
    def show_message(self, message: str, duration_ms: int = 2000):
        """Show temporary message"""
        # Create blank frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add message
        frame = self.add_overlay(
            frame, 
            message, 
            position=(320, 240), 
            color=(255, 255, 255),
            font_scale=1.2,
            centered=True
        )
        
        self.show_frame(frame)
        cv2.waitKey(duration_ms)
    
    def add_status_bar(self, frame: np.ndarray, status_text: str) -> np.ndarray:
        """Add status bar at bottom of frame"""
        height, width = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        bar_height = 40
        cv2.rectangle(overlay, (0, height - bar_height), (width, height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Add status text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        
        # Get text size to center it
        (text_width, text_height), baseline = cv2.getTextSize(
            status_text, font, font_scale, thickness
        )
        
        text_x = 10
        text_y = height - (bar_height // 2) + (text_height // 2)
        
        # Add text with shadow for better readability
        cv2.putText(frame, status_text, (text_x + 1, text_y + 1),
                   font, font_scale, (0, 0, 0), thickness + 1)
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
        """Add text overlay to frame with shadow"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        x, y = position
        
        # Center text if requested
        if centered:
            (text_width, text_height), baseline = cv2.getTextSize(
                text, font, font_scale, thickness
            )
            x = x - (text_width // 2)
            y = y + (text_height // 2)
        
        # Draw shadow (black outline)
        cv2.putText(frame, text, (x + 2, y + 2),
                   font, font_scale, (0, 0, 0), thickness + 2)
        
        # Draw main text
        cv2.putText(frame, text, (x, y),
                   font, font_scale, color, thickness)
        
        return frame
    
    def draw_box(self,
                 frame: np.ndarray,
                 top_left: Tuple[int, int],
                 bottom_right: Tuple[int, int],
                 color: Tuple[int, int, int] = (0, 255, 0),
                 thickness: int = 2,
                 label: str = None) -> np.ndarray:
        """Draw bounding box with optional label"""
        x1, y1 = top_left
        x2, y2 = bottom_right
        
        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label if provided
        if label:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            label_thickness = 2
            
            # Get label size
            (label_w, label_h), baseline = cv2.getTextSize(
                label, font, font_scale, label_thickness
            )
            
            # Draw label background
            label_y = y1 - 10
            if label_y < 0:
                label_y = y2 + label_h + 10
            
            cv2.rectangle(frame, 
                         (x1, label_y - label_h - 5),
                         (x1 + label_w + 5, label_y + 5),
                         color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x1 + 3, label_y),
                       font, font_scale, (255, 255, 255), label_thickness)
        
        return frame
    
    def add_info_panel(self,
                       frame: np.ndarray,
                       title: str,
                       info_lines: list,
                       position: Tuple[int, int] = (50, 50),
                       width: int = 400,
                       bg_color: Tuple[int, int, int] = (0, 0, 0),
                       text_color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
        """Add info panel with multiple lines"""
        x, y = position
        line_height = 30
        padding = 15
        
        # Calculate panel height
        panel_height = len(info_lines) * line_height + padding * 2 + 40
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + width, y + panel_height), bg_color, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw border
        cv2.rectangle(frame, (x, y), (x + width, y + panel_height), (100, 100, 100), 2)
        
        # Draw title
        current_y = y + padding + 25
        cv2.putText(frame, title, (x + padding, current_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Draw separator
        current_y += 10
        cv2.line(frame, (x + padding, current_y), 
                (x + width - padding, current_y), (100, 100, 100), 1)
        
        # Draw info lines
        current_y += line_height
        for line in info_lines:
            cv2.putText(frame, line, (x + padding, current_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
            current_y += line_height
        
        return frame
    
    def destroy(self):
        """Destroy display window"""
        if self.window_created:
            cv2.destroyAllWindows()
            logger.info("Display window destroyed")