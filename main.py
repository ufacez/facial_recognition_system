import logging
import threading
import time
import sys
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
from config.settings import Config
from config.database import MySQLDatabase, SQLiteDatabase
from models.face_recognizer import FaceRecognizer
from models.attendance_logger import AttendanceLogger
from models.sync_manager import SyncManager
from utils.camera import Camera
from utils.gpio_handler import GPIOHandler
from utils.display import Display

# Setup logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AttendanceSystem:
    """Main attendance system - 30 FPS optimized"""
    
    def __init__(self):
        logger.info("Initializing TrackSite Attendance System...")
        
        # Database connections
        self.mysql_db: Optional[MySQLDatabase] = None
        self.sqlite_db: Optional[SQLiteDatabase] = None
        
        # Core components
        self.face_recognizer: Optional[FaceRecognizer] = None
        self.attendance_logger: Optional[AttendanceLogger] = None
        self.sync_manager: Optional[SyncManager] = None
        
        # Hardware interfaces
        self.camera: Optional[Camera] = None
        self.gpio: Optional[GPIOHandler] = None
        self.display: Optional[Display] = None
        
        # System state
        self.is_running = False
        self.timeout_mode = False
        self.last_recognition_time: Optional[datetime] = None
        
        # Performance optimization
        self.target_fps = 30
        self.frame_time = 1.0 / self.target_fps
        self.skip_frames = 3  # Process every 3rd frame for recognition
        self.frame_counter = 0
        
        # Threading
        self.sync_thread: Optional[threading.Thread] = None
        self.last_worker_info: Optional[Dict[str, Any]] = None
        self.show_result_lock = threading.Lock()
    
    def initialize(self) -> bool:
        """Initialize all system components"""
        logger.info("Initializing components...")
        
        try:
            # Initialize databases
            logger.info("Connecting to databases...")
            self.mysql_db = MySQLDatabase()
            self.sqlite_db = SQLiteDatabase()
            
            mysql_connected = self.mysql_db.connect()
            if mysql_connected:
                logger.info("✓ MySQL connected")
            else:
                logger.warning("⚠ MySQL unavailable - offline mode")
            
            logger.info("✓ SQLite database ready")
            
            # Initialize core components
            logger.info("Initializing core components...")
            self.face_recognizer = FaceRecognizer(self.mysql_db, self.sqlite_db)
            self.attendance_logger = AttendanceLogger(self.mysql_db, self.sqlite_db)
            self.sync_manager = SyncManager(self.mysql_db, self.sqlite_db)
            logger.info("✓ Core components initialized")
            
            # Initialize camera
            logger.info("Initializing camera...")
            self.camera = Camera()
            if not self.camera.initialize():
                logger.error("❌ Camera initialization failed")
                return False
            
            self.camera.set_fps(30)
            self.camera.set_resolution(640, 480)
            logger.info("✓ Camera initialized (640x480 @ 30fps)")
            
            # Setup GPIO
            logger.info("Initializing GPIO...")
            self.gpio = GPIOHandler()
            self.gpio.add_button_callback(self._handle_timeout_button)
            logger.info("✓ GPIO initialized")
            
            # Create display window (FULLSCREEN)
            logger.info("Initializing display...")
            self.display = Display()
            self.display.create_window(fullscreen=True)  # Changed to fullscreen
            logger.info("✓ Display initialized (fullscreen)")
            
            # Load face encodings
            logger.info("Loading face encodings...")
            encoding_count = self.face_recognizer.load_encodings()
            if encoding_count == 0:
                logger.warning("⚠ No face encodings loaded")
            else:
                logger.info(f"✓ Loaded {encoding_count} face encodings")
            
            logger.info("="*60)
            logger.info("System ready!")
            logger.info("="*60)
            return True
            
        except Exception as e:
            logger.exception(f"Initialization error: {e}")
            return False
    
    def run(self):
        """Main loop - optimized for 30 FPS"""
        logger.info("Starting attendance system...")
        self.is_running = True
        
        # Start background sync
        self.sync_thread = threading.Thread(target=self._sync_worker, daemon=True)
        self.sync_thread.start()
        
        # Show startup message briefly
        self.display.show_message("TrackSite Ready", duration_ms=1000)
        
        try:
            last_frame_time = time.time()
            fps_values = []
            current_fps = 0
            
            while self.is_running:
                loop_start = time.time()
                
                # Read frame
                ret, frame = self.camera.read_frame()
                if not ret or frame is None:
                    logger.error("Failed to read camera frame")
                    time.sleep(0.1)
                    continue
                
                self.frame_counter += 1
                
                # Process recognition (every N frames for performance)
                worker_info = None
                if self.frame_counter % self.skip_frames == 0:
                    worker_info, frame = self.face_recognizer.recognize_face(frame)
                    
                    if worker_info:
                        self._handle_recognition(worker_info, frame)
                
                # Add status overlay
                status = self._get_status_text(current_fps)
                frame = self.display.add_status_bar(frame, status)
                
                # Add mode indicator
                if self.timeout_mode:
                    frame = self.display.add_overlay(
                        frame,
                        "⏱ TIME-OUT MODE",
                        position=(50, 50),
                        color=(0, 165, 255),
                        font_scale=1.5
                    )
                
                # Display frame
                self.display.show_frame(frame)
                
                # FPS calculation
                fps_values.append(1.0 / (time.time() - last_frame_time))
                last_frame_time = time.time()
                
                if len(fps_values) >= 10:
                    current_fps = int(sum(fps_values) / len(fps_values))
                    fps_values = []
                
                # Handle keyboard
                key = self.display.wait_key(1)
                if key == ord('q') or key == 27:  # q or ESC
                    logger.info("Quit key pressed")
                    break
                elif key == ord('t'):
                    self._toggle_timeout_mode()
                elif key == ord('r'):
                    self._reload_encodings()
                
                # Frame limiting for consistent FPS
                elapsed = time.time() - loop_start
                if elapsed < self.frame_time:
                    time.sleep(self.frame_time - elapsed)
        
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        
        except Exception as e:
            logger.exception(f"Error in main loop: {e}")
        
        finally:
            self.shutdown()
    
    def _handle_recognition(self, worker_info: Dict[str, Any], frame):
        """Handle recognized worker"""
        now = datetime.now()
        
        # Prevent rapid re-processing
        if self.last_recognition_time:
            if (now - self.last_recognition_time).total_seconds() < 3:
                return
        
        self.last_recognition_time = now
        
        worker_id = worker_info['worker_id']
        worker_name = f"{worker_info['first_name']} {worker_info['last_name']}"
        
        logger.info(f"Recognized: {worker_name} (ID: {worker_id})")
        
        # Process in background to avoid lag
        threading.Thread(
            target=self._process_attendance,
            args=(worker_info, frame.copy()),
            daemon=True
        ).start()
    
    def _process_attendance(self, worker_info: Dict[str, Any], frame):
        """Process attendance (runs in background)"""
        worker_id = worker_info['worker_id']
        worker_name = f"{worker_info['first_name']} {worker_info['last_name']}"
        
        with self.show_result_lock:
            if self.timeout_mode:
                # Time-out mode
                result = self.attendance_logger.log_timeout(worker_id)
                self._show_result(result, worker_name, frame)
                
                if result['success']:
                    self.timeout_mode = False
                    self.gpio.set_led(False)
            else:
                # Time-in mode
                result = self.attendance_logger.log_timein(worker_id)
                
                if result.get('action') == 'already_in':
                    # Auto time-out
                    timeout_result = self.attendance_logger.log_timeout(worker_id)
                    self._show_result(timeout_result, worker_name, frame)
                else:
                    self._show_result(result, worker_name, frame)
    
    def _show_result(self, result: Dict[str, Any], worker_name: str, frame):
        """Display result with better formatting"""
        current_time = datetime.now()
        
        if result['success']:
            action = result.get('action', 'unknown')
            
            if action == 'timein':
                message = f"✅ TIME-IN: {worker_name}"
                time_str = result.get('time_in', current_time.strftime('%I:%M:%S %p'))
                time_info = f"⏰ {time_str}"
                color = (0, 255, 0)
            
            elif action == 'timeout':
                message = f"🏁 TIME-OUT: {worker_name}"
                hours = result.get('hours_worked', '0.00')
                time_info = f"⏱ Hours: {hours} hrs"
                color = (0, 165, 255)
            
            else:
                message = result.get('message', 'Success')
                time_info = f"🕐 {current_time.strftime('%I:%M:%S %p')}"
                color = (0, 255, 0)
            
            # Create display frame
            display_frame = frame.copy()
            display_frame = self.display.add_overlay(
                display_frame, message,
                position=(50, 200), color=color, font_scale=2.0
            )
            
            display_frame = self.display.add_overlay(
                display_frame, time_info,
                position=(50, 280), color=(255, 255, 255), font_scale=1.5
            )
            
            date_str = f"📅 {current_time.strftime('%B %d, %Y')}"
            display_frame = self.display.add_overlay(
                display_frame, date_str,
                position=(50, 350), color=(200, 200, 200), font_scale=1.0
            )
            
            self.display.show_frame(display_frame)
            time.sleep(3)
        
        else:
            # Error
            message = result.get('message', 'Error')
            logger.warning(f"Attendance error: {message}")
            
            display_frame = frame.copy()
            display_frame = self.display.add_overlay(
                display_frame, f"❌ {message}",
                position=(50, 200), color=(0, 0, 255), font_scale=1.5
            )
            
            time_str = f"🕐 {current_time.strftime('%I:%M:%S %p')}"
            display_frame = self.display.add_overlay(
                display_frame, time_str,
                position=(50, 280), color=(255, 255, 255), font_scale=1.0
            )
            
            self.display.show_frame(display_frame)
            time.sleep(2)
    
    def _toggle_timeout_mode(self):
        """Toggle time-out mode"""
        self.timeout_mode = not self.timeout_mode
        self.gpio.set_led(self.timeout_mode)
        
        mode_text = "⏱ TIME-OUT MODE" if self.timeout_mode else "✅ TIME-IN MODE"
        logger.info(mode_text)
        
        self.display.show_message(mode_text, duration_ms=1500)
    
    def _handle_timeout_button(self):
        """GPIO button callback"""
        logger.info("Timeout button pressed")
        self._toggle_timeout_mode()
    
    def _reload_encodings(self):
        """Reload face encodings"""
        logger.info("Reloading face encodings...")
        count = self.face_recognizer.load_encodings()
        self.display.show_message(f"Loaded {count} faces", duration_ms=2000)
    
    def _get_status_text(self, fps: int = 0) -> str:
        """Get status text"""
        parts = []
        
        if self.mysql_db and self.mysql_db.is_connected:
            parts.append("🟢 ONLINE")
        else:
            parts.append("🔴 OFFLINE")
        
        if self.timeout_mode:
            parts.append("⏰ TIME-OUT")
        else:
            parts.append("✅ TIME-IN")
        
        parts.append(f"📹 {fps} FPS")
        
        now = datetime.now()
        parts.append(f"🕐 {now.strftime('%I:%M:%S %p')}")
        parts.append(f"📅 {now.strftime('%b %d, %Y')}")
        
        return " | ".join(parts)
    
    def _sync_worker(self):
        """Background sync worker"""
        logger.info("Sync worker started")
        
        while self.is_running:
            time.sleep(Config.SYNC_INTERVAL_SECONDS)
            
            try:
                if self.mysql_db and not self.mysql_db.is_connected:
                    if self.mysql_db.connect():
                        logger.info("MySQL reconnected")
                        self.face_recognizer.load_encodings()
                
                if self.sync_manager:
                    result = self.sync_manager.sync_all()
                    if result['synced'] > 0:
                        logger.info(f"Synced {result['synced']} records")
            
            except Exception as e:
                logger.error(f"Sync error: {e}")
        
        logger.info("Sync worker stopped")
    
    def shutdown(self):
        """Clean shutdown"""
        logger.info("Shutting down...")
        
        self.is_running = False
        
        if self.sync_thread:
            self.sync_thread.join(timeout=3)
        
        if self.camera:
            self.camera.release()
        
        if self.gpio:
            self.gpio.cleanup()
        
        if self.display:
            self.display.destroy()
        
        if self.mysql_db:
            self.mysql_db.close()
        
        logger.info("Shutdown complete")


def main():
    """Main entry point"""
    print("\n" + "="*70)
    print("  🎯 TrackSite Attendance System")
    print("="*70 + "\n")
    
    try:
        logger.info("Creating system instance...")
        system = AttendanceSystem()
        
        logger.info("Initializing...")
        if not system.initialize():
            logger.error("Initialization failed!")
            print("\n❌ Failed. Check logs.")
            return 1
        
        logger.info("System ready!")
        print("\n✅ System ready!")
        print("\n" + "="*70)
        print("  📋 CONTROLS")
        print("="*70)
        print("  • Press 'q' or ESC to quit")
        print("  • Press 't' to toggle Time-Out mode")
        print("  • Press 'r' to reload encodings")
        print("="*70 + "\n")
        
        system.run()
        
        print("\n✅ Shutdown complete.\n")
        return 0
    
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted\n")
        return 0
    
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        print(f"\n❌ Fatal error: {e}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())