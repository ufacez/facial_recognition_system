import logging
import threading
import time
import sys
from datetime import datetime
from queue import Queue
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
    """Main attendance system application - Optimized for 30 FPS"""
    
    def __init__(self):
        logger.info("Initializing TrackSite Attendance System...")
        
        # Database connections
        self.mysql_db = None
        self.sqlite_db = None
        
        # Core components
        self.face_recognizer = None
        self.attendance_logger = None
        self.sync_manager = None
        
        # Hardware interfaces
        self.camera = None
        self.gpio = None
        self.display = None
        
        # System state
        self.is_running = False
        self.timeout_mode = False
        self.last_recognition_time = None
        
        # Performance optimization
        self.target_fps = 30
        self.frame_time = 1.0 / self.target_fps
        self.process_every_n_frames = 2  # Process recognition every 2 frames for speed
        self.frame_counter = 0
        
        # Threading
        self.sync_thread = None
        self.recognition_queue = Queue(maxsize=2)
        self.recognition_thread = None
        self.result_queue = Queue()
    
    def initialize(self) -> bool:
        """Initialize all system components with proper connection handling"""
        logger.info("Initializing components...")
        
        try:
            # Initialize databases
            logger.info("Connecting to databases...")
            self.mysql_db = MySQLDatabase()
            self.sqlite_db = SQLiteDatabase()
            
            # Try MySQL connection (can work offline with SQLite fallback)
            mysql_connected = self.mysql_db.connect()
            if mysql_connected:
                logger.info("✓ MySQL connected")
            else:
                logger.warning("⚠ MySQL unavailable - running in offline mode")
            
            # SQLite always available
            logger.info("✓ SQLite database ready")
            
            # Initialize core components
            logger.info("Initializing core components...")
            self.face_recognizer = FaceRecognizer(self.mysql_db, self.sqlite_db)
            self.attendance_logger = AttendanceLogger(self.mysql_db, self.sqlite_db)
            self.sync_manager = SyncManager(self.mysql_db, self.sqlite_db)
            logger.info("✓ Core components initialized")
            
            # Initialize camera with optimization
            logger.info("Initializing camera...")
            self.camera = Camera()
            if not self.camera.initialize():
                logger.error("❌ Camera initialization failed")
                return False
            
            # Set camera properties for performance
            self.camera.set_fps(30)
            self.camera.set_resolution(640, 480)  # Lower resolution for speed
            logger.info("✓ Camera initialized (640x480 @ 30fps)")
            
            # Setup GPIO
            logger.info("Initializing GPIO...")
            self.gpio = GPIOHandler()
            self.gpio.add_button_callback(self._handle_timeout_button)
            logger.info("✓ GPIO initialized")
            
            # Create display window
            logger.info("Initializing display...")
            self.display = Display()
            self.display.create_window(fullscreen=False)
            logger.info("✓ Display initialized")
            
            # Load face encodings
            logger.info("Loading face encodings...")
            encoding_count = self.face_recognizer.load_encodings()
            if encoding_count == 0:
                logger.warning("⚠ No face encodings loaded - system will not recognize faces")
            else:
                logger.info(f"✓ Loaded {encoding_count} face encodings")
            
            logger.info("="*60)
            logger.info("System initialization complete - Ready to run!")
            logger.info("="*60)
            return True
            
        except Exception as e:
            logger.exception(f"Initialization error: {e}")
            return False
    
    def run(self):
        """Main application loop - Optimized for 30 FPS"""
        logger.info("Starting attendance system...")
        self.is_running = True
        
        # Start background threads
        self.sync_thread = threading.Thread(target=self._sync_worker, daemon=True)
        self.sync_thread.start()
        
        self.recognition_thread = threading.Thread(target=self._recognition_worker, daemon=True)
        self.recognition_thread.start()
        
        # Show startup message
        self.display.show_message("TrackSite Attendance System", duration_ms=1500)
        
        try:
            last_frame_time = time.time()
            fps_counter = 0
            fps_start_time = time.time()
            current_fps = 0
            
            while self.is_running:
                frame_start = time.time()
                
                # Read frame
                ret, frame = self.camera.read_frame()
                if not ret:
                    logger.error("Failed to read camera frame")
                    time.sleep(0.1)
                    continue
                
                # Process frame (lightweight operations only in main loop)
                self.frame_counter += 1
                
                # Queue frame for recognition processing (non-blocking)
                if self.frame_counter % self.process_every_n_frames == 0:
                    if self.recognition_queue.qsize() < 1:  # Don't flood the queue
                        try:
                            self.recognition_queue.put_nowait((frame.copy(), time.time()))
                        except:
                            pass  # Queue full, skip this frame
                
                # Check for recognition results
                annotated_frame = frame
                try:
                    while not self.result_queue.empty():
                        result = self.result_queue.get_nowait()
                        if result:
                            worker_info, result_frame = result
                            if worker_info:
                                self._handle_recognition(worker_info, result_frame)
                except:
                    pass
                
                # Add status overlay (lightweight)
                status = self._get_status_text(current_fps)
                annotated_frame = self.display.add_status_bar(annotated_frame, status)
                
                # Add mode indicator
                if self.timeout_mode:
                    annotated_frame = self.display.add_overlay(
                        annotated_frame,
                        "TIME-OUT MODE",
                        position=(50, 50),
                        color=(0, 165, 255)
                    )
                
                # Display frame
                self.display.show_frame(annotated_frame)
                
                # Calculate FPS
                fps_counter += 1
                if time.time() - fps_start_time >= 1.0:
                    current_fps = fps_counter
                    fps_counter = 0
                    fps_start_time = time.time()
                
                # Handle keyboard input
                key = self.display.wait_key(1)
                if key == ord('q'):
                    logger.info("Quit key pressed")
                    break
                elif key == ord('t'):
                    self._toggle_timeout_mode()
                elif key == ord('r'):
                    self._reload_encodings()
                
                # Frame rate limiting
                elapsed = time.time() - frame_start
                if elapsed < self.frame_time:
                    time.sleep(self.frame_time - elapsed)
        
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        
        except Exception as e:
            logger.exception(f"Error in main loop: {e}")
        
        finally:
            self.shutdown()
    
    def _recognition_worker(self):
        """Background thread for CPU-intensive face recognition"""
        logger.info("Recognition worker started")
        
        while self.is_running:
            try:
                # Get frame from queue (blocking with timeout)
                frame, timestamp = self.recognition_queue.get(timeout=0.1)
                
                # Perform recognition (CPU-intensive operation)
                worker_info, annotated_frame = self.face_recognizer.recognize_face(frame)
                
                # Put result in result queue
                if worker_info:
                    try:
                        self.result_queue.put_nowait((worker_info, annotated_frame))
                    except:
                        pass  # Queue full, skip
                
            except:
                continue  # Queue empty or timeout
        
        logger.info("Recognition worker stopped")
    
    def _handle_recognition(self, worker_info: dict, frame):
        """Handle recognized worker - optimized to prevent lag"""
        # Prevent rapid re-processing
        now = datetime.now()
        if self.last_recognition_time:
            if (now - self.last_recognition_time).seconds < 2:
                return
        
        self.last_recognition_time = now
        
        worker_id = worker_info['worker_id']
        worker_name = f"{worker_info['first_name']} {worker_info['last_name']}"
        
        logger.info(f"Recognized: {worker_name} (ID: {worker_id})")
        
        # Time-in or time-out (run in background to avoid blocking)
        threading.Thread(
            target=self._process_attendance,
            args=(worker_info, frame),
            daemon=True
        ).start()
    
    def _process_attendance(self, worker_info: dict, frame):
        """Process attendance in background thread"""
        worker_id = worker_info['worker_id']
        worker_name = f"{worker_info['first_name']} {worker_info['last_name']}"
        
        if self.timeout_mode:
            result = self.attendance_logger.log_timeout(worker_id)
            self._show_result(result, worker_name, frame)
            
            # Auto-disable timeout mode
            if result['success']:
                self.timeout_mode = False
                self.gpio.set_led(False)
        else:
            result = self.attendance_logger.log_timein(worker_id)
            
            # Handle already-in scenario
            if result.get('action') == 'already_in':
                self._prompt_timeout(worker_info, frame)
            else:
                self._show_result(result, worker_name, frame)
    
    def _show_result(self, result: dict, worker_name: str, frame):
        """Display attendance result with timestamp"""
        current_time = datetime.now()
        
        if result['success']:
            action = result.get('action', 'unknown')
            
            if action == 'timein':
                message = f"✅ TIME-IN: {worker_name}"
                time_str = result.get('time_in', current_time.strftime('%I:%M:%S %p'))
                time_info = f"⏰ Time: {time_str}"
                color = (0, 255, 0)  # Green
            
            elif action == 'timeout':
                message = f"🏁 TIME-OUT: {worker_name}"
                hours = result.get('hours_worked', '0.00')
                time_str = current_time.strftime('%I:%M:%S %p')
                time_info = f"⏱️ Hours Worked: {hours} | Time: {time_str}"
                color = (0, 165, 255)  # Orange
            
            else:
                message = result.get('message', 'Success')
                time_info = f"🕐 {current_time.strftime('%I:%M:%S %p')}"
                color = (0, 255, 0)
            
            # Display success message
            display_frame = frame.copy()
            display_frame = self.display.add_overlay(
                display_frame, message,
                position=(50, 100), color=color, font_scale=1.5
            )
            
            if time_info:
                display_frame = self.display.add_overlay(
                    display_frame, time_info,
                    position=(50, 150), color=(255, 255, 255), font_scale=1.0
                )
            
            # Add date
            date_str = f"📅 {current_time.strftime('%B %d, %Y')}"
            display_frame = self.display.add_overlay(
                display_frame, date_str,
                position=(50, 200), color=(200, 200, 200), font_scale=0.8
            )
            
            self.display.show_frame(display_frame)
            time.sleep(Config.DISPLAY_FEEDBACK_SECONDS)
        
        else:
            # Display error
            message = result.get('message', 'Error')
            logger.warning(f"Attendance error: {message}")
            
            display_frame = frame.copy()
            display_frame = self.display.add_overlay(
                display_frame, f"❌ {message}",
                position=(50, 100), color=(0, 0, 255), font_scale=1.2
            )
            
            # Add timestamp to error
            time_str = f"🕐 {current_time.strftime('%I:%M:%S %p')}"
            display_frame = self.display.add_overlay(
                display_frame, time_str,
                position=(50, 150), color=(255, 255, 255), font_scale=0.8
            )
            
            self.display.show_frame(display_frame)
            time.sleep(2)
    
    def _prompt_timeout(self, worker_info: dict, frame):
        """Prompt for time-out confirmation"""
        worker_name = f"{worker_info['first_name']} {worker_info['last_name']}"
        
        logger.info(f"Prompting time-out for {worker_name}")
        
        # Display prompt
        prompt_frame = frame.copy()
        prompt_frame = self.display.add_overlay(
            prompt_frame,
            f"{worker_name} - Already Timed In",
            position=(50, 100), color=(255, 255, 0), font_scale=1.2
        )
        prompt_frame = self.display.add_overlay(
            prompt_frame,
            "Press ENTER for Time-Out or ESC to Cancel",
            position=(50, 150), color=(255, 255, 255), font_scale=0.8
        )
        
        self.display.show_frame(prompt_frame)
        
        # Wait for confirmation
        start_time = time.time()
        while time.time() - start_time < Config.TIMEOUT_CONFIRMATION_SECONDS:
            key = self.display.wait_key(100)
            
            if key == 13:  # ENTER key
                result = self.attendance_logger.log_timeout(worker_info['worker_id'])
                self._show_result(result, worker_name, frame)
                return
            
            elif key == 27:  # ESC key
                logger.info("Time-out cancelled")
                return
        
        logger.info("Time-out prompt timed out")
    
    def _toggle_timeout_mode(self):
        """Toggle time-out mode"""
        self.timeout_mode = not self.timeout_mode
        self.gpio.set_led(self.timeout_mode)
        
        mode_text = "TIME-OUT MODE ENABLED" if self.timeout_mode else "TIME-IN MODE"
        logger.info(mode_text)
        
        self.display.show_message(mode_text, duration_ms=1500)
    
    def _handle_timeout_button(self):
        """Callback for GPIO timeout button"""
        logger.info("Timeout button pressed")
        self._toggle_timeout_mode()
    
    def _reload_encodings(self):
        """Reload face encodings from database"""
        logger.info("Reloading face encodings...")
        count = self.face_recognizer.load_encodings()
        self.display.show_message(f"Loaded {count} faces", duration_ms=2000)
    
    def _get_status_text(self, fps: int = 0) -> str:
        """Get system status text with time"""
        status_parts = []
        
        # Connection status
        if self.mysql_db and self.mysql_db.is_connected:
            status_parts.append("🟢 ONLINE")
        else:
            status_parts.append("🔴 OFFLINE")
        
        # Mode
        if self.timeout_mode:
            status_parts.append("⏰ TIME-OUT")
        else:
            status_parts.append("✅ TIME-IN")
        
        # FPS
        status_parts.append(f"📹 {fps} FPS")
        
        # Current time
        now = datetime.now()
        status_parts.append(f"🕐 {now.strftime('%I:%M:%S %p')}")
        
        # Date
        status_parts.append(f"📅 {now.strftime('%b %d, %Y')}")
        
        return " | ".join(status_parts)
    
    def _sync_worker(self):
        """Background thread for database synchronization"""
        logger.info("Sync worker started")
        
        while self.is_running:
            time.sleep(Config.SYNC_INTERVAL_SECONDS)
            
            try:
                # Reconnect to MySQL if disconnected
                if self.mysql_db and not self.mysql_db.is_connected:
                    if self.mysql_db.connect():
                        logger.info("MySQL reconnected")
                        # Reload encodings after reconnection
                        self.face_recognizer.load_encodings()
                
                # Sync buffered records
                if self.sync_manager:
                    result = self.sync_manager.sync_all()
                    
                    if result['synced'] > 0:
                        logger.info(f"Synced {result['synced']} records")
            
            except Exception as e:
                logger.error(f"Sync error: {e}")
        
        logger.info("Sync worker stopped")
    
    def shutdown(self):
        """Clean shutdown of all components"""
        logger.info("Shutting down system...")
        
        self.is_running = False
        
        # Wait for threads
        if self.sync_thread and self.sync_thread.is_alive():
            self.sync_thread.join(timeout=3)
        
        if self.recognition_thread and self.recognition_thread.is_alive():
            self.recognition_thread.join(timeout=3)
        
        # Cleanup resources
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
    print("  🎯 TrackSite Attendance System - Face Recognition")
    print("="*70 + "\n")
    
    try:
        # Create system instance
        logger.info("Creating system instance...")
        print("📦 Creating system instance...")
        system = AttendanceSystem()
        
        # Initialize components
        logger.info("Initializing system components...")
        print("🔧 Initializing system components...")
        if not system.initialize():
            logger.error("System initialization failed!")
            print("\n❌ System initialization failed. Check logs for details.")
            return 1
        
        logger.info("System initialized successfully!")
        print("\n✅ System initialized successfully!")
        print("\n" + "="*70)
        print("  📋 CONTROLS")
        print("="*70)
        print("  • Press 'q' to quit")
        print("  • Press 't' to toggle Time-Out mode")
        print("  • Press 'r' to reload face encodings")
        print("="*70)
        print("\n🎥 Starting camera feed...")
        print(f"⏰ System started at: {datetime.now().strftime('%I:%M:%S %p on %B %d, %Y')}\n")
        
        # Run the main loop
        system.run()
        
        print("\n✅ System shutdown complete.")
        print(f"⏰ Shutdown at: {datetime.now().strftime('%I:%M:%S %p')}\n")
        return 0
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        print(f"⏰ Stopped at: {datetime.now().strftime('%I:%M:%S %p')}\n")
        return 0
    
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        print(f"\n❌ Fatal error: {e}")
        print("📄 Check logs/system.log for details.")
        print(f"⏰ Error occurred at: {datetime.now().strftime('%I:%M:%S %p')}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())