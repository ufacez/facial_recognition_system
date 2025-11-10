"""
Face Training Script - Simplified for PC Development
Usage: python train_face.py --worker_id 1
"""

import argparse
import cv2
import sys
from config.database import MySQLDatabase, SQLiteDatabase
from models.face_recognizer import FaceRecognizer

def capture_training_images(worker_id: int, num_images: int = 5):
    """Capture training images from webcam"""
    import face_recognition
    
    images = []
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Cannot open webcam!")
        return []
    
    # Set higher resolution for better face detection
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print(f"\n📸 Capturing {num_images} training images for worker {worker_id}")
    print("=" * 60)
    print("Instructions:")
    print("  - Look at camera from different angles")
    print("  - Keep your face clearly visible")
    print("  - Ensure good lighting")
    print("  - Press SPACE to capture image")
    print("  - Press Q to quit")
    print("=" * 60)
    
    count = 0
    face_detected = False
    
    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Cannot read from webcam")
            break
        
        # DO NOT flip the frame - remove mirroring
        display_frame = frame.copy()
        
        # Detect faces in current frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame, model='hog')
        face_detected = len(face_locations) > 0
        
        # Draw rectangles around detected faces
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(display_frame, "Face Detected", (left, top - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Add semi-transparent background for text
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (0, 0), (650, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)
        
        # Add instructions overlay with better formatting
        status_color = (0, 255, 0) if face_detected else (0, 0, 255)
        status_text = "FACE DETECTED - Ready to capture" if face_detected else "NO FACE DETECTED - Position yourself"
        
        cv2.putText(display_frame, status_text, 
                   (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        cv2.putText(display_frame, f"Images Captured: {count}/{num_images}", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        cv2.putText(display_frame, "Instructions:", 
                   (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.putText(display_frame, "1. Ensure good lighting on your face", 
                   (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(display_frame, "2. Look directly at the camera", 
                   (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(display_frame, "3. Try different angles after first capture", 
                   (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(display_frame, "4. Press SPACE when face detected | Press Q to quit", 
                   (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add face detection rectangle guide
        h, w = display_frame.shape[:2]
        face_guide_color = (0, 255, 0) if count < num_images else (0, 255, 255)
        center_x, center_y = w // 2, h // 2
        guide_size = 250
        
        # Draw face guide rectangle
        cv2.rectangle(display_frame, 
                     (center_x - guide_size, center_y - guide_size),
                     (center_x + guide_size, center_y + guide_size),
                     face_guide_color, 2)
        
        cv2.putText(display_frame, "Align your face here", 
                   (center_x - 120, center_y - guide_size - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, face_guide_color, 2)
        
        cv2.imshow("TrackSite - Face Training", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            # Save the original non-flipped frame
            images.append(frame.copy())
            count += 1
            print(f"✓ Captured image {count}/{num_images}")
            
            # Brief flash feedback
            flash = display_frame.copy()
            cv2.rectangle(flash, (0, 0), (w, h), (255, 255, 255), 30)
            cv2.imshow("TrackSite - Face Training", flash)
            cv2.waitKey(100)
            
        elif key == ord('q') or key == ord('Q'):
            print("\n⚠ Training cancelled by user")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    return images

def main():
    parser = argparse.ArgumentParser(description='Train facial recognition for a worker')
    parser.add_argument('--worker_id', type=int, required=True, 
                       help='Worker ID from database')
    parser.add_argument('--num_images', type=int, default=5,
                       help='Number of training images to capture (default: 5)')
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("TrackSite Face Training System")
    print("=" * 60)
    
    # Initialize databases
    mysql_db = MySQLDatabase()
    sqlite_db = SQLiteDatabase()
    
    if not mysql_db.connect():
        print("\n❌ ERROR: Cannot connect to MySQL database!")
        print("Make sure:")
        print("  1. XAMPP MySQL is running")
        print("  2. Database 'construction_management' exists")
        print("  3. .env file has correct credentials")
        return 1
    
    # Verify worker exists
    worker = mysql_db.fetch_one(
        "SELECT * FROM workers WHERE worker_id = %s", 
        (args.worker_id,)
    )
    
    if not worker:
        print(f"\n❌ ERROR: Worker ID {args.worker_id} not found in database!")
        print("Please add the worker through the web dashboard first.")
        return 1
    
    print(f"\n✓ Found worker: {worker['first_name']} {worker['last_name']}")
    print(f"  Worker Code: {worker['worker_code']}")
    print(f"  Position: {worker['position']}")
    
    # Capture images
    images = capture_training_images(args.worker_id, args.num_images)
    
    if len(images) < 3:
        print(f"\n❌ ERROR: Need at least 3 images (captured {len(images)})")
        return 1
    
    print(f"\n⏳ Processing {len(images)} images...")
    
    # Train face
    recognizer = FaceRecognizer(mysql_db, sqlite_db)
    success = recognizer.train_new_face(images, args.worker_id)
    
    if success:
        print("\n" + "=" * 60)
        print("✅ Face training successful!")
        print("=" * 60)
        print(f"Worker {worker['first_name']} {worker['last_name']} can now use facial recognition")
        return 0
    else:
        print("\n❌ Face training failed!")
        print("Please try again with clearer images")
        return 1

if __name__ == "__main__":
    sys.exit(main())