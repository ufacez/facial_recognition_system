"""Test face_recognition installation"""
import sys

print("Testing face_recognition installation...")
print("=" * 60)

# Test 1: Import face_recognition
try:
    import face_recognition
    print("✅ face_recognition imported successfully")
    print(f"   Version: {face_recognition.__version__}")
except ImportError as e:
    print(f"❌ face_recognition import failed: {e}")
    sys.exit(1)

# Test 2: Import face_recognition_models
try:
    import face_recognition_models
    print("✅ face_recognition_models imported successfully")
    print(f"   Location: {face_recognition_models.__file__}")
except ImportError as e:
    print(f"❌ face_recognition_models import failed: {e}")
    sys.exit(1)

# Test 3: Try to use face_recognition
try:
    import numpy as np
    import cv2
    
    # Create a dummy image
    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Try face detection
    face_locations = face_recognition.face_locations(dummy_image)
    print("✅ Face detection function works")
    
except Exception as e:
    print(f"❌ Face recognition function failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ All face_recognition tests passed!")
print("You're ready to use the system!")