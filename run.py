#!/usr/bin/env python3
"""
TrackSite Attendance System - Entry Point
Run this file to start the system
"""
import sys
import logging

# Import the AttendanceSystem class from main.py
from main import AttendanceSystem

logger = logging.getLogger(__name__)


def main():
    """Main entry point"""
    print("\n" + "="*60)
    print("  TrackSite Attendance System")
    print("="*60 + "\n")
    
    try:
        # Create system instance
        logger.info("Creating system instance...")
        system = AttendanceSystem()
        
        # Initialize components
        logger.info("Initializing system components...")
        if not system.initialize():
            logger.error("System initialization failed!")
            print("\n❌ System initialization failed. Check logs for details.")
            return 1
        
        logger.info("System initialized successfully!")
        print("\n✓ System initialized successfully!")
        print("\nControls:")
        print("  - Press 'q' to quit")
        print("  - Press 't' to toggle Time-Out mode")
        print("  - Press 'r' to reload face encodings")
        print("\nStarting camera feed...\n")
        
        # Run the main loop
        system.run()
        
        print("\n✓ System shutdown complete.")
        return 0
    
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        return 0
    
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        print(f"\n❌ Fatal error: {e}")
        print("Check logs/system.log for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())