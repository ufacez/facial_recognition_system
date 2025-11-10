#!/usr/bin/env python3
"""
TrackSite Attendance System - Entry Point
"""
import sys
import os

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Import main
from main import main

if __name__ == "__main__":
    sys.exit(main())