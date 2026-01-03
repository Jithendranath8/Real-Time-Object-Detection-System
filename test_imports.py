#!/usr/bin/env python3
"""
Quick import test to verify all modules can be imported correctly.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("Testing imports...")
print("=" * 60)

try:
    print("1. Testing src.config...")
    import src.config as config
    print("   ✅ src.config imported successfully")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    sys.exit(1)

try:
    print("2. Testing src.detection_engine...")
    from src.detection_engine import DetectionEngine
    print("   ✅ DetectionEngine imported successfully")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    sys.exit(1)

try:
    print("3. Testing src.video_stream...")
    from src.video_stream import VideoStream, create_video_stream
    print("   ✅ VideoStream and create_video_stream imported successfully")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    sys.exit(1)

try:
    print("4. Testing src.utils...")
    from src.utils import draw_detections, draw_fps, draw_detection_count, convert_bgr_to_rgb
    print("   ✅ All utility functions imported successfully")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    sys.exit(1)

print("=" * 60)
print("✅ All imports successful! The app should work now.")
print("=" * 60)

