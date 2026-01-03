"""
Configuration module for the Real-Time Object Detection System.

This module contains all configurable parameters that can be adjusted
without modifying the core detection logic.
"""

# Model Configuration
MODEL_NAME = "yolov8n.pt"  # YOLOv8 nano model (lightweight, CPU-friendly)
MODEL_PATH = None  # If None, ultralytics will download automatically

# Detection Thresholds
CONF_THRESHOLD = 0.5  # Confidence threshold (0.0 to 1.0)
IOU_THRESHOLD = 0.45  # IoU threshold for Non-Maximum Suppression (0.0 to 1.0)

# Device Configuration
# MPS (Metal Performance Shaders) for Apple Silicon Macs
# Falls back to CPU if MPS is not available
USE_MPS = True  # Try to use MPS if available (Apple Silicon)
USE_ONNX = False  # Set to True to use ONNX Runtime for CPU optimization (optional)
DEVICE = "cpu"  # Will be set automatically by DetectionEngine (mps or cpu)

# FPS Configuration
DEFAULT_TARGET_FPS = 12
MIN_TARGET_FPS = 5
MAX_TARGET_FPS = 15  # Maximum target FPS (realistic upper limit)
MAX_FRAME_SKIP = 2  # Maximum frame skip value (0-2)

# Class Filtering
# Set to None to detect all classes, or provide a list of class names to filter
# Example: ALLOWED_CLASSES = ["person", "car", "bicycle"]
ALLOWED_CLASSES = None  # None means detect all classes

# Video Input Configuration
DEFAULT_CAMERA_INDEX = 0  # Default webcam index
DEFAULT_FRAME_WIDTH = 320  # Frame width for processing (smaller = faster on CPU)
DEFAULT_FRAME_HEIGHT = 240  # Frame height for processing

# Display Configuration
FONT_SCALE = 0.6  # Font scale for labels
FONT_THICKNESS = 2  # Thickness of text and boxes
BOX_COLOR = (0, 255, 0)  # BGR color for bounding boxes (green)
TEXT_COLOR = (0, 255, 0)  # BGR color for text labels

