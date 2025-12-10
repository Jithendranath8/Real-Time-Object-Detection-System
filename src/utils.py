"""
Utility Functions Module

This module contains helper functions for drawing detections,
calculating FPS, and other utility operations.
"""

import sys
from pathlib import Path

# Add project root to Python path to allow imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import cv2
import numpy as np
import time
from typing import List, Tuple
import src.config as config


def draw_detections(
    frame: np.ndarray,
    detections: List[Tuple[int, int, int, int, float, int, str]]
) -> np.ndarray:
    """
    Draw bounding boxes and labels on a frame.
    
    Args:
        frame: Input frame as numpy array (BGR format)
        detections: List of detections, where each detection is:
                   (xmin, ymin, xmax, ymax, confidence, class_id, class_name)
    
    Returns:
        Frame with drawn detections
    """
    # Create a copy to avoid modifying the original
    output_frame = frame.copy()
    
    for detection in detections:
        xmin, ymin, xmax, ymax, confidence, class_id, class_name = detection
        
        # Draw bounding box
        cv2.rectangle(
            output_frame,
            (xmin, ymin),
            (xmax, ymax),
            config.BOX_COLOR,
            config.FONT_THICKNESS
        )
        
        # Prepare label text: "class_name confidence%"
        label = f"{class_name} {confidence:.1%}"
        
        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            config.FONT_SCALE,
            config.FONT_THICKNESS
        )
        
        # Draw background rectangle for text (for better visibility)
        cv2.rectangle(
            output_frame,
            (xmin, ymin - text_height - baseline - 5),
            (xmin + text_width, ymin),
            config.BOX_COLOR,
            -1  # Filled rectangle
        )
        
        # Draw text label
        cv2.putText(
            output_frame,
            label,
            (xmin, ymin - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            config.FONT_SCALE,
            (255, 255, 255),  # White text for visibility
            config.FONT_THICKNESS,
            cv2.LINE_AA
        )
    
    return output_frame


def calculate_fps(prev_time: float) -> Tuple[float, float]:
    """
    Calculate current FPS based on time elapsed since previous frame.
    
    Args:
        prev_time: Previous frame time (from time.time())
    
    Returns:
        Tuple of (current_fps, current_time)
    """
    current_time = time.time()
    elapsed = current_time - prev_time
    
    if elapsed > 0:
        fps = 1.0 / elapsed
    else:
        fps = 0.0
    
    return fps, current_time


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    """
    Draw FPS counter on the frame.
    
    Args:
        frame: Input frame as numpy array
        fps: Current FPS value
    
    Returns:
        Frame with FPS text drawn
    """
    output_frame = frame.copy()
    
    # Format FPS text
    fps_text = f"FPS: {fps:.1f}"
    
    # Draw FPS in top-left corner with background
    (text_width, text_height), baseline = cv2.getTextSize(
        fps_text,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        2
    )
    
    # Background rectangle
    cv2.rectangle(
        output_frame,
        (10, 10),
        (20 + text_width, 20 + text_height),
        (0, 0, 0),  # Black background
        -1
    )
    
    # FPS text
    cv2.putText(
        output_frame,
        fps_text,
        (15, 15 + text_height),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),  # Green text
        2,
        cv2.LINE_AA
    )
    
    return output_frame


def draw_detection_count(frame: np.ndarray, count: int) -> np.ndarray:
    """
    Draw detection count on the frame.
    
    Args:
        frame: Input frame as numpy array
        count: Number of detections
    
    Returns:
        Frame with detection count text drawn
    """
    output_frame = frame.copy()
    
    # Format count text
    count_text = f"Detections: {count}"
    
    # Draw count below FPS
    (text_width, text_height), baseline = cv2.getTextSize(
        count_text,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        2
    )
    
    # Background rectangle
    cv2.rectangle(
        output_frame,
        (10, 40),
        (20 + text_width, 50 + text_height),
        (0, 0, 0),  # Black background
        -1
    )
    
    # Count text
    cv2.putText(
        output_frame,
        count_text,
        (15, 45 + text_height),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),  # Green text
        2,
        cv2.LINE_AA
    )
    
    return output_frame


def convert_bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """
    Convert frame from BGR (OpenCV) to RGB (Streamlit) format.
    
    Args:
        frame: Frame in BGR format
    
    Returns:
        Frame in RGB format
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

