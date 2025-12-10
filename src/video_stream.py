"""
Video Stream Module

This module handles video input from webcam or video files.
It provides a clean interface for frame capture and FPS calculation.
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
from typing import Optional, Generator, Tuple
import src.config as config


class VideoStream:
    """
    A class to handle video input from webcam or video files.
    
    This class manages the video capture, frame reading, and FPS calculation.
    """
    
    def __init__(self, source: Optional[str] = None, camera_index: int = 0):
        """
        Initialize the video stream.
        
        Args:
            source: Path to video file. If None, uses webcam.
            camera_index: Camera index to use (default: 0)
        """
        self.source = source
        self.camera_index = camera_index
        self.cap = None
        self.is_webcam = source is None
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.current_fps = 0.0
        self.frame_width = config.DEFAULT_FRAME_WIDTH
        self.frame_height = config.DEFAULT_FRAME_HEIGHT
        
    def open(self) -> bool:
        """
        Open the video source (webcam or file).
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.is_webcam:
                # Try to open webcam
                self.cap = cv2.VideoCapture(self.camera_index)
                if not self.cap.isOpened():
                    return False
                
                # Set webcam resolution for better performance on CPU
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
                
            else:
                # Open video file
                self.cap = cv2.VideoCapture(self.source)
                if not self.cap.isOpened():
                    return False
                
                # Get video properties
                self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Reset FPS calculation
            self.fps_start_time = time.time()
            self.fps_frame_count = 0
            
            return True
            
        except Exception as e:
            print(f"Error opening video source: {str(e)}")
            return False
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a single frame from the video source.
        
        Returns:
            Tuple of (success, frame) where success is bool and frame is numpy array or None
        """
        if self.cap is None:
            return False, None
        
        ret, frame = self.cap.read()
        
        if ret:
            # Resize frame if needed for better CPU performance
            if self.is_webcam:
                # For webcam, resize to configured dimensions
                frame = cv2.resize(frame, (self.frame_width, self.frame_height))
            else:
                # For video files, optionally resize if too large
                h, w = frame.shape[:2]
                if w > 1280 or h > 720:
                    # Scale down large videos for CPU performance
                    scale = min(1280 / w, 720 / h)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    frame = cv2.resize(frame, (new_w, new_h))
            
            # Update FPS
            self._update_fps()
        
        return ret, frame
    
    def _update_fps(self):
        """
        Update FPS calculation using a simple moving average approach.
        """
        self.fps_frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.fps_start_time
        
        if elapsed >= 1.0:  # Update FPS every second
            self.current_fps = self.fps_frame_count / elapsed
            self.fps_frame_count = 0
            self.fps_start_time = current_time
    
    def get_fps(self) -> float:
        """
        Get current FPS.
        
        Returns:
            Current frames per second
        """
        return self.current_fps
    
    def release(self):
        """
        Release the video capture resource.
        """
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def is_opened(self) -> bool:
        """
        Check if video source is opened.
        
        Returns:
            True if opened, False otherwise
        """
        return self.cap is not None and self.cap.isOpened()
    
    def get_frame_size(self) -> Tuple[int, int]:
        """
        Get current frame dimensions.
        
        Returns:
            Tuple of (width, height)
        """
        return (self.frame_width, self.frame_height)


def create_video_stream(source: Optional[str] = None, camera_index: int = 0) -> VideoStream:
    """
    Factory function to create a VideoStream instance.
    
    Args:
        source: Path to video file. If None, uses webcam.
        camera_index: Camera index to use (default: 0)
        
    Returns:
        VideoStream instance
    """
    return VideoStream(source=source, camera_index=camera_index)

