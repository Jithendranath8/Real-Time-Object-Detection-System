"""
Detection Engine Module

This module handles YOLOv8 model loading and inference.
It provides a clean interface for object detection on CPU-only systems.
"""

import sys
from pathlib import Path

# Add project root to Python path to allow imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
from typing import List, Tuple, Optional
from ultralytics import YOLO
import src.config as config

# Try to import torch for MPS detection
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def get_device() -> str:
    """
    Get the best available device for inference.
    Tries MPS (Apple Silicon) first, falls back to CPU.
    
    Returns:
        Device string: "mps" or "cpu"
    """
    if not TORCH_AVAILABLE:
        return "cpu"
    
    if config.USE_MPS:
        try:
            # Check if MPS is available (Apple Silicon Macs)
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            # MPS not available or error checking
            pass
    
    return "cpu"


class DetectionEngine:
    """
    A class to handle YOLOv8 model loading and object detection.
    
    This engine supports MPS (Apple Silicon) and CPU inference.
    Automatically selects the best available device.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the detection engine.
        
        Args:
            model_path: Path to the YOLOv8 model file. If None, uses config.MODEL_NAME
                       and ultralytics will download it automatically.
        """
        self.model_path = model_path or config.MODEL_NAME
        self.model = None
        self.class_names = []
        self.device = get_device()  # Get best available device (MPS or CPU)
        config.DEVICE = self.device  # Update config with selected device
        self._load_model()
    
    def _load_model(self):
        """
        Load the YOLOv8 model and ensure it runs on the selected device (MPS or CPU).
        
        The model is loaded using ultralytics YOLO class, which automatically
        handles model downloading if the weights file doesn't exist locally.
        """
        try:
            print(f"Loading YOLOv8 model: {self.model_path}")
            print(f"Selected device: {self.device}")
            
            # Load YOLOv8 model - ultralytics will download if needed
            self.model = YOLO(self.model_path)
            
            # Move model to selected device (MPS or CPU)
            # This ensures optimal performance on Apple Silicon or CPU fallback
            self.model.to(self.device)
            
            # Get class names from the model
            # YOLOv8 COCO dataset has 80 classes
            if hasattr(self.model, 'names'):
                self.class_names = list(self.model.names.values())
            else:
                # Fallback: use COCO class names
                self.class_names = self._get_coco_class_names()
            
            print(f"Model loaded successfully on {self.device}")
            print(f"Available classes: {len(self.class_names)}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLOv8 model: {str(e)}")
    
    def _get_coco_class_names(self) -> List[str]:
        """
        Get COCO dataset class names as fallback.
        
        Returns:
            List of 80 COCO class names
        """
        return [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
    
    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float, int, str]]:
        """
        Perform object detection on a single frame.
        
        Args:
            frame: Input frame as numpy array in BGR format (OpenCV format)
            
        Returns:
            List of detections, where each detection is a tuple:
            (xmin, ymin, xmax, ymax, confidence, class_id, class_name)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")
        
        # Run inference on the frame
        # The model automatically handles resizing and preprocessing
        # Results are returned in a Results object
        results = self.model.predict(
            frame,
            conf=config.CONF_THRESHOLD,  # Confidence threshold
            iou=config.IOU_THRESHOLD,    # IoU threshold for NMS
            device=self.device,          # Use selected device (MPS or CPU)
            verbose=False                # Suppress verbose output
        )
        
        # Parse results
        detections = []
        
        # results is a list of Results objects (one per image)
        if len(results) > 0:
            result = results[0]  # Get first (and only) result
            
            # Check if there are any detections
            if result.boxes is not None and len(result.boxes) > 0:
                # Extract bounding boxes, confidences, and class IDs
                # Always convert to CPU numpy arrays regardless of device
                boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes in xyxy format
                confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
                class_ids = result.boxes.cls.cpu().numpy().astype(int)  # Class IDs
                
                # Process each detection
                for i in range(len(boxes)):
                    xmin, ymin, xmax, ymax = boxes[i]
                    confidence = float(confidences[i])
                    class_id = int(class_ids[i])
                    
                    # Get class name
                    if class_id < len(self.class_names):
                        class_name = self.class_names[class_id]
                    else:
                        class_name = f"class_{class_id}"
                    
                    # Apply class filtering if configured
                    if config.ALLOWED_CLASSES is not None:
                        if class_name not in config.ALLOWED_CLASSES:
                            continue  # Skip this detection
                    
                    # Add detection to list
                    detections.append((
                        int(xmin), int(ymin), int(xmax), int(ymax),
                        confidence, class_id, class_name
                    ))
        
        return detections
    
    def get_class_names(self) -> List[str]:
        """
        Get list of all available class names.
        
        Returns:
            List of class names
        """
        return self.class_names.copy()
    
    def get_device_name(self) -> str:
        """
        Get the device name being used for inference.
        
        Returns:
            Device string: "mps" or "cpu"
        """
        return self.device
    
    def update_thresholds(self, conf_threshold: float, iou_threshold: float):
        """
        Update detection thresholds.
        
        Args:
            conf_threshold: New confidence threshold (0.0 to 1.0)
            iou_threshold: New IoU threshold (0.0 to 1.0)
        """
        config.CONF_THRESHOLD = max(0.0, min(1.0, conf_threshold))
        config.IOU_THRESHOLD = max(0.0, min(1.0, iou_threshold))

