# 🎯 Real-Time Object Detection System

<div align="center">

**A production-ready object detection system using YOLOv8n with multiple detection modes**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)](https://ultralytics.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Detection Modes](#-detection-modes)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage Guide](#-usage-guide)
- [Performance](#-performance)
- [Configuration](#-configuration)
- [Project Structure](#-project-structure)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)

---

## 🎯 Overview

A comprehensive, production-style object detection system built with YOLOv8n (nano) model. This system offers three distinct detection modes optimized for different use cases, from zero-lag snapshot detection to real-time video streaming. The system automatically leverages Apple Silicon GPU acceleration (MPS) when available, with seamless CPU fallback for maximum compatibility.

### Why This Project?

- ✅ **Zero-lag snapshot mode** for instant detection on demand
- ✅ **Real-time streaming** for continuous monitoring
- ✅ **Apple Silicon optimized** with automatic MPS support
- ✅ **CPU-only compatible** for universal deployment
- ✅ **Production-ready** with error handling and modular architecture

---

## ✨ Key Features

### Core Capabilities

- **🎯 Multiple Detection Modes**
  - **Webcam Capture Mode**: Zero-lag snapshot detection (default)
  - **Video File Stream**: Process pre-recorded videos
  - **Real-Time Webcam**: Continuous live detection

- **🚀 Performance Optimizations**
  - Automatic MPS (Metal Performance Shaders) support for Apple Silicon
  - CPU-only fallback for universal compatibility
  - Configurable FPS throttling (5-15 FPS)
  - Frame skipping for reduced computational load
  - Optimized frame resolution (320x240 default)

- **🎨 Interactive UI Controls**
  - Real-time confidence threshold adjustment (0.0 - 1.0)
  - IoU/NMS threshold control (0.0 - 1.0)
  - Multi-select class filtering (80 COCO classes)
  - Target FPS slider with realistic ranges
  - Frame skip control for performance tuning

- **📊 Real-Time Monitoring**
  - Live FPS display
  - Detection count overlay
  - Detection statistics sidebar
  - Detailed detection lists with confidence scores

- **🛡️ Robust Error Handling**
  - Graceful camera access handling
  - Invalid video file detection
  - Model loading error recovery
  - Dependency verification

---

## 🎬 Detection Modes

### 1. 📸 Webcam Capture Mode (Recommended)

**Best for**: On-demand detection, zero-lag requirements, battery efficiency

- **Zero-lag snapshot detection** - YOLO runs only when you click capture
- Smooth webcam preview without detection overhead
- Instant results with annotated image display
- Detailed detection list with confidence scores
- Perfect for quick object identification tasks

**How it works:**
1. Select "Webcam Capture" mode (default)
2. Allow camera access when prompted
3. Position objects in view
4. Click "📸 Capture Image" button
5. View instant detection results

### 2. 📹 Video File Stream Mode

**Best for**: Processing recorded videos, batch analysis, offline detection

- Stream detection from local video files
- Supports MP4, AVI, MOV, and other OpenCV-compatible formats
- Configurable FPS for smooth playback
- Frame-by-frame detection with statistics

**How it works:**
1. Select "Video File Stream" mode
2. Enter path to video file
3. Click "▶️ Start Detection"
4. Watch annotated video with real-time detections

### 3. ⚡ Real-Time Webcam Mode (High CPU)

**Best for**: Continuous monitoring, live surveillance, real-time analysis

- Continuous frame-by-frame detection
- Fragment-based updates for smooth video (no blinking)
- Real-time FPS and detection statistics
- Configurable performance settings

**Note**: This mode is CPU/GPU intensive. Use Webcam Capture mode for better performance.

---

## 🛠️ Tech Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| **Language** | Python | 3.10+ |
| **ML Framework** | Ultralytics YOLOv8 | Latest |
| **Model** | YOLOv8n (nano) | Pre-trained COCO |
| **Computer Vision** | OpenCV | 4.8+ |
| **Web Framework** | Streamlit | 1.28+ |
| **Numerical Computing** | NumPy | 1.24+ |
| **GPU Acceleration** | PyTorch MPS | (Apple Silicon) |

### Device Support

- ✅ **Apple Silicon (M1/M2/M3)**: Automatic MPS acceleration
- ✅ **Intel/AMD CPUs**: Full CPU-only support
- ✅ **Windows/Linux/macOS**: Cross-platform compatibility

---

## 📦 Installation

### Prerequisites

- Python 3.10 or higher
- Webcam (for capture/real-time modes) or video files
- Internet connection (for first-time model download)
- 200MB+ free disk space (for model weights)

### Step-by-Step Setup

#### 1. Clone or Navigate to Project

```bash
cd Real-Time-Object-Detection-System
```

#### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4. Verify Installation

```bash
python check_setup.py
```

This will verify:
- ✅ Python version (3.10+)
- ✅ All required packages
- ✅ Project structure
- ✅ Device availability (MPS/CPU)

#### 5. Model Download

The YOLOv8n model (`yolov8n.pt`) will be automatically downloaded on first run. No manual download required.

---

## 🚀 Quick Start

### 1. Activate Virtual Environment

```bash
source venv/bin/activate  # macOS/Linux
# OR
venv\Scripts\activate     # Windows
```

### 2. Launch Application

```bash
streamlit run src/app_streamlit.py
```

### 3. Open Browser

The application will automatically open at `http://localhost:8501`

### 4. Start Detecting

1. **Select Mode**: Choose "Webcam Capture" (recommended for first-time use)
2. **Allow Camera Access**: Grant permissions when prompted
3. **Adjust Settings** (optional): Set confidence threshold, filter classes
4. **Capture Image**: Click "📸 Capture Image" button
5. **View Results**: See annotated image and detection list

---

## 📖 Usage Guide

### Webcam Capture Mode

**Recommended for most users**

1. **Select Mode**: Choose "Webcam Capture" from sidebar
2. **Configure Settings** (optional):
   - Adjust confidence threshold (default: 0.5)
   - Set IoU threshold (default: 0.45)
   - Filter specific classes (optional)
3. **Capture**: Click "📸 Capture Image" when ready
4. **Review**: Examine annotated image and detection details

**Tips:**
- Higher confidence threshold = fewer but more accurate detections
- Use class filtering to focus on specific objects
- Capture multiple images for different angles

### Video File Stream Mode

1. **Select Mode**: Choose "Video File Stream"
2. **Enter Path**: Provide full path to video file
   - Example: `sample_videos/demo.mp4`
   - Or: `/Users/username/Videos/test.mp4`
3. **Configure Performance**:
   - Set target FPS (5-15 recommended)
   - Adjust frame skip if needed
4. **Start**: Click "▶️ Start Detection"
5. **Monitor**: Watch annotated video with real-time stats

### Real-Time Webcam Mode

1. **Select Mode**: Choose "Real-Time Webcam (High CPU)"
2. **Configure Performance**:
   - Set target FPS (5-12 recommended for CPU)
   - Use frame skip (1-2) to reduce load
3. **Start**: Click "▶️ Start Detection"
4. **Monitor**: View live detections with FPS counter

**Performance Tips:**
- Lower FPS = less CPU usage
- Frame skip = 1 (detect every 2nd frame) for better performance
- Close other applications for smoother operation

---

## ⚡ Performance

### Device-Specific Performance

| Device Type | Expected FPS | Notes |
|------------|--------------|-------|
| **Apple M2/M3** (MPS) | 15-25 FPS | GPU acceleration enabled |
| **Apple M1** (MPS) | 12-20 FPS | GPU acceleration enabled |
| **Intel i7/i9** (CPU) | 8-15 FPS | High-end CPU |
| **Intel i5** (CPU) | 5-10 FPS | Mid-range CPU |
| **Lower-end CPUs** | 3-8 FPS | Basic performance |

### Optimization Strategies

1. **Frame Resolution**: Lower resolution = higher FPS
   - Default: 320x240 (optimized for performance)
   - Configurable in `config.py`

2. **Frame Skipping**: Reduce detection frequency
   - 0 = detect every frame (most accurate, slowest)
   - 1 = detect every 2nd frame (balanced)
   - 2 = detect every 3rd frame (fastest)

3. **Target FPS**: Match to your hardware
   - 5-8 FPS: Lower-end CPUs
   - 10-12 FPS: Mid-range CPUs
   - 12-15 FPS: High-end CPUs / Apple Silicon

4. **Class Filtering**: Fewer classes = faster processing
   - Filter to only needed classes
   - Reduces post-processing overhead

### Apple Silicon Optimization

The system automatically detects and uses MPS (Metal Performance Shaders) on Apple Silicon Macs:

- **Automatic Detection**: No configuration needed
- **GPU Acceleration**: 2-3x faster than CPU-only
- **Seamless Fallback**: Works on all systems
- **Device Display**: Shows "MPS" or "CPU" in UI

---

## ⚙️ Configuration

### Quick Configuration (`src/config.py`)

```python
# Model Configuration
MODEL_NAME = "yolov8n.pt"  # Change to yolov8s.pt for better accuracy

# Detection Thresholds
CONF_THRESHOLD = 0.5   # Default confidence (0.0 - 1.0)
IOU_THRESHOLD = 0.45   # Default IoU for NMS (0.0 - 1.0)

# Device Configuration
USE_MPS = True  # Enable MPS on Apple Silicon (auto-detected)

# FPS Configuration
DEFAULT_TARGET_FPS = 12  # Default target FPS
MIN_TARGET_FPS = 5       # Minimum FPS
MAX_TARGET_FPS = 15      # Maximum FPS

# Frame Resolution
DEFAULT_FRAME_WIDTH = 320   # Lower = faster
DEFAULT_FRAME_HEIGHT = 240  # Lower = faster

# Display Settings
BOX_COLOR = (0, 255, 0)  # BGR color (green)
FONT_SCALE = 0.6         # Text size
```

### Advanced Configuration

**Change Model**: Edit `MODEL_NAME` in `config.py`
- `yolov8n.pt` - Nano (fastest, least accurate)
- `yolov8s.pt` - Small (balanced)
- `yolov8m.pt` - Medium (slower, more accurate)

**Custom Classes**: Set `ALLOWED_CLASSES` in `config.py`
```python
ALLOWED_CLASSES = ["person", "car", "bicycle"]  # Only detect these
```

---

## 📁 Project Structure

```
Real-Time-Object-Detection-System/
├── README.md                 # This file
├── QUICKSTART.md            # Quick reference guide
├── requirements.txt         # Python dependencies
├── check_setup.py          # Setup verification script
├── test_imports.py         # Import testing utility
├── .gitignore              # Git ignore rules
│
├── src/                    # Source code
│   ├── __init__.py
│   ├── config.py          # Configuration settings
│   ├── detection_engine.py # YOLO model & inference
│   ├── video_stream.py    # Video capture & FPS
│   ├── utils.py           # Drawing & utilities
│   └── app_streamlit.py   # Main Streamlit app
│
├── models/                 # Model weights (auto-downloaded)
│   └── yolov8n.pt         # YOLOv8 nano weights
│
└── sample_videos/          # Demo videos (optional)
    └── .gitkeep
```

### Module Responsibilities

- **`config.py`**: Centralized configuration management
- **`detection_engine.py`**: Model loading, device selection, inference
- **`video_stream.py`**: Webcam/video capture, FPS calculation
- **`utils.py`**: Drawing functions, format conversions
- **`app_streamlit.py`**: UI, mode routing, user interaction

---

## 🐛 Troubleshooting

### Common Issues

#### ❌ "ModuleNotFoundError: No module named 'ultralytics'"

**Solution:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

#### ❌ "Failed to open video source"

**For Webcam:**
- Check camera permissions (System Settings → Privacy → Camera)
- Ensure camera isn't used by another application
- Try different camera index in `config.py` (0, 1, 2...)

**For Video File:**
- Verify file path is correct (use absolute path)
- Check file format is supported (MP4, AVI, MOV)
- Ensure file isn't corrupted

#### ❌ Low FPS / Choppy Video

**Solutions:**
1. **Reduce Target FPS**: Set to 5-8 FPS
2. **Increase Frame Skip**: Set to 1 or 2
3. **Lower Resolution**: Already optimized at 320x240
4. **Use Capture Mode**: Zero-lag alternative
5. **Close Other Apps**: Free up CPU resources
6. **Check Device**: Ensure MPS is active on Apple Silicon

#### ❌ "Failed to load YOLOv8 model"

**Solutions:**
- Check internet connection (first-time download)
- Verify `ultralytics` is installed: `pip install ultralytics`
- Check disk space (model is ~6MB)
- Try manual download: Model will be in `~/.ultralytics/weights/`

#### ❌ Camera Not Working on macOS

**Solution:**
1. System Settings → Privacy & Security → Camera
2. Enable camera access for Terminal (or your IDE)
3. Restart the application

#### ❌ MPS Not Detected on Apple Silicon

**Check:**
```bash
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

**If False:**
- Update PyTorch: `pip install --upgrade torch`
- Check macOS version (MPS requires macOS 12.3+)
- System will automatically fall back to CPU

---

## 🎓 Advanced Usage

### Custom Model Support

Replace the model in `config.py`:

```python
MODEL_NAME = "path/to/your/custom_model.pt"
```

### Batch Processing

For processing multiple images/videos, modify `app_streamlit.py` to add batch processing logic.

### Exporting Results

Detection results are available in session state. Add export functionality:

```python
# In app_streamlit.py, add export button
if st.button("Export Results"):
    import json
    results = {
        "detections": st.session_state.last_detections,
        "timestamp": time.time()
    }
    st.download_button("Download JSON", json.dumps(results), "detections.json")
```

---

## 🔮 Future Enhancements

Potential improvements for future versions:

- [ ] Object tracking (SORT/DeepSORT)
- [ ] Region-based alerts
- [ ] Detection logging to CSV/JSON
- [ ] Video recording with annotations
- [ ] Multi-camera support
- [ ] ONNX Runtime optimization
- [ ] Custom model training integration
- [ ] Real-time detection export
- [ ] WebSocket streaming support
- [ ] Docker containerization

---

## 📚 Additional Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [PyTorch MPS Guide](https://pytorch.org/docs/stable/notes/mps.html)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is provided as-is for educational and development purposes.

---

## 🙏 Acknowledgments

- [Ultralytics](https://ultralytics.com/) for YOLOv8
- [Streamlit](https://streamlit.io/) for the web framework
- [OpenCV](https://opencv.org/) for computer vision capabilities

---

<div align="center">

**Built with ❤️ for efficient object detection**

[Report Bug](https://github.com/yourusername/Real-Time-Object-Detection-System/issues) · [Request Feature](https://github.com/yourusername/Real-Time-Object-Detection-System/issues)

</div>
