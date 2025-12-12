"""
Main Streamlit Application

Entry point for the Real-Time Object Detection System.
Run locally with: streamlit run src/app_streamlit.py
"""

import sys
import traceback
from pathlib import Path

# Add project root to Python path to allow imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Required libraries
try:
    import streamlit as st
except Exception as e:
    # If Streamlit can't be imported, we cannot render anything.
    print("ERROR: streamlit is not installed or failed to import.")
    raise

# Standard imports that should exist in deployed environment
import time

# Defensive imports with helpful messages (these will raise if not installed)
try:
    from ultralytics import YOLO
except Exception:
    # Allow app to start and show error in UI at runtime
    YOLO = None

try:
    import cv2
except Exception:
    cv2 = None

try:
    import numpy as np
except Exception:
    np = None

# Import our modules (these are local project modules)
try:
    import src.config as config
    from src.detection_engine import DetectionEngine
    from src.video_stream import VideoStream, create_video_stream
    from src.utils import (
        draw_detections,
        draw_fps,
        draw_detection_count,
        convert_bgr_to_rgb
    )
except Exception as e:
    # If project module imports fail, show a clear error in the UI later via the main wrapper
    # Re-raise after main wrapper displays a helpful message.
    # Save the exception to re-raise later inside the Streamlit UI.
    module_import_error = e
else:
    module_import_error = None


# Page configuration
st.set_page_config(
    page_title="Real-Time Object Detection",
    page_icon="🎯",
    layout="wide"
)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'detection_engine' not in st.session_state:
        st.session_state.detection_engine = None

    if 'video_stream' not in st.session_state:
        st.session_state.video_stream = None

    if 'is_running' not in st.session_state:
        st.session_state.is_running = False

    if 'last_detections' not in st.session_state:
        st.session_state.last_detections = []

    if 'frame_placeholder' not in st.session_state:
        st.session_state.frame_placeholder = None

    if 'frame_skip_counter' not in st.session_state:
        st.session_state.frame_skip_counter = 0

    if 'target_fps' not in st.session_state:
        st.session_state.target_fps = getattr(config, "DEFAULT_TARGET_FPS", 10)

    if 'last_frame_time' not in st.session_state:
        st.session_state.last_frame_time = time.time()


def load_detection_engine():
    """Load the YOLOv8 detection engine lazily and show spinner in UI."""
    if module_import_error is not None:
        # If we failed importing local modules, raise here so main() shows it
        raise module_import_error

    if st.session_state.detection_engine is None:
        try:
            with st.spinner("Loading YOLOv8 model... This may take a moment on first run."):
                st.session_state.detection_engine = DetectionEngine()
            st.success("Model loaded successfully!")
        except Exception as e:
            st.error(f"Failed to load model: {str(e)}")
            st.stop()

    return st.session_state.detection_engine


def run_capture_mode():
    """
    Webcam Capture Mode: Snapshot-based detection with zero lag.
    Shows webcam preview, captures single frame on button click, runs YOLO once.
    """
    st.header("📸 Webcam Capture Mode")
    st.markdown("**Zero-lag snapshot detection** — Click Capture to detect objects in a single frame.")

    # Show detection settings in sidebar for capture mode too
    st.sidebar.header("⚙️ Detection Settings")

    # Confidence threshold slider
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=getattr(config, "CONF_THRESHOLD", 0.5),
        step=0.05,
        help="Minimum confidence score for detections (higher = fewer but more confident detections)"
    )

    # IoU threshold slider
    iou_threshold = st.sidebar.slider(
        "IoU Threshold (NMS)",
        min_value=0.0,
        max_value=1.0,
        value=getattr(config, "IOU_THRESHOLD", 0.45),
        step=0.05,
        help="IoU threshold for Non-Maximum Suppression (higher = more overlapping boxes allowed)"
    )

    # Load detection engine (lazy)
    detection_engine = load_detection_engine()

    # Class filtering
    all_classes = detection_engine.get_class_names()
    selected_classes = st.sidebar.multiselect(
        "Filter Classes (leave empty for all)",
        options=all_classes,
        default=None,
        help="Select specific classes to detect. Leave empty to detect all classes."
    )

    # Update config with selected classes and thresholds
    if selected_classes:
        config.ALLOWED_CLASSES = selected_classes
    else:
        config.ALLOWED_CLASSES = None

    config.CONF_THRESHOLD = conf_threshold
    config.IOU_THRESHOLD = iou_threshold

    # Show webcam preview using st.camera_input (client-side capture)
    img = st.camera_input("Webcam Preview", label_visibility="visible")

    # Capture button
    capture_button = st.button("📸 Capture Image", type="primary", use_container_width=True)

    # Handle capture
    if capture_button:
        if img is None:
            st.warning("⚠️ Please allow camera access to capture images.")
            return

        if cv2 is None or np is None:
            st.error("❌ OpenCV or NumPy is not available on the server. Cannot decode camera input.")
            return

        # Convert Streamlit camera input to OpenCV format
        img_bytes = img.getvalue()
        np_img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

        if np_img is None:
            st.error("❌ Failed to decode image. Please try again.")
            return

        # Resize if needed (for consistency with other modes)
        frame_w = getattr(config, "DEFAULT_FRAME_WIDTH", 640)
        frame_h = getattr(config, "DEFAULT_FRAME_HEIGHT", 480)
        if np_img.shape[1] != frame_w or np_img.shape[0] != frame_h:
            np_img = cv2.resize(np_img, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)

        # Run YOLO detection ONCE (this is the only inference call)
        with st.spinner("🔍 Detecting objects..."):
            detections = detection_engine.detect(np_img)

        # Draw detections on frame
        annotated = draw_detections(np_img, detections)

        # Convert BGR to RGB for Streamlit
        rgb = convert_bgr_to_rgb(annotated)

        # Display annotated image
        st.subheader("📊 Detection Result")
        st.image(rgb, caption="Annotated Image with Detections", use_container_width=True)

        # Display detection list
        st.subheader("🔍 Detections")
        if len(detections) == 0:
            st.info("No objects detected. Try adjusting the confidence threshold or capturing a different image.")
        else:
            # Create a summary
            detection_summary = {}
            for det in detections:
                class_name = det[6]  # class_name is at index 6
                if class_name not in detection_summary:
                    detection_summary[class_name] = []
                detection_summary[class_name].append(det[4])  # confidence

            # Display summary with counts and average confidence
            for class_name in sorted(detection_summary.keys()):
                confidences = detection_summary[class_name]
                avg_conf = sum(confidences) / len(confidences)
                st.write(f"• **{class_name}** – {len(confidences)} detected (avg confidence: {avg_conf:.1%})")

            # Detailed list in expander
            with st.expander("View detailed detections"):
                for i, det in enumerate(detections, 1):
                    xmin, ymin, xmax, ymax, confidence, class_id, class_name = det
                    st.text(f"{i}. {class_name}: {confidence:.1%} at ({xmin}, {ymin}) - ({xmax}, {ymax})")


def update_video_frame(frame_placeholder):
    """
    Regular function that updates the video frame.
    Note: Streamlit's non-standard decorator was removed for compatibility.
    This function is called from main() when streaming is active.
    """
    if not st.session_state.is_running or not st.session_state.video_stream:
        return

    if not st.session_state.video_stream.is_opened():
        return

    # Get current settings
    target_fps = st.session_state.get('target_fps', getattr(config, "DEFAULT_TARGET_FPS", 10))
    target_frame_time = 1.0 / target_fps  # Target time per frame in seconds

    # Simple timing-based throttling
    now = time.time()
    elapsed = now - st.session_state.last_frame_time

    # Only process if enough time has passed (respects target FPS)
    if elapsed < target_frame_time:
        return  # Skip this cycle to maintain target FPS

    # Update last frame time
    st.session_state.last_frame_time = now

    # Read frame - FAST, non-blocking
    ret, frame = st.session_state.video_stream.read_frame()

    if not ret or frame is None:
        return

    # Resize frame to configured resolution for faster inference/drawing
    frame_w = getattr(config, "DEFAULT_FRAME_WIDTH", 640)
    frame_h = getattr(config, "DEFAULT_FRAME_HEIGHT", 480)
    frame = cv2.resize(frame, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)

    # Get detection engine
    detection_engine = st.session_state.detection_engine

    # Get frame skip setting
    frame_skip = st.session_state.get('current_frame_skip', 0)

    # Initialize detections - use empty list if no detections yet (shows raw video first)
    detections = []

    if detection_engine is not None:
        # Frame skipping logic:
        # 0 = detect on every frame
        # 1 = detect on every 2nd frame
        # 2 = detect on every 3rd frame
        should_detect = (st.session_state.frame_skip_counter % (frame_skip + 1)) == 0

        if should_detect:
            # Perform detection
            try:
                detections = detection_engine.detect(frame)
                st.session_state.last_detections_for_drawing = detections
                st.session_state.last_detections = detections[:10]
            except Exception:
                # If detection fails, reuse last known detections
                detections = st.session_state.get('last_detections_for_drawing', [])
        else:
            # Use last detections (frame skip active)
            detections = st.session_state.get('last_detections_for_drawing', [])

        # Increment counter
        st.session_state.frame_skip_counter += 1
    else:
        # No detection engine - just show raw video
        st.session_state.frame_skip_counter += 1

    # Draw detections on frame (empty list = no detections, shows raw video)
    frame_with_detections = draw_detections(frame, detections)

    # Get FPS
    fps = st.session_state.video_stream.get_fps()

    # Draw FPS and detection count
    frame_with_detections = draw_fps(frame_with_detections, fps)
    frame_with_detections = draw_detection_count(frame_with_detections, len(detections))

    # Convert to RGB
    frame_rgb = convert_bgr_to_rgb(frame_with_detections)

    # Update frame ONCE - no double updates, no blinking!
    frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)


def main():
    """Main application function."""
    # If modules failed to import earlier, show a clear message and stop
    if module_import_error is not None:
        st.title("🚨 Startup Error")
        st.error("Failed to import project modules. See details below.")
        st.text(str(module_import_error))
        st.text(traceback.format_exc())
        return

    # Initialize session state
    initialize_session_state()

    # Title and description
    st.title("🎯 Real-Time Object Detection System")

    # Get device info from detection engine (if loaded)
    device_info = "cpu"  # Default
    if st.session_state.detection_engine is not None:
        device_info = st.session_state.detection_engine.get_device_name()

    st.markdown(f"""
    **Object Detection using YOLOv8n**

    This application performs real-time object detection using the YOLOv8 nano model.
    **Running on device:** {device_info.upper()}
    """)

    # Mode selection - MUST be first in sidebar
    st.sidebar.header("🎯 Detection Mode")
    mode = st.sidebar.radio(
        "Select Mode:",
        ["Webcam Capture", "Video File Stream", "Real-Time Webcam (High CPU)"],
        index=0,  # Default to Webcam Capture
        help="Webcam Capture: Zero-lag snapshot detection. Video File: Stream from file. Real-Time: Continuous webcam detection (CPU intensive)."
    )

    # Route to appropriate mode
    if mode == "Webcam Capture":
        run_capture_mode()
        return  # Exit early - don't run real-time code

    # Sidebar controls (only shown for streaming modes)
    st.sidebar.header("⚙️ Detection Settings")

    # Confidence threshold slider
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=getattr(config, "CONF_THRESHOLD", 0.5),
        step=0.05,
        help="Minimum confidence score for detections (higher = fewer but more confident detections)"
    )

    # IoU threshold slider
    iou_threshold = st.sidebar.slider(
        "IoU Threshold (NMS)",
        min_value=0.0,
        max_value=1.0,
        value=getattr(config, "IOU_THRESHOLD", 0.45),
        step=0.05,
        help="IoU threshold for Non-Maximum Suppression (higher = more overlapping boxes allowed)"
    )

    # Load detection engine to get class names
    detection_engine = load_detection_engine()

    # Class filtering
    all_classes = detection_engine.get_class_names()
    selected_classes = st.sidebar.multiselect(
        "Filter Classes (leave empty for all)",
        options=all_classes,
        default=None,
        help="Select specific classes to detect. Leave empty to detect all classes."
    )

    # Update config with selected classes
    if selected_classes:
        config.ALLOWED_CLASSES = selected_classes
    else:
        config.ALLOWED_CLASSES = None

    # Update thresholds in config
    config.CONF_THRESHOLD = conf_threshold
    config.IOU_THRESHOLD = iou_threshold

    # Performance settings
    st.sidebar.header("⚡ Performance")

    # Display device information
    if st.session_state.detection_engine is not None:
        device_name = st.session_state.detection_engine.get_device_name()
        device_display = "MPS (Apple Silicon)" if device_name == "mps" else "CPU"
        st.sidebar.markdown(f"**Inference device:** {device_display}")
        st.sidebar.caption(f"Using {device_name.upper()} for detection")

    st.sidebar.caption("💡 **Tip:** Adjust FPS and frame skip for optimal performance")
    target_fps = st.sidebar.slider(
        "Target FPS (approx)",
        min_value=getattr(config, "MIN_TARGET_FPS", 5),
        max_value=getattr(config, "MAX_TARGET_FPS", 15),
        value=min(getattr(config, "MAX_TARGET_FPS", 15), max(getattr(config, "MIN_TARGET_FPS", 5), st.session_state.target_fps)),
        step=1,
        help=f"Target frames per second ({getattr(config, 'MIN_TARGET_FPS', 5)}-{getattr(config, 'MAX_TARGET_FPS', 15)}). Lower values reduce CPU/GPU load."
    )
    st.session_state.target_fps = target_fps

    frame_skip = st.sidebar.slider(
        "Frame Skip",
        min_value=0,
        max_value=getattr(config, "MAX_FRAME_SKIP", 2),
        value=1,
        step=1,
        help="Skip N frames between detections: 0 = every frame, 1 = every 2nd frame, 2 = every 3rd frame. Reduces detection load while keeping smooth video."
    )

    # Input source selection (only for streaming modes)
    st.sidebar.header("📹 Video Source")

    if mode == "Video File Stream":
        input_mode = "Video File"
        video_path = st.sidebar.text_input(
            "Video file path:",
            placeholder="Enter path to video file (e.g., sample_videos/demo.mp4)",
            help="Enter the full path to your video file"
        )
    else:  # Real-Time Webcam mode
        input_mode = "Webcam"
        video_path = None
        st.sidebar.info("⚠️ **High CPU Usage**\n\nReal-time mode continuously processes frames. Use Webcam Capture mode for better performance.")

    # Main area - simplified layout to minimize re-rendering
    st.header("📺 Detection View")

    # Control buttons - use fixed keys to prevent re-creation
    button_col1, button_col2 = st.columns([1, 1])

    with button_col1:
        start_button = st.button("▶️ Start Detection", type="primary", use_container_width=True, key="start_btn")

    with button_col2:
        stop_button = st.button("⏹️ Stop Detection", use_container_width=True, key="stop_btn")

    # Status message placeholder
    status_placeholder = st.empty()

    # Video frame placeholder - this is the only thing that updates frequently
    frame_placeholder = st.empty()

    # Handle button clicks
    if start_button:
        # Reset frame counter and timing for immediate start
        st.session_state.frame_skip_counter = 0
        st.session_state.last_detections_for_drawing = []
        st.session_state.last_frame_time = time.time()  # Reset timing for FPS throttling
        st.session_state.is_running = True

        # Initialize video stream
        try:
            if input_mode == "Webcam":
                st.session_state.video_stream = create_video_stream()
            else:
                if not video_path:
                    status_placeholder.error("Please enter a video file path!")
                    st.session_state.is_running = False
                else:
                    st.session_state.video_stream = create_video_stream(source=video_path)

            # Open video stream
            if st.session_state.video_stream and st.session_state.video_stream.open():
                status_placeholder.success(f"✅ Video source opened: {input_mode}")
            else:
                status_placeholder.error(f"❌ Failed to open video source: {input_mode}")
                if input_mode == "Webcam":
                    status_placeholder.info("💡 Make sure your webcam is connected and not being used by another application.")
                else:
                    status_placeholder.info("💡 Please check that the video file path is correct and the file exists.")
                st.session_state.is_running = False

        except Exception as e:
            status_placeholder.error(f"Error initializing video stream: {str(e)}")
            st.session_state.is_running = False

    if stop_button:
        if st.session_state.video_stream:
            st.session_state.video_stream.release()
        st.session_state.is_running = False
        st.session_state.video_stream = None
        st.session_state.frame_skip_counter = 0
        status_placeholder.info("⏹️ Detection stopped.")

    # Store current frame skip setting for updater
    st.session_state.current_frame_skip = frame_skip

    # Call update function once per Streamlit run when running
    if st.session_state.is_running:
        if st.session_state.video_stream and st.session_state.video_stream.is_opened():
            # Call the updater: it will decide whether to process based on timing
            update_video_frame(frame_placeholder)
        else:
            status_placeholder.error("❌ Video stream is not opened. Please click 'Start Detection'.")
            st.session_state.is_running = False
    else:
        # Clear frame when stopped
        frame_placeholder.empty()

    # Display detection statistics
    if st.session_state.last_detections:
        st.sidebar.header("📊 Recent Detections")

        # Create a summary of detections
        detection_summary = {}
        for det in st.session_state.last_detections:
            class_name = det[6]  # class_name is at index 6
            if class_name not in detection_summary:
                detection_summary[class_name] = 0
            detection_summary[class_name] += 1

        # Display summary
        for class_name, count in sorted(detection_summary.items()):
            st.sidebar.metric(class_name, count)

        # Show detailed list
        with st.sidebar.expander("View all detections"):
            for i, det in enumerate(st.session_state.last_detections):
                xmin, ymin, xmax, ymax, confidence, class_id, class_name = det
                st.text(f"{i+1}. {class_name}: {confidence:.1%}")

    # Footer information
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.info("""
    **CPU-Only Detection**

    This system is optimized for CPU-only inference.
    Performance depends on your hardware.

    **Expected FPS:** 8-15 FPS on typical laptops
    """)

    # Cleanup on app close (best-effort)
    if st.session_state.video_stream:
        # Resources will be released when the app stops/restarts
        pass


# Run main inside try/except to show runtime issues in the Streamlit UI (prevents blank page)
try:
    main()
except Exception as exc:
    st.title("🚨 Application Error")
    st.error("An unexpected error occurred while running the app.")
    st.text(str(exc))
    st.text(traceback.format_exc())
    # Optionally re-raise during development (comment out in deployment)
    # raise
