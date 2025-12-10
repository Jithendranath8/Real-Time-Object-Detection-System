#!/usr/bin/env python3
"""
Setup Verification Script

This script checks if all required dependencies are installed correctly.
Run this before starting the Streamlit app to ensure everything is set up properly.
"""

import sys
from pathlib import Path

def check_python_version():
    """Check if Python version is 3.10 or higher."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print(f"❌ Python 3.10+ required. Current version: {version.major}.{version.minor}")
        return False
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def get_device():
    """
    Get the best available device for inference.
    Tries MPS (Apple Silicon) first, falls back to CPU.
    """
    try:
        import torch
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"

def check_dependencies():
    """Check if all required packages are installed."""
    required_packages = {
        'ultralytics': 'ultralytics',
        'cv2': 'opencv-python',
        'streamlit': 'streamlit',
        'numpy': 'numpy',
    }
    
    missing_packages = []
    
    for module_name, package_name in required_packages.items():
        try:
            if module_name == 'cv2':
                import cv2
            elif module_name == 'ultralytics':
                from ultralytics import YOLO
            elif module_name == 'streamlit':
                import streamlit
            elif module_name == 'numpy':
                import numpy
            print(f"✅ {package_name} is installed")
        except ImportError:
            print(f"❌ {package_name} is NOT installed")
            missing_packages.append(package_name)
    
    return missing_packages

def check_device():
    """Check which device will be used for inference."""
    print("4. Checking inference device...")
    device = get_device()
    if device == "mps":
        print(f"✅ MPS (Apple Silicon) is available - will use GPU acceleration")
    else:
        print(f"✅ Will use CPU for inference")
    print(f"   Selected device: {device}")
    print()
    return device

def check_project_structure():
    """Check if project structure is correct."""
    project_root = Path(__file__).parent
    required_files = [
        'src/__init__.py',
        'src/config.py',
        'src/detection_engine.py',
        'src/video_stream.py',
        'src/utils.py',
        'src/app_streamlit.py',
        'requirements.txt',
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = project_root / file_path
        if not full_path.exists():
            print(f"❌ Missing: {file_path}")
            missing_files.append(file_path)
        else:
            print(f"✅ Found: {file_path}")
    
    return missing_files

def main():
    """Run all checks."""
    print("=" * 60)
    print("Real-Time Object Detection System - Setup Verification")
    print("=" * 60)
    print()
    
    all_checks_passed = True
    
    # Check Python version
    print("1. Checking Python version...")
    if not check_python_version():
        all_checks_passed = False
    print()
    
    # Check dependencies
    print("2. Checking dependencies...")
    missing = check_dependencies()
    if missing:
        all_checks_passed = False
        print()
        print("⚠️  Missing packages. Install them with:")
        print(f"   pip install {' '.join(missing)}")
    print()
    
    # Check project structure
    print("3. Checking project structure...")
    missing_files = check_project_structure()
    if missing_files:
        all_checks_passed = False
    print()
    
    # Check device
    device = check_device()
    
    # Final result
    print("=" * 60)
    if all_checks_passed:
        print("✅ All checks passed! You're ready to run the app.")
        print()
        print("To start the app, run:")
        print("   streamlit run src/app_streamlit.py")
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        sys.exit(1)
    print("=" * 60)

if __name__ == "__main__":
    main()

