"""
Utility functions for the CAIE LED Interpretation System
"""

import torch
import logging
from pathlib import Path

# Model paths
MODEL_DIR = Path(__file__).parent.parent / "models"
YOLO_MODEL_PATH = MODEL_DIR / "yolov8_led.pt"
LUX_MODEL_PATH = MODEL_DIR / "lux_model.pth"

def get_device():
    """Get the available device (CUDA or CPU)"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def load_image(image_path):
    """Common image loading function"""
    from PIL import Image
    return Image.open(image_path)

# Add more utility functions as needed