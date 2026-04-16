import cv2
from pathlib import Path
import streamlit as st
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent


def find_existing_path(candidates, file_label):
    for path in candidates:
        if path.exists():
            return path

    checked_paths = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"{file_label} not found.\nChecked these locations:\n{checked_paths}"
    )


YOLO_CANDIDATES = [
    PROJECT_ROOT / "runs" / "detect" / "color_detection_model" / "weights" / "best.pt",
    PROJECT_ROOT / "models" / "best.pt",
    PROJECT_ROOT / "src" / "models" / "best.pt",
    BASE_DIR / "models" / "best.pt",
]


@st.cache_resource
def load_yolo():
    weights_path = find_existing_path(YOLO_CANDIDATES, "YOLO weights")
    return YOLO(str(weights_path))


def detect_color(frame, yolo_model, conf_threshold=0.5):
    output_frame = frame.copy()
    results = yolo_model(output_frame, conf=conf_threshold)[0]

    if len(results.boxes) == 0:
        return None

    box = results.boxes.xyxy[0].cpu().numpy().astype(int)
    x1, y1, x2, y2 = box

    h, w = output_frame.shape[:2]
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))

    if x2 <= x1 or y2 <= y1:
        return None

    cls_id = int(results.boxes.cls[0].item())
    color = yolo_model.names[cls_id]

    crop = output_frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    return {
        "color": color,
        "box": (x1, y1, x2, y2),
        "crop": crop,
    }