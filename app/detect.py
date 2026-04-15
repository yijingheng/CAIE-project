import cv2
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.models as models
import streamlit as st
from ultralytics import YOLO

from intensity import preprocess
from llm_module import interpret_led

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
YOLO_WEIGHTS = PROJECT_ROOT / "runs" / "detect" / "color_detection_model" / "weights" / "best.pt"
LUX_WEIGHTS = PROJECT_ROOT / "lux_model_physics.pth"


@st.cache_resource
def load_yolo(weights_path: Path = YOLO_WEIGHTS):
    if not weights_path.exists():
        raise FileNotFoundError(f"YOLO weights not found: {weights_path}")
    return YOLO(str(weights_path))


@st.cache_resource
def load_lux(model_path: Path = LUX_WEIGHTS):
    if not model_path.exists():
        raise FileNotFoundError(f"Lux model not found: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class LuxModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn = models.resnet18(weights=None)
            self.cnn.fc = nn.Identity()

            self.brightness_fc = nn.Sequential(
                nn.Linear(1, 16),
                nn.ReLU(),
            )

            self.final = nn.Sequential(
                nn.Linear(512 + 16, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )

        def forward(self, img, brightness):
            img_feat = self.cnn(img)
            bright_feat = self.brightness_fc(brightness)
            combined = torch.cat([img_feat, bright_feat], dim=1)
            return self.final(combined)

    model = LuxModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model, device


def predict(frame, yolo_model, lux_model, device, conf_threshold, exposure_factor):
    results = yolo_model(frame, conf=conf_threshold)[0]

    if len(results.boxes) == 0:
        return frame, "No LED detected"

    box = results.boxes.xyxy[0].cpu().numpy().astype(int)
    x1, y1, x2, y2 = box

    cls_id = int(results.boxes.cls[0])
    color = yolo_model.names[cls_id]

    crop = frame[y1:y2, x1:x2]

    img_tensor, brightness_tensor = preprocess(crop, exposure_factor, device)

    with torch.no_grad():
        lux = lux_model(img_tensor, brightness_tensor).item()

    initial_interpretation = interpret_led(color, lux)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        frame,
        f"{color} | {lux:.1f} lux",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    result_text = f"""
Color: {color}
Lux: {lux:.2f}

=== Assessment ===
{initial_interpretation}
"""

    return frame, result_text
