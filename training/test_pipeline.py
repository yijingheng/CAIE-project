import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from ultralytics import YOLO

# =========================
# PROJECT ROOT / IMPORT FIX
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.llm_module import interpret_led  # noqa: E402

# =========================
# CAMERA PARAMETERS
# =========================
F_NUMBER = 1.9
SHUTTER = 1 / 60
EXPOSURE_FACTOR = SHUTTER / (F_NUMBER ** 2)

# =========================
# PATHS
# =========================
YOLO_WEIGHTS = PROJECT_ROOT / "runs" / "detect" / "color_detection_model" / "weights" / "best.pt"
LUX_MODEL_PATH = PROJECT_ROOT / "models" / "lux_model_physics.pth"
DEFAULT_TEST_IMAGE = PROJECT_ROOT /"data" / "sample" / "images"/"frame_001300_jpg.rf.pnQf6eDGzHXmAeRxdO1w.jpg"
OUTPUT_PATH = PROJECT_ROOT / "output.jpg"

# =========================
# YOLO MODEL
# =========================
if not YOLO_WEIGHTS.exists():
    raise FileNotFoundError(
        f"YOLO weights not found: {YOLO_WEIGHTS}\n"
        f"Update YOLO_WEIGHTS to your actual best.pt location."
    )

yolo_model = YOLO(str(YOLO_WEIGHTS))

# =========================
# LUX MODEL
# =========================
class LuxModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = models.resnet18(weights=None)
        self.cnn.fc = nn.Identity()

        self.brightness_fc = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU()
        )

        self.final = nn.Sequential(
            nn.Linear(512 + 16, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, img, brightness):
        img_feat = self.cnn(img)
        bright_feat = self.brightness_fc(brightness)
        combined = torch.cat([img_feat, bright_feat], dim=1)
        return self.final(combined)

# =========================
# LOAD LUX MODEL
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not LUX_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Lux model weights not found: {LUX_MODEL_PATH}\n"
        f"Update LUX_MODEL_PATH to your actual lux_model_physics.pth location."
    )

model = LuxModel().to(device)
model.load_state_dict(torch.load(LUX_MODEL_PATH, map_location=device))
model.eval()

print("✅ Physics model loaded correctly")

# =========================
# PREPROCESS
# =========================
def preprocess(img):
    if img is None or img.size == 0:
        raise ValueError("Received an empty image crop for preprocessing.")

    img_resized = cv2.resize(img, (224, 224))
    img_norm = img_resized.astype(np.float32) / 255.0
    img_transposed = np.transpose(img_norm, (2, 0, 1))
    img_tensor = torch.tensor(img_transposed, dtype=torch.float32).unsqueeze(0)

    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    mean_intensity = gray.mean() / 255.0

    normalized_brightness = mean_intensity / EXPOSURE_FACTOR
    normalized_brightness = max(0.0, min(float(normalized_brightness), 5.0))

    print(f"Brightness: {normalized_brightness:.3f}")

    brightness_tensor = torch.tensor([[normalized_brightness]], dtype=torch.float32)

    return img_tensor.to(device), brightness_tensor.to(device)

# =========================
# PIPELINE
# =========================
def predict(image_path):
    image_path = Path(image_path)

    if not image_path.exists():
        print(f"❌ Image file not found: {image_path}")
        return

    img = cv2.imread(str(image_path))
    if img is None:
        print(f"❌ Failed to read image: {image_path}")
        return

    results = yolo_model(img)[0]

    if results.boxes is None or len(results.boxes) == 0:
        print("❌ No LED detected")
        return

    # Use first detection
    box = results.boxes.xyxy[0].cpu().numpy().astype(int)
    x1, y1, x2, y2 = box

    # Clamp bounding box to image size
    h, w = img.shape[:2]
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        print("❌ Invalid bounding box returned by YOLO")
        return

    # Get color label
    cls_id = int(results.boxes.cls[0].item())
    color_raw = str(yolo_model.names[cls_id]).strip().lower()

    color_aliases = {
        "gween": "green",
    }
    color = color_aliases.get(color_raw, color_raw)

    # Crop LED
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        print("❌ Empty crop after applying YOLO bounding box")
        return

    # Preprocess
    img_tensor, brightness_tensor = preprocess(crop)

    # Predict lux
    with torch.no_grad():
        lux = model(img_tensor, brightness_tensor).item()

    print(f"💡 Predicted Lux: {lux:.2f}")
    print(f"🎨 Detected Color: {color_raw} -> normalized: {color}")

    # LLM explanation
    explanation = interpret_led(color, lux)

    print("🧠 Explanation:")
    print(explanation)

    # Draw result
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        img,
        f"{color} | {lux:.1f} lux",
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imwrite(str(OUTPUT_PATH), img)
    print(f"✅ Saved result as {OUTPUT_PATH}")

# =========================
# TEST
# =========================
if __name__ == "__main__":
    predict(DEFAULT_TEST_IMAGE)