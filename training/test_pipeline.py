import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from llm_module import interpret_led, MODEL_NAME as LLM_MODEL_NAME

# =========================
# CAMERA PARAMETERS
# =========================
F_NUMBER = 1.9
SHUTTER = 1 / 60
EXPOSURE_FACTOR = SHUTTER / (F_NUMBER ** 2)

BASE_DIR = Path(__file__).resolve().parent


# =========================
# YOLO MODEL
# =========================
yolo_weights = BASE_DIR / "runs" / "detect" / "color_detection_model" / "weights" / "best.pt"
yolo_model = YOLO(str(yolo_weights))

# =========================
# LUX MODEL (MATCH TRAINING)
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
# LOAD MODEL
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LuxModel().to(device)
model.load_state_dict(torch.load(BASE_DIR / "lux_model_physics.pth", map_location=device))
model.eval()

print("✅ Physics model loaded correctly")
print(f"✅ LLM model configured: {LLM_MODEL_NAME}")

# =========================
# PREPROCESS
# =========================
def preprocess(img):
    img_resized = cv2.resize(img, (224, 224))
    img_norm = img_resized / 255.0
    img_transposed = np.transpose(img_norm, (2, 0, 1))
    img_tensor = torch.tensor(img_transposed, dtype=torch.float32).unsqueeze(0)

    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    mean_intensity = gray.mean() / 255.0

    normalized_brightness = mean_intensity / EXPOSURE_FACTOR

    # ✅ Clamp
    normalized_brightness = max(0, min(normalized_brightness, 5))

    print(f"Brightness: {normalized_brightness:.3f}")

    brightness_tensor = torch.tensor([[normalized_brightness]], dtype=torch.float32)

    return img_tensor.to(device), brightness_tensor.to(device)

# =========================
# PIPELINE
# =========================
def predict(image_path):
    img = cv2.imread(str(image_path))

    if img is None:
        print(f"❌ Failed to read image: {image_path}")
        return

    results = yolo_model(img)[0]

    if len(results.boxes) == 0:
        print("❌ No LED detected")
        return

    # Get bounding box
    box = results.boxes.xyxy[0].cpu().numpy().astype(int)
    x1, y1, x2, y2 = box

    # ✅ GET COLOR LABEL (IMPORTANT)
    cls_id = int(results.boxes.cls[0].cpu().numpy())
    color_raw = str(yolo_model.names[cls_id]).strip().lower()

    color_aliases = {
        "gween": "green",
    }
    color = color_aliases.get(color_raw, color_raw)

    # Crop LED
    crop = img[y1:y2, x1:x2]

    # Preprocess
    img_tensor, brightness_tensor = preprocess(crop)

    # Predict lux
    with torch.no_grad():
        lux = model(img_tensor, brightness_tensor).item()

    print(f"💡 Predicted Lux: {lux:.2f}")
    print(f"🎨 Detected Color: {color_raw} -> normalized: {color}")

    # LLM
    explanation = interpret_led(color, lux)

    print("🧠 Explanation:")
    print(explanation)

    # Draw result
    cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
    cv2.putText(img, f"{color} | {lux:.1f} lux", (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    output_path = BASE_DIR / "output.jpg"
    cv2.imwrite(str(output_path), img)
    print(f"✅ Saved result as {output_path}")



# =========================
# TEST
# =========================
if __name__ == "__main__":
    test_image = BASE_DIR / "frames" / "frame_001300.jpg"
    predict(test_image)