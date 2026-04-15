import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "lux_model_physics.pth"
DEFAULT_IMAGE_PATH = PROJECT_ROOT / "data" / "sample" / "images"

# =========================
# CAMERA PARAMETERS (SAME AS TRAINING)
# =========================
F_NUMBER = 1.9
SHUTTER = 1 / 60
EXPOSURE_FACTOR = SHUTTER / (F_NUMBER ** 2)


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load the physics-informed lux model and run a single example prediction."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the saved lux model weights.",
    )
    parser.add_argument(
        "--image-path",
        type=Path,
        default=None,
        help="Path to a test image. If omitted, the first sample image is used.",
    )
    return parser.parse_args()


def load_model(model_path: Path, device: torch.device):
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    model = LuxModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def predict_lux(image_path: Path, model, device):
    image = Image.open(image_path).convert("RGB")

    gray = np.array(image.convert("L"), dtype=np.float32)
    mean_intensity = gray.mean() / 255.0
    normalized_brightness = mean_intensity / EXPOSURE_FACTOR

    brightness_tensor = torch.tensor([[normalized_brightness]], dtype=torch.float32).to(device)
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(img_tensor, brightness_tensor)

    return pred.item()


def find_default_image() -> Path:
    if not DEFAULT_IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Default sample image directory not found: {DEFAULT_IMAGE_PATH}."
        )

    images = sorted(
        [p for p in DEFAULT_IMAGE_PATH.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )
    if not images:
        raise FileNotFoundError(f"No sample images found in {DEFAULT_IMAGE_PATH}.")
    return images[0]


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model_path, device)

    image_path = args.image_path or find_default_image()
    lux = predict_lux(image_path, model, device)
    print(f"Predicted Lux for {image_path}: {lux:.2f}")