import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "lux_model_physics.pth"
DEFAULT_SAMPLE_IMAGES = PROJECT_ROOT / "data" / "sample" / "images"
DEFAULT_SAMPLE_LABELS = PROJECT_ROOT / "data" / "sample" / "labels"

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
        description="Run a lux prediction test with the physics-informed model."
    )
    parser.add_argument(
        "--image-path",
        type=Path,
        default=None,
        help="Path to a test image. If omitted, the first sample image is used.",
    )
    parser.add_argument(
        "--label-path",
        type=Path,
        default=None,
        help="Optional label file for polygon cropping.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the physics-informed lux model weights.",
    )
    return parser.parse_args()


def find_default_image() -> Path:
    if not DEFAULT_SAMPLE_IMAGES.exists():
        raise FileNotFoundError(
            f"No sample images found at {DEFAULT_SAMPLE_IMAGES}. Provide --image-path."
        )

    candidates = sorted(
        [p for p in DEFAULT_SAMPLE_IMAGES.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )
    if not candidates:
        raise FileNotFoundError(f"No image files found in {DEFAULT_SAMPLE_IMAGES}.")
    return candidates[0]


def load_model(model_path: Path, device: torch.device):
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    model = LuxModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def crop_from_polygon(image_path: Path, label_path: Path) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    w, h = image.size

    if not label_path.exists():
        print("⚠️ No label found, using full image")
        return image

    with label_path.open("r", encoding="utf-8") as f:
        line = f.readline().strip()

    if line == "":
        print("⚠️ Empty label, using full image")
        return image

    parts = list(map(float, line.split()))
    if len(parts) < 7:
        print("⚠️ Invalid polygon format, using full image")
        return image

    coords = parts[1:]
    xs = [int(x * w) for x in coords[0::2]]
    ys = [int(y * h) for y in coords[1::2]]

    xmin, xmax = max(min(xs), 0), min(max(xs), w)
    ymin, ymax = max(min(ys), 0), min(max(ys), h)

    if xmin >= xmax or ymin >= ymax:
        print("⚠️ Bad crop coordinates, using full image")
        return image

    return image.crop((xmin, ymin, xmax, ymax))


def compute_brightness(pil_crop: Image.Image) -> torch.Tensor:
    gray = np.array(pil_crop.convert("L"), dtype=np.float32)
    mean_intensity = gray.mean() / 255.0
    normalized_brightness = mean_intensity / EXPOSURE_FACTOR
    return torch.tensor([[normalized_brightness]], dtype=torch.float32)


def predict(image_path: Path, label_path: Path | None, model, device: torch.device) -> float:
    if label_path:
        crop = crop_from_polygon(image_path, label_path)
    else:
        crop = Image.open(image_path).convert("RGB")

    brightness = compute_brightness(crop)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    img_tensor = transform(crop).unsqueeze(0).to(device)
    brightness = brightness.to(device)

    with torch.no_grad():
        pred = model(img_tensor, brightness)

    return float(pred.item())


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model_path, device)

    image_path = args.image_path or find_default_image()
    label_path = args.label_path
    if label_path is None and DEFAULT_SAMPLE_LABELS.exists():
        candidate_label = DEFAULT_SAMPLE_LABELS / image_path.with_suffix(".txt").name
        label_path = candidate_label if candidate_label.exists() else None

    lux = predict(image_path, label_path, model, device)
    print(f"💡 Predicted Lux for {image_path}: {lux:.2f}")


if __name__ == "__main__":
    main()
