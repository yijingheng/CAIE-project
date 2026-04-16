import cv2
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import streamlit as st

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


LUX_CANDIDATES = [
    PROJECT_ROOT / "lux_model_physics.pth",
    PROJECT_ROOT / "models" / "lux_model_physics.pth",
    PROJECT_ROOT / "src" / "models" / "lux_model_physics.pth",
    BASE_DIR / "models" / "lux_model_physics.pth",
]


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


@st.cache_resource
def load_lux_model():
    model_path = find_existing_path(LUX_CANDIDATES, "Lux model")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LuxModel().to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    return model, device


def preprocess_crop(crop: np.ndarray, exposure_factor: float, device: torch.device):
    if crop is None or crop.size == 0:
        raise ValueError("Input crop is empty.")

    img_resized = cv2.resize(crop, (224, 224))
    img_norm = img_resized.astype(np.float32) / 255.0
    img_transposed = np.transpose(img_norm, (2, 0, 1))

    img_tensor = torch.tensor(
        img_transposed,
        dtype=torch.float32
    ).unsqueeze(0)

    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    mean_intensity = gray.mean() / 255.0

    exposure_factor = max(exposure_factor, 1e-6)
    normalized_brightness = mean_intensity / exposure_factor

    brightness_tensor = torch.tensor(
        [[normalized_brightness]],
        dtype=torch.float32
    )

    return img_tensor.to(device), brightness_tensor.to(device)


def predict_lux_from_crop(crop, lux_model, device, exposure_factor):
    img_tensor, brightness_tensor = preprocess_crop(
        crop=crop,
        exposure_factor=exposure_factor,
        device=device
    )

    with torch.no_grad():
        lux = lux_model(img_tensor, brightness_tensor).item()

    return lux