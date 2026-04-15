import argparse
from pathlib import Path
import os

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from PIL import Image

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

# =========================
# CAMERA PARAMETERS
# =========================
F_NUMBER = 1.9
SHUTTER = 1 / 60
EXPOSURE_FACTOR = SHUTTER / (F_NUMBER ** 2)

# =========================
# PATH HANDLING (GITHUB SAFE)
# =========================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "sample"
DEFAULT_SAVE_PATH = PROJECT_ROOT / "src" / "models" / "lux_model_physics.pth"

# =========================
# ARGUMENTS
# =========================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--save_path", type=Path, default=DEFAULT_SAVE_PATH)
    return parser.parse_args()

# =========================
# POLYGON → BBOX
# =========================
def get_bbox(label_path, w, h):
    try:
        if not label_path.exists():
            return None

        with open(label_path, "r") as f:
            lines = f.readlines()

        if len(lines) == 0:
            return None

        parts = list(map(float, lines[0].split()))
        coords = parts[1:]

        xs = coords[0::2]
        ys = coords[1::2]

        xmin = int(min(xs) * w)
        xmax = int(max(xs) * w)
        ymin = int(min(ys) * h)
        ymax = int(max(ys) * h)

        return xmin, ymin, xmax, ymax
    except:
        return None

# =========================
# DATASET
# =========================
class LuxDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.img_dir = data_dir / "images"
        self.label_dir = data_dir / "labels"
        self.csv_path = data_dir / "lux_labels.csv"

        self.df = pd.read_csv(self.csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = self.img_dir / row["image"]
        label_path = self.label_dir / (Path(row["image"]).stem + ".txt")

        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        bbox = get_bbox(label_path, w, h)

        # fallback if no label
        if bbox is None:
            crop = image
        else:
            xmin, ymin, xmax, ymax = bbox
            if xmin >= xmax or ymin >= ymax:
                crop = image
            else:
                crop = image.crop((xmin, ymin, xmax, ymax))

        # =========================
        # BRIGHTNESS FEATURE
        # =========================
        gray = np.array(crop.convert("L"), dtype=np.float32)
        mean_intensity = gray.mean() / 255.0
        normalized_brightness = mean_intensity / EXPOSURE_FACTOR

        brightness = torch.tensor([normalized_brightness], dtype=torch.float32)

        if self.transform:
            crop = self.transform(crop)

        lux = torch.tensor(row["lux"], dtype=torch.float32)

        return crop, brightness, lux

# =========================
# MODEL
# =========================
class LuxModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = models.resnet18(weights="IMAGENET1K_V1")
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
# MAIN TRAINING
# =========================
def main():
    args = parse_args()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = LuxDataset(args.data_dir, transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LuxModel().to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # =========================
    # TRAIN LOOP
    # =========================
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        for imgs, brightness, lux in loader:
            imgs = imgs.to(device)
            brightness = brightness.to(device)
            lux = lux.to(device).unsqueeze(1)

            preds = model(imgs, brightness)

            loss = criterion(preds, lux)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch [{epoch+1}/{args.epochs}] | MAE: {avg_loss:.2f} lux")

    # =========================
    # SAVE
    # =========================
    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.save_path)

    print(f"✅ Model saved to {args.save_path}")

# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    main()