import cv2
import numpy as np
import torch


def preprocess(img: np.ndarray, exposure_factor: float, device: torch.device):
    img_resized = cv2.resize(img, (224, 224))
    img_norm = img_resized / 255.0
    img_transposed = np.transpose(img_norm, (2, 0, 1))

    img_tensor = torch.tensor(img_transposed, dtype=torch.float32).unsqueeze(0)

    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    mean_intensity = gray.mean() / 255.0

    normalized_brightness = mean_intensity / exposure_factor
    brightness_tensor = torch.tensor([[normalized_brightness]], dtype=torch.float32)

    return img_tensor.to(device), brightness_tensor.to(device)
