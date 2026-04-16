import os
from ultralytics import YOLO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Build absolute path to data.yaml
DATA_YAML = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data", "dataset_structure_example", "data.yaml"))

print("Script dir:", SCRIPT_DIR)
print("Looking for data.yaml at:", DATA_YAML)
print("Exists:", os.path.exists(DATA_YAML))

# Load a pretrained YOLOv8 model (you can choose n, s, m, l, x)
model = YOLO("yolov8n.pt")

# Train the model
model.train(
    data=DATA_YAML,
    epochs=50,
    imgsz=640,
    batch=16,
    name="color_detection_model"
)

# Validate the model
metrics = model.val()

print(metrics)