from ultralytics import YOLO

# Load a pretrained YOLOv8 model (you can choose n, s, m, l, x)
model = YOLO("yolov8n.pt")

# Train the model
model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    name="color_detection_model"
)

# Validate the model
metrics = model.val()

print(metrics)