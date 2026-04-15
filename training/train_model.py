import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a YOLOv8 model for LED detection."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path.cwd() / "yolov8n.pt",
        help="Path to the YOLOv8 base weights file.",
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        default=Path.cwd() / "data" / "dataset_structure_example" / "data.yaml",
        help="Path to the YOLO data configuration file.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size for training.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Training batch size.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="color_detection_model",
        help="Experiment name / output folder prefix.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.weights.exists():
        raise FileNotFoundError(f"YOLO weights not found: {args.weights}")
    if not args.data_config.exists():
        raise FileNotFoundError(f"Data config not found: {args.data_config}")

    print(f"Training YOLO model with weights: {args.weights}")
    print(f"Using data config: {args.data_config}")

    model = YOLO(str(args.weights))
    model.train(
        data=str(args.data_config),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
    )

    print("Training complete. Running validation...")
    metrics = model.val()
    print(metrics)


if __name__ == "__main__":
    main()
