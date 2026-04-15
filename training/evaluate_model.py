import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the YOLO model on a validation dataset."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "runs" / "detect" / "color_detection_model" / "weights" / "best.pt",
        help="Path to YOLO model weights.",
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "dataset_structure_example" / "data.yaml",
        help="Path to the YOLO data configuration file.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size to use during evaluation.",
    )
    return parser.parse_args()


def evaluate_yolo_model(weights: Path, data_config: Path, batch: int):
    if not weights.exists():
        raise FileNotFoundError(f"YOLO weights not found: {weights}")
    if not data_config.exists():
        raise FileNotFoundError(f"Data config not found: {data_config}")

    print(f"Evaluating YOLO model: {weights}")
    print(f"Using data config: {data_config}")

    model = YOLO(str(weights))
    metrics = model.val(data=str(data_config), batch=batch)

    print("=== Evaluation Results ===")
    print(metrics)
    return metrics


def main():
    args = parse_args()
    evaluate_yolo_model(args.weights, args.data_config, args.batch)


if __name__ == "__main__":
    main()
