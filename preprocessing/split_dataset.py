import argparse
import random
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split a YOLO dataset into train and validation folders."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path.cwd() / "data" / "sample",
        help="Directory containing 'images' and 'labels' subfolders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd() / "dataset_split",
        help="Output folder for the train/val split.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of images to include in the training set.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset shuffling.",
    )
    return parser.parse_args()


def copy_files(file_list, image_src, label_src, image_dst, label_dst):
    for img_file in file_list:
        label_file = Path(img_file).with_suffix(".txt").name

        shutil.copy2(image_src / img_file, image_dst / img_file)

        label_path = label_src / label_file
        if label_path.exists():
            shutil.copy2(label_path, label_dst / label_file)


def main():
    args = parse_args()
    random.seed(args.seed)

    images_dir = args.dataset_dir / "images"
    labels_dir = args.dataset_dir / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory layout must include 'images/' and 'labels/'."
        )

    train_images = args.output_dir / "train" / "images"
    train_labels = args.output_dir / "train" / "labels"
    val_images = args.output_dir / "val" / "images"
    val_labels = args.output_dir / "val" / "labels"

    for path in [train_images, train_labels, val_images, val_labels]:
        path.mkdir(parents=True, exist_ok=True)

    images = [f.name for f in images_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if not images:
        raise ValueError(f"No image files found in {images_dir}")

    random.shuffle(images)

    split_idx = int(len(images) * args.train_ratio)
    train_files = images[:split_idx]
    val_files = images[split_idx:]

    copy_files(train_files, images_dir, labels_dir, train_images, train_labels)
    copy_files(val_files, images_dir, labels_dir, val_images, val_labels)

    print(f"Dataset split completed successfully! Output saved to {args.output_dir}")


if __name__ == "__main__":
    main()
