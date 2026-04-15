import argparse
from pathlib import Path
import re
import pandas as pd
from datetime import datetime, timedelta
import cv2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Match extracted frame images to lux sensor readings."
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path.cwd() / "frames",
        help="Directory containing extracted frame images.",
    )
    parser.add_argument(
        "--lux-file",
        type=Path,
        default=Path.cwd() / "lux_data.txt",
        help="Raw lux sensor data file.",
    )
    parser.add_argument(
        "--video-path",
        type=Path,
        required=True,
        help="Path to the source video file.",
    )
    parser.add_argument(
        "--video-date",
        type=str,
        required=True,
        help="Video end date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end-time",
        type=str,
        required=True,
        help="Video end time in HH:MM:SS format.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional output CSV file path.",
    )
    return parser.parse_args()


def get_video_metadata(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise ValueError(f"Invalid FPS extracted from video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, total_frames


def get_frame_number(filename: str):
    match = re.search(r"frame_(\d+)", filename)
    return int(match.group(1)) if match else None


def load_lux_data(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"Lux file not found: {file_path}")

    timestamps = []
    lux_values = []

    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                ts_str, lux = line.strip().split("|")
                ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")
                timestamps.append(ts)
                lux_values.append(float(lux))
            except ValueError:
                continue

    df = pd.DataFrame({"timestamp": timestamps, "lux": lux_values})
    df["timestamp"] = df["timestamp"].dt.round("1ms")
    return df.sort_values("timestamp").reset_index(drop=True)


def round_to_ms(dt: datetime) -> datetime:
    return (dt + timedelta(microseconds=500)).replace(
        microsecond=(dt.microsecond // 1000) * 1000
    )


def match_images_to_lux(image_dir: Path, lux_df: pd.DataFrame, start_time: datetime, fps: float) -> pd.DataFrame:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    results = []

    for img_file in sorted(image_dir.iterdir()):
        if img_file.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        frame_id = get_frame_number(img_file.name)
        if frame_id is None:
            continue

        elapsed_sec = frame_id / fps
        img_time = start_time + timedelta(seconds=elapsed_sec)
        img_time = round_to_ms(img_time)

        lux_df["time_diff"] = (lux_df["timestamp"] - img_time).abs()
        nearest = lux_df.loc[lux_df["time_diff"].idxmin()]

        results.append(
            {
                "image": img_file.name,
                "frame_id": frame_id,
                "image_time": img_time,
                "matched_lux_time": nearest["timestamp"],
                "lux": nearest["lux"],
                "time_diff_ms": nearest["time_diff"].total_seconds() * 1000,
            }
        )

    return pd.DataFrame(results)


def parse_end_datetime(end_time_str: str, video_date: str) -> datetime:
    return datetime.strptime(f"{video_date} {end_time_str}", "%Y-%m-%d %H:%M:%S")


def main():
    args = parse_args()
    fps, total_frames = get_video_metadata(args.video_path)

    start_time = parse_end_datetime(args.end_time, args.video_date) - timedelta(seconds=total_frames / fps)

    print(f"FPS: {fps}")
    print(f"Video duration: {total_frames / fps:.2f}s")
    print(f"Computed start time: {start_time}")

    lux_df = load_lux_data(args.lux_file)
    result_df = match_images_to_lux(args.image_dir, lux_df, start_time, fps)

    output_path = args.output_csv or args.image_dir / "image_lux_matches.csv"
    result_df.to_csv(output_path, index=False)
    print(f"Done! Results saved to {output_path}")


if __name__ == "__main__":
    main()
