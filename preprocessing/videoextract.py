import argparse
from pathlib import Path
import cv2
from datetime import datetime, timedelta


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract frames from a video and label them with estimated timestamps."
    )
    parser.add_argument(
        "--video-path",
        type=Path,
        required=True,
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        default=Path.cwd() / "frames",
        help="Directory to save extracted frames.",
    )
    parser.add_argument(
        "--end-time",
        type=str,
        required=True,
        help="Video end time in HH:MM:SS format.",
    )
    parser.add_argument(
        "--video-date",
        type=str,
        default=None,
        help="Optional video date in YYYY-MM-DD format. If omitted, today's date is used.",
    )
    return parser.parse_args()


def parse_end_datetime(end_time_str: str, video_date: str | None) -> datetime:
    if video_date:
        return datetime.strptime(f"{video_date} {end_time_str}", "%Y-%m-%d %H:%M:%S")
    today = datetime.today().date()
    return datetime.strptime(f"{today} {end_time_str}", "%Y-%m-%d %H:%M:%S")


def main():
    args = parse_args()
    if not args.video_path.exists():
        raise FileNotFoundError(f"Video file not found: {args.video_path}")

    args.output_folder.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {args.video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise ValueError("Unable to determine FPS from video.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps

    end_time = parse_end_datetime(args.end_time, args.video_date)
    start_time = end_time - timedelta(seconds=duration_sec)

    print(f"FPS: {fps}")
    print(f"Video duration: {duration_sec:.2f}s")
    print(f"Computed start time: {start_time}")

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        elapsed_sec = frame_id / fps
        real_time = start_time + timedelta(seconds=elapsed_sec)
        label = real_time.strftime("%Y-%m-%d %H:%M:%S")

        cv2.putText(
            frame,
            label,
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        output_path = args.output_folder / f"frame_{frame_id:06d}.jpg"
        cv2.imwrite(str(output_path), frame)
        frame_id += 1

    cap.release()
    print(f"Done! Saved {frame_id} frames to {args.output_folder}")


if __name__ == "__main__":
    main()
