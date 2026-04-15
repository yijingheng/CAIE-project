import csv
from pathlib import Path

INPUT_FILE = Path("serial_monitor_copy.txt")
OUTPUT_FILE = Path("led_data.csv")


def parse_line(line: str):
    """
    Expected format:
    00:38:59.730 -> 107.81,84.86,106.71,91.31
    """

    line = line.strip()

    if not line or "->" not in line:
        return None

    try:
        timestamp_part, data_part = line.split("->")
        timestamp = timestamp_part.strip()

        values = [float(x.strip()) for x in data_part.strip().split(",")]

        if len(values) != 4:
            return None

        return [timestamp] + values

    except ValueError:
        return None


def main():
    if not INPUT_FILE.exists():
        print(f"ERROR: Input file not found -> {INPUT_FILE.resolve()}")
        return

    rows = []

    with INPUT_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            parsed = parse_line(line)
            if parsed:
                rows.append(parsed)

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Header
        writer.writerow(["Timestamp", "R", "G", "B", "Brightness"])

        # Data
        writer.writerows(rows)

    print(f"✅ Saved {len(rows)} rows to {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()