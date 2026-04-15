# Sample Data

This directory contains sample data for testing and demonstration purposes.

## Images
- `images/`: Sample frame images from video processing
- Format: JPEG images extracted from video frames

## Labels
- `labels/`: YOLO format annotation files (.txt)
- Format: `<class> <x_center> <y_center> <width> <height>`
- Class 0: LED

## Lux Labels
- `lux_labels_sample.csv`: Sample lux sensor readings matched to frames
- Columns: timestamp, lux_value, frame_id, etc.

## Dataset Structure
See `../dataset_structure_example/data.yaml` for full dataset configuration.