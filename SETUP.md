# Setup Guide

This guide explains how to run the CAIE LED interpretation project locally.

## 1. Prerequisites

Install the following first:

- Python **3.10 or 3.11**
- Git
- Ollama
- A camera source for the live demo:
  - phone IP camera stream, or
  - your own webcam adaptation if you modify the app

## 2. Clone the repository

```bash
git clone https://github.com/yijingheng/CAIE-project.git
cd CAIE-project
```

## 3. Create a virtual environment

### Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 4. Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 5. Prepare Ollama

Install Ollama, then pull the model expected by the app:

```bash
ollama pull llama2
```

Start the local Ollama server:

```bash
ollama serve
```

The application checks for:
- endpoint: `http://localhost:11434/api/tags`
- generation API: `http://localhost:11434/api/generate`
- model: `llama2:latest`

## 6. Confirm model files

Before launching the app, make sure these model files exist.

### Lux model
Required at one of the following locations:
- `models/lux_model_physics.pth`
- `lux_model_physics.pth` at the project root

### YOLO weights
Required at one of the following locations:
- `runs/detect/color_detection_model/weights/best.pt`
- `models/best.pt`

The base `models/yolov8n.pt` file is used for training initialization, not as the fine-tuned inference weight.

## 7. Run the app

From the project root:

```bash
streamlit run app/app.py
```

## 8. App controls

The Streamlit sidebar includes:

- **F-Number**
- **Shutter Speed**
- **YOLO Confidence**
- **Phone Camera URL**
- **Start Live Detection**

Use a valid IP camera stream URL such as:

```text
http://<your-phone-ip>:8080/video
```

## 9. Optional: preprocessing and training

### Split the dataset
```bash
python preprocessing/split_dataset.py --dataset-dir data/sample --output-dir dataset_split
```

### Train YOLOv8
```bash
python training/train_model.py
```

### Test the lux model
```bash
python training/test_lux_mode.py
```

### Run simple lux prediction
```bash
python training/train_lux_model_with_physics.py
```

### Extract video frames
```bash
python preprocessing/videoextract.py
```

### Extract timestamps
```bash
python preprocessing/extract_timestamp.py
```

## 10. Troubleshooting

### `Cannot connect to phone camera`
- Confirm the IP camera app is running.
- Confirm the phone and laptop are on the same network.
- Check that the URL ends with `/video` if your IP camera app requires it.

### `Ollama is not running`
Run:
```bash
ollama serve
```

### `model not found`
Check that the `.pth` and `.pt` files are in one of the supported paths listed above.

### `No module named ...`
Activate the virtual environment and reinstall dependencies:
```bash
pip install -r requirements.txt
```
