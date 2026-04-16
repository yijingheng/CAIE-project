# AI-Assisted Interpretation of LED Colour and Intensity Data

A CAIE final project that combines **computer vision**, **lux prediction**, and a **large language model (LLM)** to interpret LED indicator states from a live camera feed.

The system detects LED colour with YOLOv8, estimates relative light intensity with a physics-informed PyTorch model, and generates a human-readable interpretation using an LLM through Ollama.

## Why this project

In manual test stations, operators often need to watch indicator lights and decide whether a process has passed, failed, or is still running. This project reduces reliance on subjective visual inspection by automating three steps:

1. Detect the LED region and classify its colour.
2. Estimate brightness / lux from the detected crop.
3. Convert the structured result into a readable explanation.

This directly matches the CAIE brief requirement to combine an **LLM** with at least one additional AI component in a practical workflow.

## Core capabilities

- **Computer vision:** YOLOv8-based LED colour detection (`blue`, `green`, `red`)
- **Intensity estimation:** physics-informed PyTorch model for lux prediction from image crop brightness
- **LLM interpretation:** Ollama-hosted local model generates a readable test-state explanation
- **Interface:** Streamlit app for live inspection from an IP camera / phone camera stream
- **Training utilities:** scripts for dataset splitting, YOLO training, timestamp extraction, frame extraction, and lux-model testing

## Current project structure

```text
CAIE-project/
├── app/
│   ├── app.py
│   ├── detect.py
│   ├── intensity.py
│   └── llm_module.py
├── data/
│   └── dataset_structure_example/
│       └── data.yaml
├── dataset_split/
│   ├── train/
│   └── val/
├── models/
│   ├── lux_model_physics.pth
│   ├── yolov8n.pt
│   └── README.md
├── preprocessing/
│   ├── extract_timestamp.py
│   ├── split_dataset.py
│   └── videoextract.py
├── runs/
│   └── detect/
│       └── color_detection_model/
├── training/
│   ├── test_lux_mode.py
│   ├── test_pipeline.py
│   ├── train_lux_model_with_physics.py
│   └── train_model.py
├── README.md
├── requirements.txt
└── SETUP.md
```

## Pipeline overview

### 1) Detection
The Streamlit app loads YOLO weights and detects the LED region in each processed frame.

### 2) Intensity prediction
The detected LED crop is resized and normalized. A ResNet18-based regression model combines image features with normalized brightness computed from camera exposure settings.

### 3) Interpretation
The detected colour and predicted lux value are passed to the LLM module, which returns a readable explanation such as **PASS**, **FAIL**, or **PROCESSING**.

## Dataset notes

- LED classes are:
  - `0: blue`
  - `1: green`
  - `2: red`
- Dataset format follows YOLO polygon labels.
- `data/dataset_structure_example/data.yaml` points to:
  - `train: train/images`
  - `val: val/images`
- Dataset splitting is handled by `preprocessing/split_dataset.py`.

## Requirements

- Python **3.10 or 3.11** recommended
- Windows, macOS, or Linux
- Ollama installed locally for LLM-based interpretation
- A camera source for live demo, such as:
  - webcam adapted in code, or
  - phone IP camera stream URL

## Quick start

### 1. Clone the repository
```bash
git clone https://github.com/yijingheng/CAIE-project.git
cd CAIE-project
```

### 2. Create and activate a virtual environment
**Windows (PowerShell)**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install and run Ollama
Install Ollama, then pull the model used by the app:

```bash
ollama pull llama2
ollama serve
```

The app currently expects:
- Ollama API at `http://localhost:11434`
- model name `llama2:latest`

### 5. Check model files
Make sure these files are present:

- `models/lux_model_physics.pth`
- YOLO weights at either:
  - `runs/detect/color_detection_model/weights/best.pt`, or
  - `models/best.pt`

### 6. Run the Streamlit app
From the project root:

```bash
streamlit run app/app.py
```

## Using the app

1. Start Ollama.
2. Run the Streamlit interface.
3. Enter your phone/IP camera stream URL in the sidebar.
4. Adjust F-number, shutter speed, and YOLO confidence if needed.
5. Start live detection.
6. The app displays:
   - the live frame,
   - the latest processed frame,
   - detected colour,
   - predicted lux,
   - LLM-generated assessment.

## Training and preprocessing scripts

### Split a YOLO dataset
```bash
python preprocessing/split_dataset.py --dataset-dir data/sample --output-dir dataset_split
```

### Train the YOLO model
```bash
python training/train_model.py
```

### Test the lux model on a sample image
```bash
python training/test_lux_mode.py
```

### Run a simple lux-model prediction
```bash
python training/train_lux_model_with_physics.py
```

### Extract frames from video
```bash
python preprocessing/videoextract.py
```

### Extract timestamps / align frame timing
```bash
python preprocessing/extract_timestamp.py
```

## Known limitations

- The current app is designed around a **single detected LED result per processed frame**.
- The LLM logic is tailored to a fixed interpretation mapping and local Ollama setup.
- Camera settings are user-controlled and affect lux estimation.
- Some sample-data and training workflows assume files already exist locally.

