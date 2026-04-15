# Setup Guide

## Prerequisites
- Python 3.8+
- Git
- Ollama (for LLM features)

## Installation
1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd caie-led-interpretation
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install Ollama:
   - Download from https://ollama.ai/
   - Pull required models: `ollama pull llama2`

4. Download pre-trained models:
   ```bash
   python models/download_models.py
   ```

## Running the Application
```bash
streamlit run app/app.py
```

## Training Models (Optional)
See `training/` directory for training scripts.

## Data Preparation
Sample data is provided in `data/sample/`. For full dataset, follow preprocessing scripts.