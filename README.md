# CAIE LED Interpretation System

## Overview
This project implements an AI-powered system for interpreting LED signals in educational contexts. It combines computer vision (YOLOv8), lux sensor data modeling, and large language model (LLM) interpretation to analyze LED patterns and provide meaningful insights.

## Features
- Real-time LED detection using YOLOv8
- Lux sensor data integration for brightness analysis
- Physics-informed machine learning models
- Streamlit web interface for interactive analysis
- Ollama integration for LLM-based interpretation

## Installation
See [SETUP.md](SETUP.md) for detailed installation instructions.

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Download models: `python models/download_models.py`
3. Run the app: `streamlit run app/app.py`

## Project Structure
- `app/` - Main application code
- `preprocessing/` - Data preprocessing scripts
- `training/` - Model training scripts
- `models/` - Model download and management
- `data/` - Sample data and configurations
- `docs/` - Documentation

## License
MIT