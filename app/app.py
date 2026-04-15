import cv2
import time
import streamlit as st

from detect import load_lux, load_yolo, predict
from llm_module import check_ollama_connection

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(layout="wide")
st.title("💡 LED Inspection System (Live)")

# =========================
# CHECK OLLAMA CONNECTION
# =========================
ollama_status = check_ollama_connection()
if ollama_status:
    st.success("✅ Ollama (llama2) connected and ready")
else:
    st.warning(
        "⚠️ **Ollama is not running!**\n\n"
        "Please start Ollama in a terminal before using this app:\n\n"
        "`ollama serve`"
    )

# =========================
# SIDEBAR CONTROLS
# =========================
st.sidebar.header("⚙️ Controls")

# Camera parameters
f_number = st.sidebar.slider("F-Number", 1.0, 5.0, 1.9)
shutter = st.sidebar.slider("Shutter Speed (1/x)", 30, 240, 60)
EXPOSURE_FACTOR = (1 / shutter) / (f_number ** 2)

# Detection settings
conf_threshold = st.sidebar.slider("YOLO Confidence", 0.1, 1.0, 0.5)

# Stream settings
ip_url = st.sidebar.text_input(
    "📡 Phone Camera URL",
    "http://192.168.100.34:8080/video"
)

run = st.sidebar.checkbox("▶ Start Live Detection")

# =========================
# LOAD MODELS
# =========================
yolo_model = load_yolo()
lux_model, device = load_lux()

# =========================
# PIPELINE
# =========================

# =========================
# LIVE STREAM DISPLAY
# =========================
st.subheader("📹 Live Stream")
frame_window = st.image([])
st.subheader("🖼️ Captured Processed Image")
processed_window = st.image([])
text_window = st.empty()

if run:
    # Check if Ollama is available before starting
    if not check_ollama_connection():
        st.error(
            "❌ **Cannot start - Ollama not available**\n\n"
            "Please start Ollama in a terminal:\n\n"
            "`ollama serve`"
        )
    else:
        cap = cv2.VideoCapture(ip_url)

        if not cap.isOpened():
            st.error("❌ Cannot connect to phone camera")
        else:
            last_processed_text = "Initializing..."
            frame_count = 0
            PROCESS_INTERVAL = 30  # Process every 30 frames (adjust as needed for smoothness)

            while run:
                ret, frame = cap.read()
                if not ret:
                    st.error("❌ Failed to read frame")
                    break

                # Display raw frame for smooth streaming
                frame_window.image(frame, channels="BGR")

                frame_count += 1

                # Process only at intervals
                if frame_count % PROCESS_INTERVAL == 0:
                    processed_frame, result_text = predict(
                        frame,
                        yolo_model,
                        lux_model,
                        device,
                        conf_threshold,
                        EXPOSURE_FACTOR,
                    )
                    # Update captured processed image
                    processed_window.image(processed_frame, channels="BGR")
                    text_window.text(result_text)
                    last_processed_text = result_text
                else:
                    text_window.text(last_processed_text)

                time.sleep(0.05)

            cap.release()