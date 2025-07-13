import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import urllib.request
import os

# --- Settings ---
MODEL_PATH = "cnn_model.keras"
REMOTE_MODEL_URL = (
    "https://huggingface.co/VictoryUdofia/deepfake-model-h5/resolve/main/cnn_model.h5"
)
TARGET_SIZE = (224, 224)

# --- Download model if not available ---
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        urllib.request.urlretrieve(REMOTE_MODEL_URL, MODEL_PATH)
        st.success("Model downloaded!")

# Load the model
model = load_model(MODEL_PATH)

# --- Image preprocessing ---
def preprocess_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image.")
    img = cv2.resize(img, TARGET_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# --- Streamlit UI ---
st.title("🕵️‍♂️ Deepfake Image Detector")
st.markdown("Upload an image to detect if it's Real or Fake.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    try:
        img = preprocess_image(uploaded_file)
        prediction = model.predict(img)[0][0]
        label = "🟡 Fake" if prediction >= 0.5 else "🟢 Real"
        confidence = round(float(prediction), 2)
        st.markdown(f"**Prediction:** {label}")
        st.markdown(f"**Confidence:** {confidence}")
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
