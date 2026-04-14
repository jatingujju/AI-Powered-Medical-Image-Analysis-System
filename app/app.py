import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# Load model
model = load_model("models/model.h5")

# Title
st.title("🧠 AI Medical Image Analyzer")
st.write("Upload a chest X-ray image to detect Pneumonia")

# File uploader
uploaded_file = st.file_uploader("Choose an X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert to grayscale
    img = np.array(image.convert("L"))

    # Resize
    img = cv2.resize(img, (256,256))

    # Normalize
    img = img / 255.0

    # Reshape
    img = img.reshape(1,256,256,1)

    # Predict
    pred = model.predict(img)[0][0]

    # Result
    if pred > 0.5:
        result = "🛑 PNEUMONIA DETECTED"
        confidence = pred
    else:
        result = "✅ NORMAL"
        confidence = 1 - pred

    # Display result
    st.subheader("Prediction Result")
    st.write(result)
    st.write(f"Confidence: {confidence:.2f}")