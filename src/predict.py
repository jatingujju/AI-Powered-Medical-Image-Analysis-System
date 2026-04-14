import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model

# ✅ Load trained model
MODEL_PATH = "models/model.h5"

if not os.path.exists(MODEL_PATH):
    print("❌ Model not found! Train model first.")
    exit()

model = load_model(MODEL_PATH)

# ✅ Prediction function
def predict_image(img_path):
    # Check if file exists
    if not os.path.exists(img_path):
        print(f"❌ Image path does not exist: {img_path}")
        return None, None

    # Read image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Check if image loaded properly
    if img is None:
        print("❌ Failed to load image!")
        return None, None

    # Preprocess
    img = cv2.resize(img, (256, 256))
    img = img / 255.0
    img = img.reshape(1, 256, 256, 1)

    # Predict
    pred = model.predict(img)[0][0]

    # Interpret result
    if pred > 0.5:
        label = "PNEUMONIA"
        confidence = float(pred)
    else:
        label = "NORMAL"
        confidence = float(1 - pred)

    return label, confidence


# ✅ TEST THE MODEL
if __name__ == "__main__":

    # 🔥 Step 1: Show available images
    test_folder = "data/test/NORMAL"

    if not os.path.exists(test_folder):
        print("❌ Test folder not found!")
        exit()

    files = os.listdir(test_folder)

    if len(files) == 0:
        print("❌ No images found in folder!")
        exit()

    print("📂 Available images:")
    for i, file in enumerate(files[:5]):
        print(f"{i+1}. {file}")

    # 🔥 Step 2: Pick first image automatically
    sample_image = os.path.join(test_folder, files[0])

    print("\n🔍 Testing image:", sample_image)

    result, confidence = predict_image(sample_image)

    if result is not None:
        print(f"\n✅ Prediction: {result}")
        print(f"📊 Confidence: {confidence:.2f}")