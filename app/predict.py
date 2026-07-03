import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

# ----------------------------------------
# Configuration
# ----------------------------------------

MODEL_PATH = "model/medical_model.keras"

CLASS_NAMES = [
    "NORMAL",
    "PNEUMONIA"
]

# ----------------------------------------
# Load Model
# ----------------------------------------

print("Loading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")

# ----------------------------------------
# Prediction Function
# ----------------------------------------

def predict_image(img_path):
    """
    Predict disease from an image.

    Parameters:
        img_path (str): Path to image

    Returns:
        prediction (str)
        confidence (float)
    """

    # Load image
    img = image.load_img(
        img_path,
        target_size=(224, 224)
    )

    # Convert to array
    img_array = image.img_to_array(img)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # EfficientNet preprocessing
    img_array = preprocess_input(img_array)

    # Predict
    prediction = model.predict(
        img_array,
        verbose=0
    )

    probability = float(prediction[0][0])

    if probability >= 0.5:
        label = CLASS_NAMES[1]
        confidence = probability
    else:
        label = CLASS_NAMES[0]
        confidence = 1 - probability

    confidence = round(confidence * 100, 2)

    return label, confidence


# ----------------------------------------
# Test
# ----------------------------------------

if __name__ == "__main__":

    img_path = input("Enter image path: ")

    label, confidence = predict_image(img_path)

    print("\nPrediction")
    print("-------------------------")
    print(f"Disease   : {label}")
    print(f"Confidence: {confidence}%")