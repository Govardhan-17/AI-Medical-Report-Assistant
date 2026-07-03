import os
import uuid
from PIL import Image
import numpy as np

# ----------------------------------------
# Configuration
# ----------------------------------------

UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

IMAGE_SIZE = (224, 224)

# ----------------------------------------
# Save Uploaded Image
# ----------------------------------------

def save_uploaded_file(uploaded_file):
    """
    Save uploaded Streamlit file to uploads folder.

    Returns:
        file_path
    """

    extension = uploaded_file.name.split(".")[-1]

    filename = f"{uuid.uuid4()}.{extension}"

    file_path = os.path.join(UPLOAD_FOLDER, filename)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path


# ----------------------------------------
# Load Image
# ----------------------------------------

def load_image(file_path):
    """
    Load image using PIL.
    """

    image = Image.open(file_path)

    return image


# ----------------------------------------
# Preprocess Image
# ----------------------------------------

def preprocess_image(file_path):
    """
    Convert image into model input.
    """

    img = Image.open(file_path)

    img = img.convert("RGB")

    img = img.resize(IMAGE_SIZE)

    img = np.array(img)

    img = img.astype("float32") / 255.0

    img = np.expand_dims(img, axis=0)

    return img


# ----------------------------------------
# Image Information
# ----------------------------------------

def get_image_info(file_path):

    img = Image.open(file_path)

    width, height = img.size

    return {
        "Width": width,
        "Height": height,
        "Mode": img.mode,
        "Format": img.format
    }


# ----------------------------------------
# Delete Uploaded Image
# ----------------------------------------

def delete_image(file_path):

    if os.path.exists(file_path):
        os.remove(file_path)


# ----------------------------------------
# Test
# ----------------------------------------

if __name__ == "__main__":

    print("Utility functions loaded successfully.")