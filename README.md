# AI Medical Report Assistant

## Overview

AI Medical Report Assistant is an end-to-end deep learning application that analyzes medical images, predicts disease, and generates an AI-assisted medical report using a Large Language Model (LLM).

This project combines TensorFlow, Transfer Learning, Google Gemini, Streamlit, and SQLite into a single medical image analysis system.

---

## Features

* Medical image classification using TensorFlow
* Transfer Learning with EfficientNetB0
* AI-generated medical report using Google Gemini
* Interactive Streamlit web application
* Upload medical images for prediction
* Display prediction confidence
* Store prediction history in SQLite
* Download prediction history as CSV
* Training and evaluation visualization

---

## Technologies Used

* Python
* TensorFlow
* EfficientNetB0
* Streamlit
* SQLite
* Google Gemini API
* Scikit-learn
* OpenCV
* Matplotlib
* Pandas

---

## Project Structure

```text
AI-Medical-Report-Assistant/
│
├── app/
│   ├── app.py
│   ├── predict.py
│   ├── database.py
│   ├── llm.py
│   └── utils.py
│
├── dataset/
│   ├── train/
│   ├── val/
│   └── test/
│
├── database/
├── uploads/
├── images/
├── model/
├── notebooks/
├── reports/
│
├── train.py
├── evaluate.py
├── requirements.txt
└── README.md
```

---

## Dataset

Dataset Used:

Chest X-Ray Images (Pneumonia)

Classes:

* NORMAL
* PNEUMONIA

Dataset contains separate Training, Validation, and Test folders.

---

## Model

Transfer Learning Model:

EfficientNetB0

Image Size:

224 × 224

Optimizer:

Adam

Loss Function:

Binary Crossentropy

Metrics:

* Accuracy
* AUC

---

## Evaluation Metrics

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC Curve
* AUC Score
* Validation AUC
* Confusion Matrix
* Classification Report

---

## AI Medical Report

After prediction, Google Gemini generates a concise AI-assisted medical report that includes:

* Prediction
* Possible findings
* Possible symptoms
* General precautions
* Recommendation to consult a healthcare professional
* Disclaimer that the output is not a medical diagnosis

---

## SQLite Database

Prediction history is stored with:

* Image Name
* Prediction
* Confidence
* AI Report
* Timestamp

---

## Installation

Clone the repository.

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## Train the Model

```bash
python train.py
```

---

## Evaluate the Model

```bash
python evaluate.py
```

---

## Run the Streamlit Application

```bash
streamlit run app/app.py
```

---

## Expected Output

The application allows users to:

* Upload a chest X-ray image
* Predict whether the image is NORMAL or PNEUMONIA
* Display confidence score
* Generate an AI-assisted medical report
* Save prediction history
* View previous predictions
* Download prediction history as CSV

---

## Future Improvements

* Multi-class disease classification
* Grad-CAM visualization
* PDF medical report generation
* User authentication
* Cloud deployment
* Docker support

---

## Disclaimer

This project is intended solely for educational and research purposes.

The AI prediction and generated report should not be considered a medical diagnosis. Always consult a qualified healthcare professional for diagnosis and treatment.
