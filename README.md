# 🩺 AI Medical Report Assistant

An end-to-end AI-powered medical image analysis system that classifies Chest X-ray images as **NORMAL** or **PNEUMONIA** using **TensorFlow Transfer Learning (EfficientNetB0)** and generates an **AI-assisted medical report** using **Google Gemini**.

The application is deployed using **Streamlit**, stores prediction history in **SQLite**, and automatically generates a **professional PDF medical report** for every prediction.

---

# 🚀 Features

- ✅ Chest X-ray image classification
- ✅ Transfer Learning using EfficientNetB0
- ✅ AI-generated medical report using Google Gemini
- ✅ Automatic PDF medical report generation
- ✅ Download AI-generated PDF reports
- ✅ Interactive Streamlit web application
- ✅ Upload Chest X-ray images for prediction
- ✅ Display prediction confidence score
- ✅ Store prediction history in SQLite
- ✅ Download prediction history as CSV
- ✅ Accuracy, Loss, AUC, ROC Curve, and Confusion Matrix visualization

---

# 🛠️ Technologies Used

- Python 3.12+
- TensorFlow / Keras
- EfficientNetB0
- Streamlit
- Google Gemini API
- SQLite
- ReportLab
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

---

# 📁 Project Structure

```text
AI-Medical-Report-Assistant/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── predict.py
│   ├── database.py
│   ├── llm.py
│   ├── report_generator.py
│   └── utils.py
│
├── database/
│
├── dataset/
│   ├── train/
│   ├── val/
│   └── test/
│
├── images/
│   ├── accuracy.png
│   ├── loss.png
│   ├── auc.png
│   ├── confusion_matrix.png
│   └── roc_curve.png
│
├── reports/
│   ├── report_YYYYMMDD_HHMMSS.pdf
│
├── model/
│
├── uploads/
│
├── train.py
├── evaluate.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 📊 Dataset

### Dataset Name

**Chest X-Ray Images (Pneumonia)**

### Source

https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

---

### Dataset Statistics

| Split | NORMAL | PNEUMONIA | Total |
|-------|-------:|----------:|------:|
| Train | 1341 | 3875 | 5216 |
| Validation | 8 | 8 | 16 |
| Test | 234 | 390 | 624 |

---

### Classes

- NORMAL
- PNEUMONIA

---

# 📈 Label Distribution

| Class | Images |
|-------|-------:|
| NORMAL | 1341 |
| PNEUMONIA | 3875 |

The dataset is imbalanced; therefore, **class weights** were used during training to improve model performance.

---

# 🧠 Deep Learning Model

Transfer Learning Model:

## EfficientNetB0

### Image Size

224 × 224

### Optimizer

Adam

### Loss Function

Binary Crossentropy

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC Curve
- AUC
- Validation AUC
- Confusion Matrix
- Classification Report

---

# 📈 Model Performance

| Metric | Value |
|--------|-------:|
| Accuracy | **89.10%** |
| Precision | **86.76%** |
| Recall | **97.44%** |
| F1 Score | **91.79%** |
| Test AUC | **97.28%** |

---

# 🤖 AI Medical Report

After prediction, **Google Gemini** generates an AI-assisted medical report containing:

- Disease Prediction
- Confidence Score
- Possible Findings
- Possible Symptoms
- General Precautions
- Recommendation to consult a healthcare professional
- Medical Disclaimer

The generated report is automatically converted into a **professional PDF report**, saved inside the **reports/** folder, and can be downloaded directly from the Streamlit application.

---

# 📄 PDF Report Generation

For every prediction, the application automatically creates a PDF medical report.

Each report includes:

- Image Name
- Prediction
- Confidence Score
- AI-generated Medical Report
- Medical Disclaimer
- Timestamp

Generated reports are stored in:

```text
reports/
```

Users can also download the report directly from the application.

---

# 💾 SQLite Database

The application stores prediction history including:

- Image Name
- Prediction
- Confidence Score
- AI-generated Medical Report
- Timestamp

Additional Features:

- View prediction history
- Download prediction history as CSV
- Download AI-generated PDF reports

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/Govardhan-17/AI-Medical-Report-Assistant.git

cd AI-Medical-Report-Assistant
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it.

### Windows

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Create a `.env` file in the project root.

```env
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
```

Generate your Gemini API Key from:

https://aistudio.google.com/app/apikey

---

# 🏋️ Train the Model

```bash
python train.py
```

---

# 📊 Evaluate the Model

```bash
python evaluate.py
```

This generates:

- Accuracy Curve
- Loss Curve
- AUC Curve
- Confusion Matrix
- ROC Curve

---

# 🌐 Run the Application

```bash
streamlit run app/main.py
```

---

# 🖥️ Application Workflow

1. Upload a Chest X-ray image.
2. Preprocess the image.
3. EfficientNetB0 predicts the disease.
4. Display prediction confidence.
5. Google Gemini generates an AI-assisted medical report.
6. Automatically generate a PDF report.
7. Save the PDF inside the **reports/** folder.
8. Store prediction details in SQLite.
9. Display the report.
10. Allow users to download the PDF report.
11. Display prediction history.
12. Download prediction history as CSV.

---

# 📷 Generated Outputs

The project generates:

- Accuracy Graph
- Loss Graph
- AUC Graph
- Confusion Matrix
- ROC Curve
- AI-generated PDF Medical Reports

Graphs are stored in:

```text
images/
```

Medical reports are stored in:

```text
reports/
```

---

# 🚀 Future Improvements

- Multi-class disease classification
- Grad-CAM visualization
- Docker support
- Cloud deployment (AWS / Azure / GCP)
- User authentication
- REST API
- Patient dashboard
- Multi-language AI medical reports
- Email PDF reports directly to patients

---

# ⚠️ Disclaimer

This project is developed **only for educational and research purposes**.

The prediction generated by the deep learning model and the AI-generated medical report **must not be considered a medical diagnosis**.

Always consult a qualified healthcare professional for diagnosis, treatment, and medical advice.

---

# 👨‍💻 Author

**Govardhan Sunkari**

B.Tech – Artificial Intelligence & Machine Learning

GitHub:
https://github.com/Govardhan-17
