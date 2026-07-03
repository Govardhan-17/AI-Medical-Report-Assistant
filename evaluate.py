import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

# ----------------------------------------
# Configuration
# ----------------------------------------

MODEL_PATH = "model/medical_model.keras"
TEST_DIR = "dataset/test"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# ----------------------------------------
# Check Model
# ----------------------------------------

print("Model exists:", os.path.exists(MODEL_PATH))
print("Loading:", MODEL_PATH)

print("\nLoading trained model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")

# ----------------------------------------
# Load Test Dataset
# ----------------------------------------

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

# ----------------------------------------
# Predict
# ----------------------------------------

print("\nPredicting...\n")

predictions = model.predict(test_generator)

y_prob = predictions.flatten()

print("\nPrediction Range")
print("Min :", np.min(y_prob))
print("Max :", np.max(y_prob))
print("Mean:", np.mean(y_prob))

y_pred = (y_prob >= 0.5).astype(int)

y_true = test_generator.classes

# ----------------------------------------
# Metrics
# ----------------------------------------

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("\n" + "=" * 50)
print("Evaluation Results")
print("=" * 50)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# ----------------------------------------
# Classification Report
# ----------------------------------------

print("\nClassification Report\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=list(test_generator.class_indices.keys()),
        zero_division=0
    )
)

# ----------------------------------------
# Confusion Matrix
# ----------------------------------------

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=test_generator.class_indices.keys(),
    yticklabels=test_generator.class_indices.keys()
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

os.makedirs("images", exist_ok=True)

plt.savefig("images/confusion_matrix.png")

plt.show()

# ----------------------------------------
# ROC Curve
# ----------------------------------------

fpr, tpr, _ = roc_curve(y_true, y_prob)

roc_auc = auc(fpr, tpr)

print(f"\nAUC Score : {roc_auc:.4f}")

plt.figure(figsize=(6,5))

plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1],[0,1],'k--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.tight_layout()

plt.savefig("images/roc_curve.png")

plt.show()

print("\nEvaluation Completed Successfully.")