from builtins import dict, enumerate
import os
import matplotlib.pyplot as plt
from rich import print
import tensorflow as tf
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# -----------------------------
# Configuration
# -----------------------------

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15

TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"
TEST_DIR = "dataset/test"

MODEL_DIR = "model"

os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# Data Augmentation
# -----------------------------

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)
print("\nClass Mapping")
print(train_generator.class_indices)
labels = train_generator.classes

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels),
    y=labels
)

class_weights = dict(enumerate(class_weights))

print("\nClass Weights")
print(class_weights)

# -----------------------------
# Load EfficientNetB0
# -----------------------------

base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = True

for layer in base_model.layers[:-20]:
    layer.trainable = False

# -----------------------------
# Custom Layers
# -----------------------------

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)

output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

# -----------------------------
# Compile
# -----------------------------

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc")
    ]
)

model.summary()

# -----------------------------
# Callbacks
# -----------------------------

checkpoint = ModelCheckpoint(
    "model/medical_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=2,
    min_lr=1e-6,
    verbose=1
)
# -----------------------------
# Train
# -----------------------------

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[
        checkpoint,
        early_stop,
        reduce_lr
    ]
)

# -----------------------------
# Evaluate
# -----------------------------

print("\nEvaluating on Test Data...\n")

loss, accuracy, auc = model.evaluate(test_generator)

print(f"\nTest Loss      : {loss:.4f}")
print(f"Test Accuracy  : {accuracy:.4f}")
print(f"Test AUC       : {auc:.4f}")

# -----------------------------
# Save Final Model
# -----------------------------

model.save("model/medical_model_final.keras")

print("\nModel Saved Successfully!")

# -----------------------------
# Accuracy Graph
# -----------------------------

plt.figure(figsize=(8,5))

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.savefig("images/accuracy.png")
plt.show()

# -----------------------------
# Loss Graph
# -----------------------------

plt.figure(figsize=(8,5))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.savefig("images/loss.png")
plt.show()

# -----------------------------
# AUC Graph
# -----------------------------

plt.figure(figsize=(8,5))

plt.plot(history.history["auc"], label="Training AUC")
plt.plot(history.history["val_auc"], label="Validation AUC")

plt.title("AUC")
plt.xlabel("Epoch")
plt.ylabel("AUC")
plt.legend()

plt.savefig("images/auc.png")
plt.show()

print("\nTraining Completed Successfully!")