"""
evaluate.py - Evaluate the trained model on the validation split.
Creates outputs/confusion_matrix.png and outputs/classification_report.txt

Run:  python evaluate.py   (after train.py)
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras

from common import *


def main():
    if not os.path.isfile(MODEL_PATH):
        sys.exit(f"ERROR: model '{MODEL_PATH}' not found. Run 'python train.py' first.")
    if not os.path.isdir(DATA_DIR):
        sys.exit(f"ERROR: '{DATA_DIR}/' folder not found.")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        model = keras.models.load_model(MODEL_PATH)
    except Exception as e:
        sys.exit(f"ERROR: could not load model: {e}")

    # Same seed + split as train.py => same validation images
    val_ds = keras.utils.image_dataset_from_directory(
        DATA_DIR, labels="inferred", label_mode="int", class_names=EXPECTED_CLASSES,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT, subset="validation", seed=SEED)

    # Predict and collect true labels in ONE pass (keeps them aligned)
    y_true, y_pred = [], []
    for images, labels in val_ds:
        probs = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(probs, axis=1))
        y_true.extend(labels.numpy())

    report = classification_report(y_true, y_pred, target_names=EXPECTED_CLASSES, digits=3)
    print(report)
    with open(REPORT_PATH, "w") as f:
        f.write(report)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Greens")
    ax.set_xticks(range(6)); ax.set_xticklabels(EXPECTED_CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(6)); ax.set_yticklabels(EXPECTED_CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion Matrix")
    for i in range(6):
        for j in range(6):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(CM_PATH, dpi=150)
    plt.close()
    print(f"Saved -> {CM_PATH}\nSaved -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
