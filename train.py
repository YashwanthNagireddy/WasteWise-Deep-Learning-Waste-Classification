"""
train.py - Train the WasteWise classifier (MobileNetV2 transfer learning).

Run:  python train.py
Needs images in data/<class_name>/ (see README.md).
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")  # save plots to file without opening a window
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from common import *


def check_dataset():
    """Make sure the dataset folder exists and has images in each class folder."""
    if not os.path.isdir(DATA_DIR):
        sys.exit(f"ERROR: '{DATA_DIR}/' folder not found. See README.md (Dataset Setup).")
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    for cls in EXPECTED_CLASSES:
        folder = os.path.join(DATA_DIR, cls)
        if not os.path.isdir(folder):
            sys.exit(f"ERROR: missing folder '{folder}'.")
        n = len([f for f in os.listdir(folder) if f.lower().endswith(exts)])
        print(f"  {cls:<10}: {n} images")
        if n == 0:
            sys.exit(f"ERROR: '{folder}' has no images. Copy the dataset images in first.")


def load_data():
    """Load train/validation datasets with an 80/20 split."""
    common = dict(
        directory=DATA_DIR, labels="inferred", label_mode="int",
        class_names=EXPECTED_CLASSES, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT, seed=SEED,
    )
    train_ds = keras.utils.image_dataset_from_directory(subset="training", **common)
    val_ds = keras.utils.image_dataset_from_directory(subset="validation", **common)
    # Prefetch so data loading overlaps with training
    return (train_ds.prefetch(tf.data.AUTOTUNE), val_ds.prefetch(tf.data.AUTOTUNE))


def build_model(num_classes):
    """Augmentation -> MobileNetV2 preprocessing -> frozen MobileNetV2 -> classifier head."""
    base = keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
    base.trainable = False  # Phase 1: keep pretrained weights frozen

    inputs = keras.Input(shape=IMG_SIZE + (3,))
    # Data augmentation (only active during training)
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomZoom(0.15)(x)
    x = layers.RandomContrast(0.15)(x)
    # MobileNetV2 expects pixels scaled to [-1, 1]
    x = layers.Rescaling(1.0 / 127.5, offset=-1)(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs), base


def compile_model(model, lr):
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])


def plot_history(history):
    """Save accuracy and loss curves."""
    acc, val_acc = history["accuracy"], history["val_accuracy"]
    loss, val_loss = history["loss"], history["val_loss"]
    epochs = range(1, len(acc) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(epochs, acc, label="Training"); ax[0].plot(epochs, val_acc, label="Validation")
    ax[0].set_title("Accuracy"); ax[0].set_xlabel("Epoch"); ax[0].legend()
    ax[1].plot(epochs, loss, label="Training"); ax[1].plot(epochs, val_loss, label="Validation")
    ax[1].set_title("Loss"); ax[1].set_xlabel("Epoch"); ax[1].legend()
    plt.tight_layout()
    plt.savefig(GRAPH_PATH, dpi=150)
    plt.close()
    print(f"Saved graph -> {GRAPH_PATH}")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Checking dataset...")
    check_dataset()
    train_ds, val_ds = load_data()

    model, base = build_model(len(EXPECTED_CLASSES))
    callbacks = [
        keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_accuracy",
                                        save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=4,
                                      restore_best_weights=True),
    ]

    # Phase 1: train only the new classifier head
    print("\n=== Phase 1: training classifier head ===")
    compile_model(model, 1e-3)
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=callbacks).history

    # Phase 2: fine-tune the top layers of MobileNetV2 with a small learning rate
    print("\n=== Phase 2: fine-tuning top MobileNetV2 layers ===")
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    compile_model(model, 1e-5)
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=callbacks).history

    history = {k: h1[k] + h2[k] for k in h1}
    plot_history(history)

    # Reload best saved model and save class order for the app
    best = keras.models.load_model(MODEL_PATH)
    loss, acc = best.evaluate(val_ds, verbose=0)
    with open(CLASSES_PATH, "w") as f:
        json.dump(EXPECTED_CLASSES, f)
    print(f"\nBest model validation accuracy: {acc * 100:.2f}% (loss {loss:.4f})")
    print(f"Model saved -> {MODEL_PATH}")
    print("Next: python evaluate.py")


if __name__ == "__main__":
    main()
