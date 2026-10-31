"""Shared constants used by train.py, evaluate.py and app.py."""
import os

IMG_SIZE = (224, 224)          # MobileNetV2 input size
BATCH_SIZE = 32
SEED = 42                      # same seed => same train/validation split everywhere
VALIDATION_SPLIT = 0.2         # 80% train, 20% validation

DATA_DIR = "data"
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"
MODEL_PATH = os.path.join(MODEL_DIR, "wastewise_model.keras")
CLASSES_PATH = os.path.join(MODEL_DIR, "class_names.json")
GRAPH_PATH = os.path.join(OUTPUT_DIR, "accuracy_loss_graph.png")
CM_PATH = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
REPORT_PATH = os.path.join(OUTPUT_DIR, "classification_report.txt")

EXPECTED_CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
