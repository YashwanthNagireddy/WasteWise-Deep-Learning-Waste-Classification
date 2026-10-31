# WasteWise: Assignment Documentation

## 1. Abstract
WasteWise is a deep learning web application that classifies waste images into six categories (cardboard, glass, metal, paper, plastic, trash). It uses transfer learning with MobileNetV2 trained on the TrashNet dataset and is deployed through a Streamlit interface that shows the predicted class and confidence.

## 2. Problem Statement
Manual waste segregation is slow, error-prone and often skipped, which lowers recycling quality and harms the environment. An automatic, image-based classifier can help people and systems sort waste correctly.

## 3. Objectives
- Build a CNN classifier for six waste classes using transfer learning.
- Preprocess and augment images to improve generalization.
- Evaluate with accuracy/loss curves, confusion matrix and classification report.
- Provide a simple web app for real-time prediction.

## 4. Existing vs Proposed System
**Existing:** manual sorting by people or workers; sensor-based sorting that is costly; classic ML needing hand-made features.
**Proposed:** automatic feature learning by a pretrained CNN, low cost (any camera/phone image), runs on ordinary computers, easy web interface.

## 5. Methodology
1. Collect TrashNet images and organize them into 6 folders.
2. Split 80% training / 20% validation (seed 42).
3. Resize to 224x224, scale to [-1,1], augment (flip, rotation, zoom, contrast).
4. Phase 1: freeze MobileNetV2, train the new classifier head (lr 1e-3).
5. Phase 2: unfreeze the last 30 layers and fine-tune (lr 1e-5).
6. Save the best model by validation accuracy.
7. Evaluate: graphs, confusion matrix, classification report.
8. Deploy via Streamlit.

## 6. System Architecture (simple explanation)
User uploads image -> app validates and resizes it -> model (MobileNetV2 features + Dense softmax layer) outputs 6 probabilities -> the highest is shown as the prediction with its confidence, plus all class probabilities.

## 7. Requirements
**Hardware:** 8 GB RAM recommended (4 GB minimum), any modern CPU (GPU optional), about 2 GB free disk.
**Software:** Windows 10/11, Python 3.10-3.12, TensorFlow, Streamlit, NumPy, Pillow, Matplotlib, scikit-learn, VS Code, Git.

## 8. Advantages
Fast training thanks to transfer learning; small model; easy to use; supports environmental awareness; extendable.

## 9. Limitations
Small dataset; clean-background images; one object per image; may mislabel unseen items; the "trash" class is vague.

## 10. Future Enhancements
More classes (e-waste, organic, hazardous); larger real-world data; webcam/mobile app; object detection for multiple items; cloud deployment; IoT smart bins.

## 11. Conclusion
WasteWise shows that transfer learning can build a practical waste classifier with limited data and modest hardware, wrapped in a user-friendly app. Report final performance only from your own training run.

## 12. Viva Cheat Sheet
- **Transfer learning?** Reusing a network pretrained on ImageNet and training only a new final layer (then lightly fine-tuning) instead of training from scratch.
- **Why MobileNetV2?** Small, fast, accurate enough, good for laptops and deployment.
- **Why augmentation?** The dataset is small; random flips/rotations make the model robust and reduce overfitting.
- **Why 224x224?** It is MobileNetV2's pretrained input size.
- **Why softmax?** It turns outputs into probabilities summing to 1; the highest is the prediction and its value is the confidence.
- **Why two phases?** Training the new head first avoids destroying pretrained weights; low-lr fine-tuning then adapts them slightly.
- **Confusion matrix?** A table of actual vs predicted classes showing which classes get mixed up (e.g., glass vs plastic).
- **Overfitting?** Training accuracy far above validation accuracy; handled with dropout, augmentation, early stopping.
- **Why no accuracy claimed?** Results must come from running train.py and evaluate.py on your machine.
