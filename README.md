# ♻️ WasteWise: Deep Learning-Based Waste Classification Using CNN and Transfer Learning

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![Model](https://img.shields.io/badge/Model-MobileNetV2-green)
![App](https://img.shields.io/badge/App-Streamlit-red)

WasteWise is a web application that classifies an image of a waste item into one of **six categories** and shows the prediction with a **confidence score**, class-wise probabilities and a short disposal tip. It uses a CNN built with **MobileNetV2 transfer learning** (TensorFlow/Keras) and a **Streamlit** interface.

## 🌐 Live Demo

**👉 [Open the WasteWise app](https://yashwanthnagireddy-wastewise-deep-learning-waste-cla-app-xw5iv4.streamlit.app/)**

> Hosted on Streamlit Community Cloud. If the app has been idle for a while it goes to sleep. Open the link and wait a minute for it to wake up. The first prediction can be slow while TensorFlow loads.

## 📸 Screenshots

| Home | Prediction |
|---|---|
| ![Home](outputs/app_home.png) | ![Prediction](outputs/app_prediction.png) |

| Accuracy and Loss | Confusion Matrix |
|---|---|
| ![Accuracy and loss](outputs/accuracy_loss_graph.png) | ![Confusion matrix](outputs/confusion_matrix.png) |

## 📌 Table of Contents
1. [Problem Statement](#-problem-statement)
2. [Features](#-features)
3. [Technologies Used](#-technologies-used)
4. [System Architecture](#-system-architecture)
5. [Dataset](#-dataset)
6. [Project Structure](#-project-structure)
7. [Installation (Windows)](#-installation-windows)
8. [Dataset Setup](#-dataset-setup)
9. [Training](#-training)
10. [Evaluation](#-evaluation)
11. [Run the App Locally](#-run-the-app-locally)
12. [Deployment](#-deployment-streamlit-community-cloud)
13. [Results](#-results)
14. [Limitations](#-limitations)
15. [Future Enhancements](#-future-enhancements)
16. [Troubleshooting](#-troubleshooting)
17. [Credits](#-credits)

## ❓ Problem Statement
Waste is often sorted into the wrong bin because people are unsure which category an item belongs to. Wrongly sorted recyclables contaminate recycling streams and increase landfill. WasteWise uses deep learning to identify the waste category from a photo and guide correct disposal.

## ✨ Features
- Upload a JPG/PNG image and see a live preview
- Predicts one of 6 classes: **cardboard, glass, metal, paper, plastic, trash**
- Shows predicted class, **confidence percentage** and **class-wise probabilities**
- Recycling/disposal tip for each category
- Clear / Reset button
- Handles invalid or corrupt images, missing model file and model-loading errors
- Low-confidence warning for unclear images
- Clean green, recycling-themed interface with a category guide in the sidebar

## 🛠 Technologies Used
| Technology | Purpose |
|---|---|
| Python | Main programming language |
| TensorFlow / Keras | Building, training, saving and loading the model |
| MobileNetV2 | Lightweight CNN pretrained on ImageNet (transfer learning) |
| Streamlit | Web interface |
| NumPy | Array handling for images and probabilities |
| Pillow | Opening, validating, converting and resizing images |
| Matplotlib | Accuracy/loss graphs and confusion matrix image |
| scikit-learn | Confusion matrix and classification report |

## 🏗 System Architecture
```
Uploaded image
   -> resize to 224 x 224
   -> data augmentation (training only): flip, rotation, zoom, contrast
   -> scale pixels to [-1, 1]
   -> MobileNetV2 base (ImageNet weights, include_top=False)
   -> GlobalAveragePooling2D -> Dropout(0.3)
   -> Dense(6, softmax)
   -> predicted class + confidence + class probabilities
   -> displayed in the Streamlit app
```

**Training strategy (two phases):**
1. **Phase 1:** MobileNetV2 frozen, only the new classifier head is trained (Adam, learning rate 1e-3, up to 10 epochs).
2. **Phase 2 (fine-tuning):** last 30 layers of MobileNetV2 unfrozen and trained with a very small learning rate (1e-5, up to 10 epochs).

The best model (highest validation accuracy) is saved automatically. Early stopping is used on validation loss.

## 📂 Dataset
- **Dataset:** TrashNet (Gary Thung and Mindy Yang), resized version
- **Size:** 2,527 images, 6 classes
- **Source:** https://huggingface.co/datasets/garythung/trashnet (file `dataset-resized.zip`, about 43 MB). Original project: https://github.com/garythung/trashnet

| Class | Images |
|---|---|
| cardboard | 403 |
| glass | 501 |
| metal | 410 |
| paper | 594 |
| plastic | 482 |
| trash | 137 |

**Split:** 80% training (2,022 images) and 20% validation (505 images), fixed seed `42`, so training and evaluation always use the same split.
**Preprocessing:** resize to 224 x 224, scale to [-1, 1]. **Augmentation:** random horizontal flip, rotation, zoom and contrast.

> The dataset images are **not** stored in this repository. Download them as described below.

## 🗂 Project Structure
```
WasteWise-Deep-Learning-Waste-Classification/
├── app.py                  # Streamlit web application
├── train.py                # Two-phase training script
├── evaluate.py             # Confusion matrix + classification report
├── common.py               # Shared settings (paths, image size, seed, split)
├── requirements.txt        # Python dependencies (pinned)
├── README.md
├── .gitignore
├── data/                   # Dataset (images NOT in Git)
│   ├── cardboard/
│   ├── glass/
│   ├── metal/
│   ├── paper/
│   ├── plastic/
│   └── trash/
├── models/
│   ├── wastewise_model.keras   # Trained model (about 19 MB, included for deployment)
│   └── class_names.json
├── outputs/
│   ├── accuracy_loss_graph.png
│   ├── confusion_matrix.png
│   ├── classification_report.txt
│   ├── app_home.png
│   └── app_prediction.png
└── docs/
    ├── ASSIGNMENT_DOCUMENTATION.md
    └── GIT_AND_GITHUB_GUIDE.md
```

## 💻 Installation (Windows)
Requirements: **Python 3.11** (TensorFlow supports 3.10 to 3.12) and Git.

```bash
git clone https://github.com/YashwanthNagireddy/WasteWise-Deep-Learning-Waste-Classification.git
cd WasteWise-Deep-Learning-Waste-Classification
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```
If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first, or skip the virtual environment.

Check the install:
```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

## 📥 Dataset Setup
Only needed if you want to **train or evaluate** (the app itself only needs the included model).

1. Download `dataset-resized.zip` from the [Hugging Face dataset page](https://huggingface.co/datasets/garythung/trashnet).
2. Extract it. You should get six folders: `cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`.
3. Copy them into the `data` folder so the images sit directly inside each class folder:
```bash
xcopy "C:\path\to\dataset-resized\dataset-resized\*" "data\" /E /I /Y
```
Final layout must be `data\glass\glass1.jpg`, not `data\dataset-resized\glass\...`.

4. Verify the counts:
```bash
python -c "import os; [print(c, len(os.listdir('data/'+c))) for c in sorted(os.listdir('data')) if os.path.isdir('data/'+c)]"
```
(Each folder shows one extra file because of the `.gitkeep` placeholder.)

## 🏋️ Training
```bash
python train.py
```
This will:
- check the dataset folders,
- run Phase 1 and Phase 2 training,
- save the best model to `models/wastewise_model.keras`,
- save `models/class_names.json` and `outputs/accuracy_loss_graph.png`,
- print the best validation accuracy.

Training runs on the CPU on Windows and can take a while. For a quick test, lower both `epochs=10` values in `train.py`.

## 📊 Evaluation
```bash
python evaluate.py
```
Creates `outputs/confusion_matrix.png` and `outputs/classification_report.txt` using the same validation images that were held out during training.

## 🚀 Run the App Locally
```bash
streamlit run app.py
```
Open http://localhost:8501, upload an image, and click **Predict**. Press Enter if Streamlit asks for an optional email.

## ☁️ Deployment (Streamlit Community Cloud)
The live demo is deployed from this repository.

1. Push the project (including `models/wastewise_model.keras` and `models/class_names.json`) to GitHub. The model is added with `git add -f` because `.gitignore` excludes model files by default.
2. Go to https://share.streamlit.io, sign in with GitHub and click **Create app**.
3. Select this repository, branch `main`, main file `app.py`.
4. In **Advanced settings**, choose **Python 3.11**, then click **Deploy**.

Every `git push` redeploys the app automatically.

## 📈 Results
> **Fill this in with your own numbers** from the final line of `python train.py` and from `outputs/classification_report.txt`. Results are not claimed here until they come from your own training run.

| Metric | Value |
|---|---|
| Training images / validation images | 2,022 / 505 |
| Best validation accuracy | `XX.XX%` |
| Validation loss | `X.XXXX` |

**Class-wise performance (from `classification_report.txt`):**

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| cardboard | | | |
| glass | | | |
| metal | | | |
| paper | | | |
| plastic | | | |
| trash | | | |

**Observations:** *(write 2 to 3 lines from your confusion matrix, for example which classes are confused most and how the small `trash` class performed)*

> Note: the sample images in the live demo may come from the training set, so demo predictions are not a fair accuracy test. Use the validation results above.

## ⚠️ Limitations
- Small dataset (2,527 images) with plain, clean backgrounds, so real-world photos may give lower accuracy.
- One item per image; the app cannot detect several objects at once.
- The `trash` class is a mixed leftover class with only 137 images, so it is the hardest to learn.
- Training on CPU limits how much experimentation is practical.
- Predictions can be wrong and should be treated as guidance, not a final decision.

## 🔮 Future Enhancements
- More categories (e-waste, organic, hazardous, batteries)
- Larger and more realistic training data with varied backgrounds
- Webcam/real-time prediction and a mobile app
- Object detection (for example YOLO) to classify multiple items in one image
- Model comparison (EfficientNet, ResNet) and class-imbalance handling
- Integration with smart bins (IoT)

## 🧰 Troubleshooting
| Problem | Fix |
|---|---|
| `Python was not found` | Install Python 3.11, tick *Add to PATH*, open a new terminal |
| TensorFlow DLL load failed | Install/repair the Microsoft Visual C++ Redistributable (x64) and restart |
| PowerShell blocks `Activate.ps1` | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |
| `Model file ... not found` | Run `python train.py`, or make sure `models/wastewise_model.keras` is in the repo |
| Streamlit shows `ERR_CONNECTION_REFUSED` | Press Enter at the email prompt so the server starts |
| Harmless TensorFlow messages (oneDNN, CPU instructions, GPU not available on Windows) | Can be ignored |

## 🙏 Credits
- **Dataset:** [TrashNet](https://github.com/garythung/trashnet) by Gary Thung and Mindy Yang (MIT licensed). If you use the dataset, please cite their repository.
- **Model:** MobileNetV2 pretrained on ImageNet, via Keras Applications.
- **Author:** Nagireddy Yashwanth Reddy ([@YashwanthNagireddy](https://github.com/YashwanthNagireddy)), B.Tech Computer Science.

*Built as a final-year college project for educational purposes.*
