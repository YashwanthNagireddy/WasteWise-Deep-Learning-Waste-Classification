"""
app.py - WasteWise Streamlit web application.

Run:  streamlit run app.py
"""
import json
import os

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

from common import CLASSES_PATH, EXPECTED_CLASSES, IMG_SIZE, MODEL_PATH

st.set_page_config(page_title="WasteWise", page_icon="♻️", layout="wide")

# Info shown for each category
INFO = {
    "cardboard": ("📦", "Recyclable", "Flatten boxes and keep them dry. Remove tape and food residue."),
    "glass":     ("🍾", "Recyclable", "Rinse bottles and jars. Do not mix with ceramics or window glass."),
    "metal":     ("🥫", "Recyclable", "Rinse cans and tins. Aluminium and steel are highly recyclable."),
    "paper":     ("📄", "Recyclable", "Keep clean and dry. Greasy or wet paper is hard to recycle."),
    "plastic":   ("🧴", "Recyclable (check the code)", "Rinse containers, check the recycling number, and remove caps if required."),
    "trash":     ("🗑️", "Non-recyclable / general waste", "Dispose of in general waste. Consider reducing such items."),
}

st.markdown("""
<style>
.hero {background: linear-gradient(135deg,#1b5e20,#43a047); padding: 28px 32px;
       border-radius: 16px; color: white; margin-bottom: 18px;}
.hero h1 {margin: 0; font-size: 2.3rem;}
.hero p {margin: 6px 0 0 0; opacity: .95;}
.result {background:#e8f5e9; border-left:8px solid #2e7d32; padding:18px 22px;
         border-radius:12px; color:#1b5e20;}
.result h2 {margin:0; color:#1b5e20;}
div.stButton > button {background:#2e7d32; color:white; border-radius:10px; border:none;
                       padding:.5rem 1.4rem; font-weight:600;}
div.stButton > button:hover {background:#1b5e20; color:white;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the trained model once. Returns (model, error_message)."""
    if not os.path.isfile(MODEL_PATH):
        return None, f"Model file '{MODEL_PATH}' not found. Run `python train.py` first."
    try:
        import tensorflow as tf
        return tf.keras.models.load_model(MODEL_PATH), None
    except Exception as e:
        return None, f"Could not load the model: {e}"


def get_class_names():
    try:
        with open(CLASSES_PATH) as f:
            return json.load(f)
    except Exception:
        return EXPECTED_CLASSES


def predict(model, image, class_names):
    """Resize image to 224x224 and return the probability for every class."""
    arr = np.array(image.resize(IMG_SIZE), dtype="float32")
    arr = np.expand_dims(arr, axis=0)        # (1, 224, 224, 3); scaling is inside the model
    probs = model.predict(arr, verbose=0)[0]
    return dict(zip(class_names, probs))


# Session state: changing the uploader key clears the uploaded file (Reset button)
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# ---------- Header ----------
st.markdown("""<div class="hero"><h1>♻️ WasteWise</h1>
<p>Deep Learning-based waste classification using CNN (MobileNetV2) and Transfer Learning.
Upload a photo of a waste item and let AI tell you how to sort it. 🌍</p></div>""",
            unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("🌱 Waste Categories")
    for name in EXPECTED_CLASSES:
        icon, kind, tip = INFO[name]
        with st.expander(f"{icon} {name.capitalize()}"):
            st.write(f"**{kind}**")
            st.write(tip)
    st.caption("Tip: use a clear photo with one item and a plain background.")

model, error = load_model()
if error:
    st.error(error)
    st.stop()
class_names = get_class_names()

# ---------- Main layout ----------
left, right = st.columns(2, gap="large")

with left:
    st.subheader("📤 Upload Image")
    file = st.file_uploader("Choose a JPG/PNG image", type=["jpg", "jpeg", "png"],
                            key=f"uploader_{st.session_state.uploader_key}")
    image = None
    if file is not None:
        try:
            image = Image.open(file)
            image.load()                      # forces decoding; raises on corrupt files
            image = image.convert("RGB")
            st.image(image, caption="Preview", use_container_width=True)
        except (UnidentifiedImageError, OSError):
            st.error("This file is not a valid image. Please upload another one.")
            image = None

    c1, c2 = st.columns(2)
    predict_clicked = c1.button("🔍 Predict", disabled=image is None, use_container_width=True)
    if c2.button("🧹 Clear / Reset", use_container_width=True):
        st.session_state.uploader_key += 1
        st.rerun()

with right:
    st.subheader("📊 Result")
    if image is None:
        st.info("Upload an image and click **Predict**.")
    elif predict_clicked:
        with st.spinner("Analysing..."):
            try:
                scores = predict(model, image, class_names)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()
        best = max(scores, key=scores.get)
        icon, kind, tip = INFO.get(best, ("♻️", "", ""))
        st.markdown(f"""<div class="result"><h2>{icon} {best.capitalize()}</h2>
        <p><b>Confidence: {scores[best] * 100:.2f}%</b><br>{kind}<br>{tip}</p></div>""",
                    unsafe_allow_html=True)
        if scores[best] < 0.5:
            st.warning("Low confidence. Try a clearer image of a single item.")
        st.markdown("#### Class-wise probabilities")
        for name, p in sorted(scores.items(), key=lambda kv: -kv[1]):
            st.write(f"{name.capitalize()} — {p * 100:.2f}%")
            st.progress(float(p))
    else:
        st.info("Image ready. Click **Predict**.")

st.markdown("---")
st.caption("WasteWise • College project • Predictions depend on the training data and may be wrong.")
