import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import gdown
import os


MODEL_FILE = "coffee_berry_classifier.h5"
FILE_ID = "1tVJnlUIDGtYU7898dT06sH-w3GX7OsMV"  # replace with your real file ID

if not os.path.exists(MODEL_FILE):
    url = f"https://drive.google.com/uc?id=1tVJnlUIDGtYU7898dT06sH-w3GX7OsMV"
    gdown.download(url, MODEL_FILE, quiet=False)

# Now load the model
model = tf.keras.models.load_model(MODEL_FILE)



# Define class labels (update if yours differ)
class_labels = ['Coffee__Berry_borer', 'Coffee__Damaged_bean', 'Coffee__Healthy_bean']

# Preprocessing function
def preprocess_image(image):
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    image = np.asarray(image) / 255.0  # normalize
    image = np.expand_dims(image, axis=0)  # add batch dimension
    return image

# App layout
st.set_page_config(page_title="Coffee Diagnosis App", layout="centered")
st.title("☕ Coffee Diagnosis App")
st.subheader("Upload a Coffee Leaf and/or Coffee Berry Image for Analysis")

# Layout
col1, col2 = st.columns(2)

# Leaf upload (no prediction yet)
with col1:
    st.header("📄 Upload Coffee Leaf")
    leaf_image = st.file_uploader("Choose a coffee leaf image", type=["jpg", "jpeg", "png"], key="leaf")
    if leaf_image:
        leaf_img = Image.open(leaf_image)
        st.image(leaf_img, caption="Coffee Leaf", use_column_width=True)

# Berry upload (with prediction)
with col2:
    st.header("🍒 Upload Coffee Berry")
    berry_image = st.file_uploader("Choose a coffee berry image", type=["jpg", "jpeg", "png"], key="berry")
    if berry_image:
        berry_img = Image.open(berry_image)
        st.image(berry_img, caption="Coffee Berry", use_column_width=True)

        # Predict
        with st.spinner("Analyzing..."):
            processed_img = preprocess_image(berry_img)
            prediction = model.predict(processed_img)[0]
            predicted_class = class_labels[np.argmax(prediction)]
            confidence = np.max(prediction)

        st.success(f"🧠 Prediction: **{predicted_class}**")
        st.caption(f"Confidence: `{confidence * 100:.2f}%`")

st.markdown("---")
st.info("More features coming soon: Leaf disease detection, camera input, and model switching!")
