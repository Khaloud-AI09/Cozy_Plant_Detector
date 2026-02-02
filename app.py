import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# 1. Page Config
st.set_page_config(page_title="Cozy Plant Care", page_icon="🌿")

# 2. Simple, Error-Free Aesthetic CSS
st.markdown("""
    <style>
    .stApp { background-color: #fdf6e3; }
    h1 { color: #6b8e23; }
    </style>
    """, unsafe_allow_html=True)

st.title("🌿 Cozy Plant Thirst Detector")

# 3. Load Model Safely
@st.cache_resource
def load_my_model():
    try:
        return load_model("keras_model.h5", compile=False)
    except Exception as e:
        st.error(f"Model failed to load: {e}")
        return None

model = load_my_model()

# Load labels
try:
    with open("labels.txt", "r") as f:
        class_names = f.readlines()
except FileNotFoundError:
    st.error("Missing labels.txt file!")
    class_names = []

# 4. App Logic
img_file = st.file_uploader("Upload your plant photo...", type=["jpg", "png", "jpeg"])

if img_file is not None and model is not None:
    image = Image.open(img_file).convert("RGB")
    st.image(image, caption="Checking your plant...", use_container_width=True)
    
    # Preprocessing
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image)
    normalized_img_array = (img_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_img_array

    # Prediction
    prediction = model.predict(data)
    index = np.argmax(prediction)
    
    if class_names:
        class_name = class_names[index].strip()
        confidence = prediction[0][index]

        st.divider()
        if "Thirsty" in class_name:
            st.error(f"✨ Status: {class_name}")
            st.write("Give it some water! 💧")
        else:
            st.success(f"✨ Status: {class_name}")
            st.balloons() # Confetti/Balloon effect added!
            st.write("Your plant is happy! ✨")
