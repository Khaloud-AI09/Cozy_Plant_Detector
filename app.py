import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="Cozy Plant Care", page_icon="🌿")

# --- CUSTOM CSS FOR AESTHETICS ---

# --- CUSTOM CSS FOR AESTHETICS ---
st.markdown("""
    <style>
    .stApp { background-color: #fdf6e3; }
    h1 { color: #6b8e23; font-family: 'Segoe UI'; }
    .stButton>button { background-color: #a3b18a; color: white; border-radius: 20px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🌿 Cozy Plant Thirst Detector")
st.write("Upload a photo of your green friend to see if it needs a drink!")

# --- LOAD MODEL ---
@st.cache_resource
def load_my_model():
    return load_model("keras_model.h5", compile=False)

model = load_my_model()
class_names = open("labels.txt", "r").readlines()

# --- IMAGE UPLOADER ---
img_file = st.file_uploader("Choose a plant photo...", type=["jpg", "png", "jpeg"])

if img_file is not None:
    image = Image.open(img_file).convert("RGB")
    st.image(image, caption="Checking your plant...", use_container_width=True)
    
    # Preprocessing the image to match Teachable Machine's format
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image)
    normalized_img_array = (img_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_img_array

    # Prediction
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    # --- DISPLAY RESULT ---
    st.divider()
    if "Thirsty" in class_name:
        st.error(f"✨ **Status:** {class_name}")
        st.write("Time for some water! 💧 Your plant is looking a bit parched.")
    else:
        st.success(f"✨ **Status:** {class_name}")
        st.write("Your plant is thriving! Keep up the good work. ✨")
    

    st.info(f"Confidence: {round(confidence_score * 100)}%")
