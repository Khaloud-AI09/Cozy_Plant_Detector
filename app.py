import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
from google import genai  # Added missing import

# 1. Page Config
st.set_page_config(page_title="Plant Care", page_icon="🪹")

# 2. Aesthetic CSS
st.markdown("""
    <style>
    .stApp { background-color: #EBF4DD; }
    h1 { color: #5A7863; }
    .stCamera > div > div > button { background-color: #EBF4DD !important; color: white !important; }
    </style>
    """, unsafe_allow_html=True)

st.title("Plant Thirst Detector")

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

# 4. Camera and File Input (Unified Input Section)
st.write("### Snap a photo or upload one!")
tab1, tab2 = st.tabs(["📸 Camera", "📁 Upload File"])

img_file = None

with tab1:
    camera_img = st.camera_input("Take a picture of your plant")
    if camera_img:
        img_file = camera_img

with tab2:
    upload_img = st.file_uploader("Choose a plant photo...", type=["jpg", "png", "jpeg"])
    if upload_img:
        img_file = upload_img

# 5. Prediction Logic (Your Keras Model)
if img_file is not None and model is not None:
    image = Image.open(img_file).convert("RGB")
    
    # Show the user what they captured
    st.image(image, caption="Analyzing this beauty...", use_container_width=True)
    
    # Preprocessing
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image)
    normalized_img_array = (img_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_img_array

    # Prediction
    with st.spinner('Thinking...'):
        prediction = model.predict(data)
        index = np.argmax(prediction)
    
    if class_names:
        class_name = class_names[index].strip()
        confidence = prediction[0][index]

        st.divider()
        if "Thirsty" in class_name:
            st.error(f" Status: {class_name}")
            st.write("Give it some water! 💧 Your plant is thirsty.")
        else:
            st.success(f" Status: {class_name}")
            st.balloons() 
            st.write("Your plant is happy and healthy! ")
        
        st.caption(f"Confidence score: {round(confidence * 100)}%")

# 6. Advanced AI Diagnostics Section (Using the SAME image)
st.divider()
st.title("🌱 AI Plant Care & Diagnostics")

if img_file is not None:
    # Simple trigger button so it only runs when your dad clicks it
    if st.button("🧠 Analyze Plant Health & Care"):
        with st.spinner(" Consulting the AI Botanist..."):
            try:
                # Re-open the unified file for Gemini API
                img = Image.open(img_file)
                
                # Initialize client (grabs standard GEMINI_API_KEY from secrets/env)
                client = genai.Client()
                
                # Prompt tailoring specific care instructions
                prompt = """
                Analyze this plant photo carefully. Provide a highly actionable, 
                gardening-focused response under these exact Markdown headers:
                
                ### 🏷️ Plant Identity
                Identify the exact common name and variety.
                
                ### 🏥 Health & Disease Report
                Look closely at leaves, stems, and soil. Note if there are any signs of:
                - Fungal/bacterial infections
                - Pest infestations
                - Nutrient deficiencies
                If diseased, give step-by-step instructions to treat it.
                
                ### 💧 Optimal Watering Schedule
                How often and exactly how much water does this specific plant prefer?
                
                ### 🪵 Manure & Fertilizer Guide
                What specific type of manure or fertilizer does it love, how much should be applied, and at what frequency?
                """
                
                # Generate content using the multimodal flash model
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=[img, prompt]
                )
                
                # Output cleanly format markdown
                st.success("Analysis Complete!")
                st.markdown(response.text)
                
            except Exception as e:
                st.error(f"Something went wrong: {e}")
else:
    st.info("Upload or snap a photo above to unlock full AI Care Diagnostics.")
