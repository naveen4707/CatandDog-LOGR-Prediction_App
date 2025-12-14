import streamlit as st
import pickle
import cv2
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------
# 1. PAGE CONFIG & STYLING
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Paws & Claws Classifier",
    page_icon="🐾",
    layout="centered"
)

# Custom CSS for Background and UI Styling
page_bg_img = """
<style>
/* 1. The Background Image (Cat and Dog Pattern) */
[data-testid="stAppViewContainer"] {
    background-image: url("https://img.freepik.com/free-vector/seamless-pattern-with-cute-cats-dogs_1284-43405.jpg?w=1380&t=st=1708800000~exp=1708800600~hmac=fake_token");
    background-size: 400px; /* Controls size of the pattern */
    background-repeat: repeat;
    background-attachment: fixed;
}

/* 2. The Semi-Transparent Overlay (Glassmorphism) */
[data-testid="stHeader"] {
    background-color: rgba(0,0,0,0);
}

.block-container {
    background-color: rgba(255, 255, 255, 0.9);
    padding: 3rem;
    border-radius: 25px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    margin-top: 2rem;
}

/* 3. Fonts and Text Styling */
h1 {
    color: #FF6F61;
    font-family: 'Comic Sans MS', 'Chalkboard SE', sans-serif;
    text-align: center;
    text-shadow: 2px 2px #ffecec;
}

p {
    font-size: 1.1rem;
    color: #333;
    text-align: center;
}

/* 4. Button Styling */
div.stButton > button {
    background-color: #FF6F61;
    color: white;
    font-size: 20px;
    padding: 10px 24px;
    border-radius: 12px;
    border: none;
    width: 100%;
}
div.stButton > button:hover {
    background-color: #ff8a7f;
    color: white;
    border: none;
}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

# ---------------------------------------------------------------------
# 2. MODEL LOADING & PREPROCESSING
# ---------------------------------------------------------------------
IMG_SIZE = 64

@st.cache_resource
def load_model():
    try:
        with open('cat_dog_model.pkl', 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        return None

model = load_model()

def preprocess_image(image):
    # Convert PIL to OpenCV (numpy)
    img_array = np.array(image)
    
    # RGB to Grayscale
    if len(img_array.shape) == 3:
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img_array
        
    # Resize to 64x64 (Same as training)
    img_resized = cv2.resize(img_gray, (IMG_SIZE, IMG_SIZE))
    
    # Flatten (1, 4096)
    img_flat = img_resized.flatten().reshape(1, -1)
    return img_flat

# ---------------------------------------------------------------------
# 3. APP UI
# ---------------------------------------------------------------------

# Title
st.title("🐾 Paws or Claws? 🐾")
st.markdown("**Upload a photo and our AI will sniff out if it's a Cat or a Dog!**")

if model is None:
    st.error("⚠️ Model file not found! Please make sure 'cat_dog_model.pkl' is in the folder.")
else:
    # Upload Area
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Layout columns to center the image
        col1, col2, col3 = st.columns([1, 2, 1])
        
        image = Image.open(uploaded_file)
        
        with col2:
            st.image(image, caption='Your Pet Photo', use_container_width=True)
            
            # Add some spacing
            st.write("") 
            
            if st.button("🔍 Analyze Image"):
                with st.spinner("Sniffing..."):
                    try:
                        # Process
                        data = preprocess_image(image)
                        # Predict
                        result = model.predict(data)[0]
                        
                        # Result Display
                        if result == 1:
                            st.balloons()
                            st.markdown(
                                """
                                <div style='text-align: center; color: #4CAF50; background-color: #e8f5e9; padding: 20px; border-radius: 15px; margin-top: 20px;'>
                                    <h2 style='margin:0;'>🐶 It's a DOG!</h2>
                                    <p>Such a good boy/girl!</p>
                                </div>
                                """, 
                                unsafe_allow_html=True
                            )
                        else:
                            st.snow()
                            st.markdown(
                                """
                                <div style='text-align: center; color: #2196F3; background-color: #e3f2fd; padding: 20px; border-radius: 15px; margin-top: 20px;'>
                                    <h2 style='margin:0;'>🐱 It's a CAT!</h2>
                                    <p>Purr-fect detection!</p>
                                </div>
                                """, 
                                unsafe_allow_html=True
                            )
                    except Exception as e:
                        st.error(f"Oof! Something went wrong: {e}")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: grey; font-size: 0.8rem;'>Built with Streamlit & Logistic Regression</div>", 
    unsafe_allow_html=True
)
