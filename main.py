import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Page Configuration
st.set_page_config(
    page_title="Heart Health Analyzer", 
    page_icon="💓", 
    layout="wide"
)

# Custom Dark Theme CSS
st.markdown("""
    <style>
    /* Dark Theme Base */
    body {
        background-color: #121212;
        color: #E0E0E0;
    }
    
    /* Custom Dark Theme Styling */
    .main-container {
        background-color: #1E1E1E;
        border-radius: 15px;
        padding: 30px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    }
    
    .main-title {
        color: #4CAF50;
        text-align: center;
        font-size: 3em;
        margin-bottom: 20px;
        text-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    .subtitle {
        color: #81C784;
        text-align: center;
        font-size: 1.3em;
        margin-bottom: 30px;
    }
    
    /* Input Styling */
    .stNumberInput > div > div > input, 
    .stSelectbox > div > div > div > select {
        background-color: #2C2C2C !important;
        color: #E0E0E0 !important;
        border: 1px solid #404040 !important;
        border-radius: 8px !important;
    }
    
    /* Button Styling */
    .stButton > button {
        background-color: #4CAF50 !important;
        color: white !important;
        font-size: 1.1em;
        border-radius: 10px;
        padding: 10px 20px;
        width: 100%;
        transition: all 0.3s ease;
        border: none !important;
    }
    
    .stButton > button:hover {
        background-color: #45A049 !important;
        transform: scale(1.02);
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
    }
    
    /* Prediction Box Styling */
    .prediction-box {
        background-color: #2C2C2C;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        font-size: 1.2em;
        color: #E0E0E0;
        border: 2px solid #4CAF50;
    }
    
    /* Sidebar Styling */
    .css-1aumxhk {
        background-color: #1A1A1A !important;
    }
    
    /* Expander Styling */
    .stExpander {
        background-color: #2C2C2C;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    with open('heart.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()

# Function to make predictions
def predict_heart_attack(features):
    prediction = model.predict([features])
    return prediction[0]

# Sidebar for feature explanations
def feature_sidebar():
    st.sidebar.header("🔍 Feature Insights")
    feature_explanations = {
        "Age": "Your current age - risk increases with age.",
        "Sex": "Biological sex can influence heart disease risk.",
        "Chest Pain Type": "Different types of chest pain indicate varying cardiovascular conditions.",
        "Resting Blood Pressure": "High blood pressure is a key heart disease indicator.",
        "Cholesterol": "Elevated cholesterol levels can increase heart disease risk.",
        "Fasting Blood Sugar": "High blood sugar can damage blood vessels.",
        "Resting ECG": "Electrocardiogram results provide insights into heart health.",
        "Max Heart Rate": "Maximum heart rate reflects cardiovascular fitness.",
        "Exercise Induced Angina": "Chest pain during exercise can signal heart issues.",
        "Major Vessels": "Number of major blood vessels affects heart function."
    }
    
    for feature, explanation in feature_explanations.items():
        with st.sidebar.expander(feature):
            st.write(explanation)

# Main app
def main():
    # Container for main content
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # Title and Introduction
    st.markdown('<h1 class="main-title">💓 Heart Health Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Advanced Machine Learning Risk Assessment</p>', unsafe_allow_html=True)
    
    # Sidebar feature explanations
    feature_sidebar()
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        # Collect user input - First Column
        age = st.number_input("🎂 Age", min_value=1, max_value=120, value=25, help="Enter your current age")
        sex = st.selectbox("♀♂ Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        cp = st.selectbox("�胸 Chest Pain Type", [0, 1, 2, 3], format_func={
            0: "Typical Angina", 
            1: "Atypical Angina", 
            2: "Non-Anginal Pain", 
            3: "Asymptomatic"
        }.get)
        trtbps = st.number_input("📊 Resting Blood Pressure (mm Hg)", min_value=50, max_value=200, value=120)
        chol = st.number_input("🩺 Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
    
    with col2:
        # Collect user input - Second Column
        fbs = st.selectbox("🍬 Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        restecg = st.selectbox("❤️ Resting ECG Results", [0, 1, 2], format_func={
            0: "Normal", 
            1: "ST-T Wave Abnormality", 
            2: "Probable Left Ventricular Hypertrophy"
        }.get)
        thalachh = st.number_input("💪 Maximum Heart Rate", min_value=60, max_value=220, value=150)
        exng = st.selectbox("🏃 Exercise Induced Angina", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        caa = st.selectbox("🩸 Major Vessels Colored", [0, 1, 2, 3], help="Number of major vessels colored by fluoroscopy")
    
    # Prepare features for prediction
    features = [age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, caa]
    
    # Predict button
    if st.button("🔬 Analyze Heart Health Risk"):
        prediction = predict_heart_attack(features)
        
        # Prediction result with styling
        if prediction == 1:
            st.markdown('<div class="prediction-box" style="border-color: #FF5252;">⚠️ Potential Heart Disease Risk Detected</div>', unsafe_allow_html=True)
            st.error("🚨 Higher risk identified. Immediate medical consultation recommended.")
        else:
            st.markdown('<div class="prediction-box" style="border-color: #4CAF50;">✅ Low Heart Disease Risk</div>', unsafe_allow_html=True)
            st.success("👍 Cardiovascular health appears stable. Continue maintaining a healthy lifestyle.")
    
    # Close main container
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("---")
    st.warning("⚠️ Disclaimer: This is a predictive tool for screening purposes only. Always consult healthcare professionals for comprehensive diagnosis.")

if __name__ == "__main__":
    main()