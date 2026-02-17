# ==========================================
# MOBILE MONEY FRAUD RISK PREDICTOR APP
# ==========================================

import streamlit as st
import pickle
import pandas as pd
import numpy as np

# ------------------------------
# Load saved model files
# ------------------------------

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))

# ------------------------------
# Page configuration
# ------------------------------

st.set_page_config(
    page_title="Mobile Money Fraud Detector",
    page_icon="💳",
    layout="wide"
)

# ------------------------------
# Custom Styling
# ------------------------------

st.markdown(
    """
    <style>
    .main {
        background-color: #0f172a;
        color: white;
    }
    .stButton>button {
        background-color: #22c55e;
        color: white;
        font-size: 16px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------
# App Title
# ------------------------------

st.title("💳 Mobile Money Transaction Fraud Risk Predictor")
st.write("AI-powered fraud detection system")

# ------------------------------
# User Inputs
# ------------------------------

col1, col2 = st.columns(2)

with col1:
    amount = st.number_input("Transaction Amount", min_value=100, max_value=100000)
    time = st.slider("Transaction Time (Hour)", 0, 23)

with col2:
    location = st.selectbox("Location", ["Urban", "Rural", "International"])
    frequency = st.slider("Transaction Frequency", 1, 20)

# ------------------------------
# Prediction
# ------------------------------

if st.button("Analyze Transaction"):

    location_encoded = encoder.transform([location])[0]

    input_data = pd.DataFrame({
        "Amount": [amount],
        "Time": [time],
        "Location": [location_encoded],
        "Frequency": [frequency]
    })

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"⚠ HIGH RISK Transaction\nFraud Probability: {probability:.2f}")
    else:
        st.success(f"✅ SAFE Transaction\nFraud Probability: {probability:.2f}")
