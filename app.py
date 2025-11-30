import streamlit as st
import numpy as np
import joblib

st.title("Breast Cancer Prediction App 🧠")

st.write("Enter tumor measurement values (30 features) to classify as Benign/Malignant")

inputs = []

for i in range(30):
    value = st.number_input(f"Feature {i+1}", 0.0, 200.0, step=0.01)
    inputs.append(value)

if st.button("Predict"):
    bundle = joblib.load("best_model.joblib")
    model = bundle["model"]
    scaler = bundle["scaler"]

    sample_scaled = scaler.transform([inputs])
    pred = model.predict(sample_scaled)[0]

    if pred == 1:
        st.success("🎉 Result: BENIGN Tumor")
    else:
        st.error("⚠ Result: MALIGNANT Tumor")
