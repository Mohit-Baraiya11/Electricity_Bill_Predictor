import streamlit as st
import numpy as np
import joblib

# Cache resources so they load only once
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

st.set_page_config(page_title="Electricity Bill Predictor", page_icon="⚡", layout="centered")
st.title("⚡ Electricity Bill Predictor")
st.write("Enter the units consumed to estimate your electricity bill.")

model, scaler = load_model_and_scaler()

# Input widget
units = st.number_input("Units Consumed (kWh)", min_value=0, step=1, value=100)

if st.button("Predict"):
    # Scale input
    X_new = np.array([[units]], dtype=float)
    X_scaled = scaler.transform(X_new)

    # Predict using the model
    pred = model.predict(X_scaled)[0]

    st.success(f"Estimated Bill: ₹ {pred:,.2f}")
    st.caption("Note: This is an estimate based on your training data.")

# About section
with st.expander("About this model"):
    st.write(
        "This app uses a Linear Regression model with StandardScaler. "
        "Your input is scaled using the same scaler as in training before prediction."
    )
