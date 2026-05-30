import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Load model and scaler
model = load_model("churn_model.h5")
scaler = joblib.load("scaler.pkl")

st.title("📊 Customer Churn Prediction")
st.write("Predict whether a customer is likely to churn.")

st.sidebar.header("Customer Information")

gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
senior = st.sidebar.selectbox("Senior Citizen", [0, 1])
partner = st.sidebar.selectbox("Partner", ["No", "Yes"])
dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"])
tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)

phone_service = st.sidebar.selectbox("Phone Service", ["No", "Yes"])
multiple_lines = st.sidebar.selectbox(
    "Multiple Lines",
    ["No", "Yes", "No phone service"]
)

internet_service = st.sidebar.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

online_security = st.sidebar.selectbox(
    "Online Security",
    ["No", "Yes", "No internet service"]
)

online_backup = st.sidebar.selectbox(
    "Online Backup",
    ["No", "Yes", "No internet service"]
)

device_protection = st.sidebar.selectbox(
    "Device Protection",
    ["No", "Yes", "No internet service"]
)

tech_support = st.sidebar.selectbox(
    "Tech Support",
    ["No", "Yes", "No internet service"]
)

streaming_tv = st.sidebar.selectbox(
    "Streaming TV",
    ["No", "Yes", "No internet service"]
)

streaming_movies = st.sidebar.selectbox(
    "Streaming Movies",
    ["No", "Yes", "No internet service"]
)

contract = st.sidebar.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

paperless_billing = st.sidebar.selectbox(
    "Paperless Billing",
    ["No", "Yes"]
)

payment_method = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

monthly_charges = st.sidebar.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0
)

total_charges = st.sidebar.number_input(
    "Total Charges",
    min_value=0.0,
    value=1000.0
)

# Encoding mappings
mappings = {
    "gender": {"Female": 0, "Male": 1},
    "partner": {"No": 0, "Yes": 1},
    "dependents": {"No": 0, "Yes": 1},
    "phone_service": {"No": 0, "Yes": 1},
    "multiple_lines": {
        "No": 0,
        "No phone service": 1,
        "Yes": 2
    },
    "internet_service": {
        "DSL": 0,
        "Fiber optic": 1,
        "No": 2
    },
    "online_security": {
        "No": 0,
        "No internet service": 1,
        "Yes": 2
    },
    "online_backup": {
        "No": 0,
        "No internet service": 1,
        "Yes": 2
    },
    "device_protection": {
        "No": 0,
        "No internet service": 1,
        "Yes": 2
    },
    "tech_support": {
        "No": 0,
        "No internet service": 1,
        "Yes": 2
    },
    "streaming_tv": {
        "No": 0,
        "No internet service": 1,
        "Yes": 2
    },
    "streaming_movies": {
        "No": 0,
        "No internet service": 1,
        "Yes": 2
    },
    "contract": {
        "Month-to-month": 0,
        "One year": 1,
        "Two year": 2
    },
    "paperless_billing": {
        "No": 0,
        "Yes": 1
    },
    "payment_method": {
        "Bank transfer (automatic)": 0,
        "Credit card (automatic)": 1,
        "Electronic check": 2,
        "Mailed check": 3
    }
}

input_data = pd.DataFrame([{
    "gender": mappings["gender"][gender],
    "SeniorCitizen": senior,
    "Partner": mappings["partner"][partner],
    "Dependents": mappings["dependents"][dependents],
    "tenure": tenure,
    "PhoneService": mappings["phone_service"][phone_service],
    "MultipleLines": mappings["multiple_lines"][multiple_lines],
    "InternetService": mappings["internet_service"][internet_service],
    "OnlineSecurity": mappings["online_security"][online_security],
    "OnlineBackup": mappings["online_backup"][online_backup],
    "DeviceProtection": mappings["device_protection"][device_protection],
    "TechSupport": mappings["tech_support"][tech_support],
    "StreamingTV": mappings["streaming_tv"][streaming_tv],
    "StreamingMovies": mappings["streaming_movies"][streaming_movies],
    "Contract": mappings["contract"][contract],
    "PaperlessBilling": mappings["paperless_billing"][paperless_billing],
    "PaymentMethod": mappings["payment_method"][payment_method],
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges
}])

if st.button("Predict Churn"):
    scaled_data = scaler.transform(input_data)

    probability = model.predict(scaled_data)[0][0]
    prediction = int(probability > 0.5)

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Customer is likely to churn")
    else:
        st.success("✅ Customer is likely to stay")

    st.metric(
        "Churn Probability",
        f"{probability * 100:.2f}%"
    )
