import streamlit as st
import os
import zipfile
import pandas as pd
import joblib
import sklearn.compose._column_transformer as ct

class _RemainderColsList:
    pass

ct._RemainderColsList = _RemainderColsList

if not os.path.exists("churn_model.joblib"):
    with zipfile.ZipFile("model.zip", 'r') as zip_ref:
        zip_ref.extractall()


# Load the trained model
model = joblib.load("churn_model.joblib")

st.set_page_config(page_title="Churn Prediction", layout="centered")
st.title("📉 Customer Churn Prediction")

st.subheader("Enter Customer Details:")

# User Inputs
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 100, 35)
tenure = st.slider("Tenure (in months)", 0, 72, 12)
usage_freq = st.slider("Usage Frequency (times/month)", 0, 30, 10)
support_calls = st.slider("Number of Support Calls", 0, 20, 3)
payment_delay = st.slider("Payment Delay (in days)", 0, 30, 5)
subscription_type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
contract_length = st.selectbox("Contract Length", ["Monthly", "Quarterly", "Annual"])
total_spend = st.number_input("Total Spend ($)", min_value=0.0, max_value=10000.0, value=200.0)
last_interaction = st.slider("Days Since Last Interaction", 0, 30, 7)

# Predict button
if st.button("🔍 Predict Churn"):
    # Create DataFrame from user input
    input_df = pd.DataFrame({
        "Gender": [gender],
        "Age": [age],
        "Tenure": [tenure],
        "Usage Frequency": [usage_freq],
        "Support Calls": [support_calls],
        "Payment Delay": [payment_delay],
        "Subscription Type": [subscription_type],
        "Contract Length": [contract_length],
        "Total Spend": [total_spend],
        "Last Interaction": [last_interaction]
    })

    # Make prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    # Display result
    st.subheader("Prediction Result:")
    if prediction == 1:
        st.error(f"❌ The customer is **likely to churn**.\nConfidence: {probability:.2f}")
    else:
        st.success(f"✅ The customer is **likely to stay**.\nConfidence: {1 - probability:.2f}")
