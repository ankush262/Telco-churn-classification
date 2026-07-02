import streamlit as st
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from src.serving.inference import predict_with_details

st.set_page_config(page_title="Telco Churn Dashboard", page_icon="📊", layout="wide")

st.title("Telco Churn Prediction Dashboard")
st.write("Fill in the customer details below to estimate churn risk.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
        Partner = st.selectbox("Partner", ["No", "Yes"])
        Dependents = st.selectbox("Dependents", ["No", "Yes"])
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=120, value=12)
        PhoneService = st.selectbox("Phone Service", ["No", "Yes"])
        MultipleLines = st.selectbox("Multiple Lines", ["No", "No phone service", "Yes"])
        InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

    with col2:
        OnlineSecurity = st.selectbox("Online Security", ["No", "No internet service", "Yes"])
        OnlineBackup = st.selectbox("Online Backup", ["No", "No internet service", "Yes"])
        DeviceProtection = st.selectbox("Device Protection", ["No", "No internet service", "Yes"])
        TechSupport = st.selectbox("Tech Support", ["No", "No internet service", "Yes"])
        StreamingTV = st.selectbox("Streaming TV", ["No", "No internet service", "Yes"])
        StreamingMovies = st.selectbox("Streaming Movies", ["No", "No internet service", "Yes"])
        Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        PaperlessBilling = st.selectbox("Paperless Billing", ["No", "Yes"])
        PaymentMethod = st.selectbox("Payment Method", ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"])
        MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
        TotalCharges = st.number_input("Total Charges", min_value=0.0, max_value=5000.0, value=500.0)

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    payload = {
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": int(tenure),
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": float(MonthlyCharges),
        "TotalCharges": float(TotalCharges),
    }

    with st.spinner("Scoring customer..."):
        result = predict_with_details(payload)

    st.success(f"Prediction: {result['prediction']}")
    st.metric("Churn probability", f"{result['probability'] * 100:.1f}%")
    if result["predicted_class"] == 1:
        st.warning("This customer is at high risk of churning.")
    else:
        st.info("This customer appears to be staying with the service.")
