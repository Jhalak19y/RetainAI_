from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st
from src.data.load_data import load_data
from src.explain.shap_explainer import ChurnExplainer

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "churn_model.joblib"

@st.cache_data
def get_data():
    return load_data()

@st.cache_resource
def get_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

@st.cache_resource
def get_explainer(_model, background_hash):
    df = get_data()
    X = df.drop(columns="Churn")
    bg = X.sample(min(40, len(X)), random_state=42)
    return ChurnExplainer(_model, bg)

def get_metrics():
    path = ROOT / "models" / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else None

def make_customer_form():
    return {
        "gender": st.selectbox("Gender", ["Female", "Male"]),
        "SeniorCitizen": st.selectbox("Senior citizen", [0, 1]),
        "Partner": st.selectbox("Partner", ["Yes", "No"]),
        "Dependents": st.selectbox("Dependents", ["Yes", "No"]),
        "tenure": st.slider("Tenure (months)", 0, 72, 12),
        "PhoneService": st.selectbox("Phone service", ["Yes", "No"]),
        "MultipleLines": st.selectbox("Multiple lines", ["No", "Yes", "No phone service"]),
        "InternetService": st.selectbox("Internet service", ["DSL", "Fiber optic", "No"]),
        "OnlineSecurity": st.selectbox("Online security", ["Yes", "No", "No internet service"]),
        "OnlineBackup": st.selectbox("Online backup", ["Yes", "No", "No internet service"]),
        "DeviceProtection": st.selectbox("Device protection", ["Yes", "No", "No internet service"]),
        "TechSupport": st.selectbox("Tech support", ["Yes", "No", "No internet service"]),
        "StreamingTV": st.selectbox("Streaming TV", ["Yes", "No", "No internet service"]),
        "StreamingMovies": st.selectbox("Streaming movies", ["Yes", "No", "No internet service"]),
        "Contract": st.selectbox("Contract", ["Month-to-month", "One year", "Two year"]),
        "PaperlessBilling": st.selectbox("Paperless billing", ["Yes", "No"]),
        "PaymentMethod": st.selectbox("Payment method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ]),
        "MonthlyCharges": st.number_input("Monthly charges", 0.0, 300.0, 70.0),
        "TotalCharges": st.number_input("Total charges", 0.0, 20000.0, 840.0),
    }

def risk_label(p):
    if p >= 0.65: return "High"
    if p >= 0.35: return "Medium"
    return "Low"
