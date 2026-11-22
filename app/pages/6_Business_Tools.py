import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.metrics import confusion_matrix
from app.common import get_data, get_model, get_metrics
from src.business import roi_calculation

st.title("Business Decision Tools")
df = get_data()
X, y = df.drop(columns="Churn"), df.Churn
model = get_model()
probs = model.predict_proba(X)[:,1]
st.subheader("Decision threshold simulator")
threshold = st.slider("Churn alert threshold", 0.05, 0.95, 0.50, 0.01)
pred = (probs >= threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
a,b,c,d = st.columns(4)
a.metric("Churners caught", int(tp))
b.metric("Churners missed", int(fn))
c.metric("Offers to non-churners", int(fp))
d.metric("True negatives", int(tn))
st.caption("These counts are in-sample diagnostics on the supplied dataset, not independent test estimates.")
st.divider()
st.subheader("Retention ROI calculator")
st.caption("Every financial input below is a user-adjustable assumption, not a measured IBM dataset field.")
with st.form("roi"):
    population = st.number_input("Eligible customer population", min_value=1, value=len(df))
    top_pct = st.slider("Target top percentage by churn risk", 1, 100, 20)
    offer_cost = st.number_input("Offer cost per targeted customer ($)", min_value=0.0, value=20.0)
    lifetime_value = st.number_input("Customer lifetime value ($)", min_value=0.0, value=500.0)
    acceptance = st.slider("Offer acceptance rate (%)", 0, 100, 25) / 100
    save_rate = st.slider("Retention success among accepting customers (%)", 0, 100, 30) / 100
    submitted = st.form_submit_button("Calculate expected ROI")
if submitted:
    n = round(population * top_pct / 100)
    result = roi_calculation(n, population, offer_cost, lifetime_value, acceptance, save_rate)
    a,b,c = st.columns(3)
    a.metric("Expected retained customers", f"{result['expected_retained']:.1f}")
    b.metric("Offer spend", f"${result['offer_spend']:,.2f}")
    c.metric("Expected net value", f"${result['net_expected_value']:,.2f}")
    st.json(result)
