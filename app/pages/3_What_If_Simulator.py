import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from app.common import get_model, get_data, make_customer_form, risk_label

st.title("What-If Simulator")
st.write("Adjust a customer profile and compare modelled churn probabilities. "
         "This is a model simulation, not proof that a change causes retention.")
df = get_data()
X = df.drop(columns="Churn")
model = get_model()
base_index = st.selectbox("Choose a sample customer", range(min(100, len(X))), format_func=lambda i: f"Sample record {i+1}")
base = X.iloc[[base_index]].copy()
st.caption("Starting profile")
st.dataframe(base, use_container_width=True)
edited = base.copy()
col1,col2,col3 = st.columns(3)
with col1:
    edited.loc[:, "Contract"] = st.selectbox("Simulated contract", ["Month-to-month","One year","Two year"],
                                             index=["Month-to-month","One year","Two year"].index(str(base.iloc[0].Contract)))
    edited.loc[:, "TechSupport"] = st.selectbox("Tech support", ["Yes","No","No internet service"],
                                                index=["Yes","No","No internet service"].index(str(base.iloc[0].TechSupport)))
with col2:
    edited.loc[:, "OnlineSecurity"] = st.selectbox("Online security", ["Yes","No","No internet service"],
                                                   index=["Yes","No","No internet service"].index(str(base.iloc[0].OnlineSecurity)))
    edited.loc[:, "MonthlyCharges"] = st.slider("Monthly charges", 0.0, 200.0, float(base.iloc[0].MonthlyCharges))
with col3:
    edited.loc[:, "tenure"] = st.slider("Tenure", 0, 72, int(base.iloc[0].tenure))
    edited.loc[:, "PaymentMethod"] = st.selectbox("Payment method", [
        "Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"
    ], index=["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"].index(str(base.iloc[0].PaymentMethod)))
before = float(model.predict_proba(base)[0,1])
after = float(model.predict_proba(edited)[0,1])
a,b,c = st.columns(3)
a.metric("Original risk", f"{before:.1%}")
b.metric("Simulated risk", f"{after:.1%}")
c.metric("Difference", f"{after-before:+.1%}")
st.plotly_chart(go.Figure(go.Bar(x=["Original","Simulated"], y=[before,after],
                                 text=[f"{before:.1%}",f"{after:.1%}"],
                                 textposition="outside")).update_layout(
                                     title="Before / after churn probability",
                                     yaxis_title="Predicted probability",
                                     yaxis_range=[0,1]), use_container_width=True)
st.write("Original risk band:", risk_label(before), "| Simulated risk band:", risk_label(after))
