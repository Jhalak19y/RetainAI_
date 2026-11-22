import json
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from app.common import get_model, get_data, get_explainer, make_customer_form, risk_label
from src.explain.shap_explainer import plain_language_reasons
from src.recommend.actions import recommend_actions
from src.db.database import save_prediction

st.title("Single Customer Predictor")
model = get_model()
df = get_data()
X = df.drop(columns="Churn")
form = st.form("customer_form")
with form:
    customer_values = make_customer_form()
    submitted = st.form_submit_button("Predict churn risk", type="primary")
if submitted:
    customer = pd.DataFrame([customer_values])[X.columns]
    probability = float(model.predict_proba(customer)[0,1])
    risk = risk_label(probability)
    st.subheader(f"{risk} risk")
    st.metric("Predicted churn probability", f"{probability:.1%}")
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=probability*100,
        title={"text":"Churn probability"},
        gauge={"axis":{"range":[0,100]},
               "steps":[{"range":[0,35],"color":"#d1fae5"},
                        {"range":[35,65],"color":"#fef3c7"},
                        {"range":[65,100],"color":"#fee2e2"}]}
    ))
    st.plotly_chart(fig, use_container_width=True)
    with st.spinner("Calculating SHAP explanation..."):
        engine = get_explainer(model, str(len(df)))
        contributions = engine.local_contributions(customer)
        reasons = plain_language_reasons(customer.iloc[0], contributions)
    st.subheader("Why this prediction?")
    st.dataframe(contributions.head(10), use_container_width=True, hide_index=True)
    for item in reasons[:6]:
        st.write(("🔺 " if item["direction"]=="increases" else "🔻 ") + item["text"])
    st.subheader("Suggested retention actions")
    st.dataframe(recommend_actions(customer.iloc[0], contributions), use_container_width=True)
    if st.session_state.get("username"):
        save_prediction(st.session_state.username, probability, risk, customer.to_json(orient="records"))
