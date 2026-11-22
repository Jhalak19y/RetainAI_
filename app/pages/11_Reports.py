import streamlit as st
import pandas as pd
from app.common import get_data, get_model, get_explainer, risk_label
from src.explain.shap_explainer import plain_language_reasons
from src.recommend.actions import recommend_actions
from src.reports.pdf_report import customer_pdf, batch_pdf

st.title("PDF Reports")
df = get_data()
X = df.drop(columns="Churn")
model = get_model()
engine = get_explainer(model, str(len(df)))
mode = st.radio("Report type", ["Individual customer brief", "Batch summary"], horizontal=True)
if mode == "Individual customer brief":
    idx = st.number_input("Customer row index", min_value=0, max_value=len(X)-1, value=0)
    customer = X.iloc[[int(idx)]]
    probability = float(model.predict_proba(customer)[0,1])
    contributions = engine.local_contributions(customer)
    reasons = plain_language_reasons(customer.iloc[0], contributions)
    actions = recommend_actions(customer.iloc[0], contributions)
    st.metric("Predicted churn probability", f"{probability:.1%}")
    pdf = customer_pdf(customer.iloc[0].to_dict(), probability, risk_label(probability), reasons, actions)
    st.download_button("Download customer PDF", pdf, "retainai_customer_brief.pdf", "application/pdf")
else:
    probs = model.predict_proba(X)[:,1]
    bands = [risk_label(p) for p in probs]
    summary = pd.Series(bands).value_counts().reindex(["High","Medium","Low"], fill_value=0).to_dict()
    st.write(summary)
    pdf = batch_pdf(summary)
    st.download_button("Download batch summary PDF", pdf, "retainai_batch_summary.pdf", "application/pdf")
