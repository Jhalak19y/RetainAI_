import streamlit as st
import plotly.express as px
from app.common import get_data, get_model, get_explainer

st.title("Explainability Studio")
df = get_data()
X = df.drop(columns="Churn")
model = get_model()
with st.spinner("Computing global SHAP values. This may take a few minutes..."):
    engine = get_explainer(model, str(len(df)))
    importance, explanation, sample = engine.global_importance(X, sample_size=120)
st.subheader("Global churn drivers")
st.caption("Mean absolute SHAP values summarize the average magnitude of each feature's contribution to modelled churn probability.")
st.plotly_chart(px.bar(
    importance.head(15).sort_values("mean_abs_shap"),
    x="mean_abs_shap", y="feature", orientation="h",
    title="Top global SHAP drivers",
    labels={"mean_abs_shap":"Mean absolute SHAP contribution","feature":"Customer feature"}
), use_container_width=True)
st.subheader("Individual customer explanation")
idx = st.selectbox("Select a customer row", range(len(X)))
customer = X.iloc[[idx]]
contrib = engine.local_contributions(customer)
st.metric("Modelled churn probability", f"{model.predict_proba(customer)[0,1]:.1%}")
st.plotly_chart(px.bar(
    contrib.head(12).sort_values("shap_value"),
    x="shap_value", y="feature", orientation="h",
    color="shap_value", color_continuous_midpoint=0,
    title="Local SHAP contributions",
    labels={"shap_value":"Contribution to churn probability","feature":"Feature"}
), use_container_width=True)
st.dataframe(contrib, use_container_width=True, hide_index=True)
