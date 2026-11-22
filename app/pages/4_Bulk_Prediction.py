import streamlit as st
import pandas as pd
from app.common import get_model, get_data, risk_label

st.title("Bulk Prediction")
st.write("Upload a CSV containing the customer feature columns used by the model. "
         "A customerID column is optional and retained in the output.")
reference = get_data()
required = [c for c in reference.columns if c != "Churn"]
uploaded = st.file_uploader("Upload customer CSV", type=["csv"])
if uploaded:
    try:
        batch = pd.read_csv(uploaded)
        if "Churn" in batch:
            batch = batch.drop(columns="Churn")
        missing = sorted(set(required)-set(batch.columns))
        extra = sorted(set(batch.columns)-set(required)-{"customerID"})
        if missing:
            st.error(f"Missing required columns: {', '.join(missing)}")
            st.stop()
        if extra:
            st.warning(f"Extra columns will be ignored: {', '.join(extra)}")
        model_input = batch[required].copy()
        for c in ["SeniorCitizen","tenure","MonthlyCharges","TotalCharges"]:
            model_input[c] = pd.to_numeric(model_input[c], errors="coerce")
        with st.spinner("Scoring uploaded customers..."):
            p = get_model().predict_proba(model_input)[:,1]
        output = batch.copy()
        output["churn_probability"] = p
        output["risk_band"] = [risk_label(v) for v in p]
        output = output.sort_values("churn_probability", ascending=False)
        st.success(f"Scored {len(output):,} customers.")
        risk_filter = st.multiselect("Risk bands", ["High","Medium","Low"],
                                     default=["High","Medium","Low"])
        st.dataframe(output[output.risk_band.isin(risk_filter)], use_container_width=True)
        st.download_button("Download scored CSV", output.to_csv(index=False).encode(),
                           "retainai_scored_customers.csv", "text/csv")
    except Exception as exc:
        st.error(f"Could not process this CSV: {exc}")
