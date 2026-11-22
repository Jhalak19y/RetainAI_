import streamlit as st
import pandas as pd
import plotly.express as px
from app.common import get_data, get_metrics

st.title("Business Dashboard")
df = get_data().copy()
metrics = get_metrics()

with st.sidebar:
    st.header("Segment filters")
    contract = st.multiselect("Contract", sorted(df.Contract.unique()), default=sorted(df.Contract.unique()))
    internet = st.multiselect("Internet service", sorted(df.InternetService.unique()), default=sorted(df.InternetService.unique()))
    payment = st.multiselect("Payment method", sorted(df.PaymentMethod.unique()), default=sorted(df.PaymentMethod.unique()))
    tenure_band = st.multiselect("Tenure band", ["0–12 months", "13–24 months", "25–48 months", "49–72 months"],
                                 default=["0–12 months", "13–24 months", "25–48 months", "49–72 months"])
df["Tenure band"] = pd.cut(df.tenure, [-1,12,24,48,72], labels=["0–12 months","13–24 months","25–48 months","49–72 months"])
df = df[df.Contract.isin(contract) & df.InternetService.isin(internet) &
        df.PaymentMethod.isin(payment) & df["Tenure band"].astype(str).isin(tenure_band)]
churn_rate = df.Churn.mean() if len(df) else 0
revenue_risk = df.loc[df.Churn.eq(1), "MonthlyCharges"].sum()
a,b,c = st.columns(3)
a.metric("Filtered customers", f"{len(df):,}")
b.metric("Observed churn rate", f"{churn_rate:.1%}")
c.metric("Monthly charges of observed churners", f"${revenue_risk:,.0f}")
st.caption("Revenue at risk is a proxy: sum of monthly charges for historical churners, not forecast revenue.")
left,right = st.columns(2)
with left:
    st.plotly_chart(px.bar(df.groupby("Contract", as_index=False).Churn.mean(),
                           x="Contract", y="Churn", title="Observed churn rate by contract",
                           labels={"Churn":"Churn rate"}), use_container_width=True)
with right:
    st.plotly_chart(px.bar(df.groupby("InternetService", as_index=False).Churn.mean(),
                           x="InternetService", y="Churn", title="Observed churn by internet service",
                           labels={"Churn":"Churn rate"}), use_container_width=True)
st.plotly_chart(px.histogram(df, x="tenure", color=df.Churn.map({0:"Stayed",1:"Churned"}),
                             barmode="overlay", title="Tenure distribution by outcome"),
                use_container_width=True)
