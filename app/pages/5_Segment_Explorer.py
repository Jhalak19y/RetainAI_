import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from app.common import get_data

st.title("Segment Explorer")
df = get_data().copy()
df["tenure_band"] = pd.cut(df.tenure, [-1,12,24,48,72],
                           labels=["0–12","13–24","25–48","49–72"])
dimension = st.selectbox("Explore segment by", ["Contract","InternetService","PaymentMethod","tenure_band"])
summary = df.groupby(dimension, observed=False).agg(
    customers=("Churn","size"), churn_rate=("Churn","mean"),
    avg_monthly_charges=("MonthlyCharges","mean")
).reset_index()
st.dataframe(summary, use_container_width=True, hide_index=True)
st.plotly_chart(px.bar(summary, x=dimension, y="churn_rate", color="customers",
                       title=f"Churn rate by {dimension}",
                       labels={"churn_rate":"Observed churn rate"}), use_container_width=True)
if st.checkbox("Explore optional K-Means clusters"):
    features = df[["tenure","MonthlyCharges","TotalCharges"]].fillna(
        df[["tenure","MonthlyCharges","TotalCharges"]].median()
    )
    scaled = StandardScaler().fit_transform(features)
    k = st.slider("Number of clusters", 2, 6, 3)
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(scaled)
    clustered = df.assign(cluster=labels)
    names = {}
    for cluster, group in clustered.groupby("cluster"):
        names[cluster] = (
            "Newer / lower tenure" if group.tenure.median() < 12
            else "Established / lower spend" if group.MonthlyCharges.median() < df.MonthlyCharges.median()
            else "Established / higher spend"
        )
    clustered["cluster_name"] = clustered.cluster.map(names)
    st.dataframe(clustered.groupby("cluster_name").agg(
        customers=("Churn","size"), churn_rate=("Churn","mean"),
        median_tenure=("tenure","median"),
        avg_monthly_charges=("MonthlyCharges","mean")
    ).reset_index(), use_container_width=True)
    st.caption("Cluster labels are heuristic descriptions derived from the displayed cluster medians.")
