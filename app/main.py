import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from app.common import get_data, get_model, get_metrics
from src.db.database import init_db

st.set_page_config(page_title="RetainAI | Churn Intelligence", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")
init_db()

st.title("RetainAI")
st.caption("Customer churn prediction & retention intelligence")

data = None
try:
    data = get_data()
except Exception as exc:
    st.error(str(exc))
    st.info("Add the IBM Telco CSV to the data folder, then refresh this page.")
    st.stop()

model = get_model()
if model is None:
    st.warning("The model has not been trained yet.")
    st.code("python -m src.train.train_model")
    st.stop()

metrics = get_metrics()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers in dataset", f"{len(data):,}")
c2.metric("Observed churn rate", f"{data.Churn.mean():.1%}")
if metrics:
    c3.metric("Test ROC-AUC", f"{metrics['test_metrics']['roc_auc']:.3f}")
    c4.metric("Test churn recall", f"{metrics['test_metrics']['recall']:.3f}")
st.divider()
st.subheader("Start exploring")
st.write("Use the pages in the sidebar to investigate churn segments, score customers, "
         "simulate changes, upload batches and estimate retention economics.")
st.info("Predictions are model estimates, not certainty. Financial outputs use explicit assumptions.")
