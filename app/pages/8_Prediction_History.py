import streamlit as st
import pandas as pd
from src.db.database import init_db, get_history, delete_prediction

st.title("Prediction History")
init_db()
username = st.session_state.get("username")
if not username:
    st.info("Sign in from the Account page to save and view personal prediction history.")
    st.stop()
rows = get_history(username)
if not rows:
    st.info("No saved predictions yet.")
    st.stop()
df = pd.DataFrame(rows)
risk = st.multiselect("Filter risk", ["High","Medium","Low"], default=["High","Medium","Low"])
df = df[df.risk.isin(risk)]
st.dataframe(df[["id","created_at","probability","risk"]], use_container_width=True)
selected = st.number_input("Prediction ID to delete", min_value=1, step=1)
if st.button("Delete selected prediction"):
    delete_prediction(username, int(selected))
    st.success("If that prediction belonged to your account, it has been deleted.")
    st.rerun()
