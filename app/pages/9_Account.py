import streamlit as st
from src.db.database import init_db, register_user, authenticate

st.title("Account")
init_db()
if st.session_state.get("username"):
    st.success(f"Signed in as {st.session_state.username}")
    if st.button("Sign out"):
        st.session_state.pop("username", None)
        st.rerun()
else:
    tab1,tab2 = st.tabs(["Sign in","Create account"])
    with tab1:
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Sign in"):
            if authenticate(username,password):
                st.session_state.username = username.strip()
                st.success("Signed in.")
                st.rerun()
            else:
                st.error("Invalid username or password.")
    with tab2:
        username = st.text_input("Choose username", key="register_user")
        password = st.text_input("Choose password (8+ characters)", type="password", key="register_pass")
        if st.button("Create account"):
            try:
                register_user(username,password)
                st.success("Account created. Sign in now.")
            except ValueError as exc:
                st.error(str(exc))
st.caption("Local demo authentication uses bcrypt password hashes and SQLite. For production, use managed identity, HTTPS, secure cookie/session controls and secrets management.")
