import streamlit as st
 
# --- Authentication Function ---
def authenticate_user(username, password):
    valid_users = {"admin": "12345", "user": "password"}
    return valid_users.get(username) == password
 
 
# --- Forgot Password Function ---
def reset_password(username, new_password):
    """Simulates a password reset process"""
    if not username or not new_password:
        return False
    return True
 
 
# --- Main App Function ---
def main():
    # Sidebar navigation
    page = st.sidebar.selectbox("Navigation", ["Login", "Forgot Password"])
 
    if page == "Login":
        st.title("🔐 Login Page")
 
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_btn = st.button("Login")
 
        if login_btn:
            if authenticate_user(username, password):
                st.success("✅ Login successful!")
            elif username == "" or password == "":
                st.warning("⚠️ Please enter both username and password.")
            else:
                st.error("❌ Invalid credentials")
 
        st.markdown("---")
        st.markdown("[Forgot Password?](#)", unsafe_allow_html=True)
 
    elif page == "Forgot Password":
        st.title("🔄 Forgot Password")
        username = st.text_input("Enter your username")
        new_password = st.text_input("Enter new password", type="password")
        reset_btn = st.button("Reset Password")
 
        if reset_btn:
            if reset_password(username, new_password):
                st.success("✅ Password reset successful!")
            else:
                st.error("❌ Please fill in all fields")
 
if __name__ == "__main__":
    main()
