import streamlit as st
import requests

# — Change this if your API is hosted elsewhere —
API_URL = "http://localhost:8000/chat"

st.title("🚗 AI Car Sales Assistant")

# --- Sidebar for selecting test customer ---
st.sidebar.header("Test Customer")
customer_id = st.sidebar.selectbox("Customer ID", ["cust123", "cust456", "unknown"])

# --- Main interaction ---
st.write("### Talk to the assistant")
message = st.text_input("Your message", placeholder="I want a Bentley")

if st.button("Send"):
    with st.spinner("Waiting for reply..."):
        try:
            resp = requests.post(API_URL, json={
                "customer_id": customer_id,
                "message": message
            })
            if resp.status_code == 200:
                st.success("Assistant replied:")
                st.write(resp.json()["reply"])
            else:
                st.error(f"Error {resp.status_code}: {resp.json().get('detail')}")
        except Exception as e:
            st.error(f"Could not connect: {e}")
