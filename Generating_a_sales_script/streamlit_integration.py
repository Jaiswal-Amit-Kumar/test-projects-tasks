import streamlit as st
import requests

FASTAPI_URL = "http://localhost:8000/analyze"

st.title("Sales Script Analyzer")

uploaded_file = st.file_uploader("Upload a transcript file", type=["txt", "pdf"])

if uploaded_file:
    st.write(f"File name: {uploaded_file.name}")
    st.write(f"File type: {uploaded_file.type}")

    if st.button("Analyze"):
        try:
            response = requests.post(FASTAPI_URL, files={"file": uploaded_file})
            response.raise_for_status()  # Raise an exception for HTTP errors
            data = response.json()
            st.write("Greeting:", data["greeting"])
            st.write("Needs:", data["needs"])
            st.write("Pitch:", data["pitch"])
            st.write("Closing:", data["closing"])
            st.write("Explanation:", data["explanation"])
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")
        except ValueError:
            st.error("Failed to decode response. The backend may have returned an invalid JSON.")
