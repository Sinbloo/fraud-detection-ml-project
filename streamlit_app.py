import streamlit as st
import requests
import pickle
import os

st.set_page_config(page_title="Fraud Detection", page_icon="🚗")

st.title("🚗 Insurance Fraud Detection")

# load feature names
FEATURE_PATH = r"C:\Users\Softlaptop\OneDrive\Desktop\ML_project\model\feature_names.pkl"
feature_names = pickle.load(open(FEATURE_PATH, "rb"))

user_input = {}

st.subheader("Enter details:")

# generate inputs automatically
for feature in feature_names:
    user_input[feature] = st.text_input(feature)

if st.button("Predict"):
    try:
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=user_input
        )

        result = response.json()

        if "FRAUD" in result["result"]:
            st.error(result["result"])
        else:
            st.success(result["result"])

    except Exception as e:
        st.error(f"Error: {e}")