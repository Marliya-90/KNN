import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="📊 KNN Pass/Fail Prediction")
st.title("📊 KNN Student Pass/Fail Prediction")

# -----------------------------
# Load Dataset (relative path)
# -----------------------------
DATA_PATH = "PassFaillKNN.csv"  # make sure this file is in your repo

try:
    df = pd.read_csv(DATA_PATH)
    st.success("✅ Dataset loaded successfully!")
except FileNotFoundError:
    st.error("❌ Dataset not found in repo! Make sure PassFaillKNN.csv is pushed to GitHub.")
    st.stop()

# Show dataset preview
st.subheader("🔍 Dataset Preview")
st.dataframe(df.head())

# -----------------------------
# Load KNN model
# -----------------------------

