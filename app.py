import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="📊 KNN Pass/Fail Prediction", layout="wide")
st.title("📊 KNN Student Pass/Fail Prediction")

# -----------------------------
# Load Dataset
# -----------------------------
try:
    df = pd.read_csv("PassFaillKNN.csv")
    st.success("✅ Dataset loaded successfully!")
except FileNotFoundError:
    st.error("❌ Dataset not found! Make sure PassFaillKNN.csv is in your repo.")
    st.stop()

# -----------------------------
# Load KNN model
# -----------------------------
try:
    knn = joblib.load("knn_model.pkl")
    st.success("✅ KNN model loaded successfully!")
except FileNotFoundError:
    st.error("❌ Model not found! Make sure knn_model.pkl is in your repo.")
    st.stop()

# -----------------------------
# Sidebar Input
# -----------------------------
st.sidebar.header("🧾 Student Study Time Input")
study_hours = st.sidebar.number_input(
    "Hours studied by student", min_value=0, max_value=10, value=3
)

# -----------------------------
# Prediction Button
# -----------------------------
if st.sidebar.button("🚀 Predict"):
    prediction = knn.predict([[study_hours]])
    
    # Colored cards
    if prediction[0] == 1:
        st.success(f"🎉 The student will PASS")
    else:
        st.error(f"❌ The student will FAIL")
    
    # -----------------------------
    # Scatter Plot
    # -----------------------------
    fig, ax = plt.subplots(figsize=(8,5))
    ax.scatter(df['studytime'], df['pass'], c=df['pass'], cmap='bwr', label='Existing Students', alpha=0.6)
    ax.scatter(study_hours, prediction[0], marker='X', s=200, color='green', label='New Student')
    ax.set_xlabel("Study Time (Hours)")
    ax.set_ylabel("Result (0 = Fail, 1 = Pass)")
    ax.set_title("KNN Student Pass/Fail Prediction")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Fail', 'Pass'])
    ax.legend()
    st.pyplot(fig)

# -----------------------------
# Dataset Preview & Stats
# -----------------------------
st.subheader("🔍 Dataset Preview")
st.dataframe(df.head())

st.subheader("📊 Dataset Stats")
total_students = len(df)
pass_count = df['pass'].sum()
fail_count = total_students - pass_count

st.write("Total students:", total_students)
st.write("Pass count:", pass_count)
st.write("Fail count:", fail_count)
