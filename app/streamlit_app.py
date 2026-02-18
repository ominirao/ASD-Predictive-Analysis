import os
import json
import traceback
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="ASD Risk Dashboard", layout="wide")

st.title("🧠 Autism Spectrum Disorder (ASD) Risk Dashboard")
st.markdown("Data-driven behavioral screening assessment tool")

# File paths
MODEL_FILE = "models/asd_model_calibrated.joblib"
SCALER_FILE = "models/scaler.joblib"
META_FILE = "asd_metadata.json"

# Load artifacts
if not (os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE) and os.path.exists(META_FILE)):
    st.error("Required files missing.")
    st.stop()

model = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)
with open(META_FILE, "r") as f:
    metadata = json.load(f)

feature_cols = metadata["feature_cols"]
question_cols = metadata["question_order"]
age_col = metadata["age_col"]
gender_col = metadata["gender_col"]
jaundice_col = metadata["jaundice_col"]
family_col = metadata["family_col"]

# -----------------------------
# Input Section
# -----------------------------

st.sidebar.header("Screening Inputs")

questions_text = [
    "Eye contact",
    "Responds to name",
    "Points to show interest",
    "Enjoys social interaction",
    "Uses gestures",
    "Repetitive behaviors",
    "Difficulty with routine changes",
    "Intense focused interests",
    "Unusual sensory response",
    "Imaginative play"
]

answers = []
for i, q in enumerate(questions_text):
    ans = st.sidebar.radio(q, ["No", "Yes"], key=f"q{i}")
    answers.append(1 if ans == "Yes" else 0)

age = st.sidebar.number_input("Age", min_value=0, max_value=200, value=24)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
jaundice = st.sidebar.selectbox("Jaundice at birth?", ["No", "Yes"])
family = st.sidebar.selectbox("Family history of ASD?", ["No", "Yes"])

gender_val = 0 if gender == "Male" else 1
jaundice_val = 1 if jaundice == "Yes" else 0
family_val = 1 if family == "Yes" else 0

# -----------------------------
# Prediction
# -----------------------------

if st.sidebar.button("Run Assessment"):

    input_dict = {}
    for i, col in enumerate(question_cols[:10]):
        input_dict[col] = answers[i]

    if age_col:
        input_dict[age_col] = age
    if gender_col:
        input_dict[gender_col] = gender_val
    if jaundice_col:
        input_dict[jaundice_col] = jaundice_val
    if family_col:
        input_dict[family_col] = family_val

    input_vector = [input_dict.get(col, 0) for col in feature_cols]
    X = np.array(input_vector).reshape(1, -1)

    X_df = pd.DataFrame(X, columns=feature_cols)
    X_scaled = scaler.transform(X_df)

    probs = model.predict_proba(X_scaled)[0]
    classes = model.classes_

    asd_label = 0 if 0 in classes else 1
    asd_index = list(classes).index(asd_label)
    asd_prob = probs[asd_index]

    score = sum(answers)
    if score <= 3:
        severity = "Low"
        color = "green"
    elif score <= 6:
        severity = "Mild"
        color = "orange"
    elif score <= 8:
        severity = "Moderate"
        color = "darkorange"
    else:
        severity = "Severe"
        color = "red"

    # -----------------------------
    # Dashboard Layout
    # -----------------------------

    col1, col2 = st.columns(2)

    with col1:
        st.metric("ASD Probability", f"{asd_prob*100:.2f}%")

    with col2:
        st.markdown(f"### Severity Level")
        st.markdown(f"<h2 style='color:{color};'>{severity}</h2>", unsafe_allow_html=True)

    st.markdown("### Risk Distribution")

    risk_data = pd.DataFrame({
        "Category": ["ASD Risk", "No ASD Risk"],
        "Probability": [asd_prob, 1 - asd_prob]
    })

    st.bar_chart(risk_data.set_index("Category"))

    st.markdown("---")
    st.markdown("### Assessment Summary")
    st.write(f"The screening responses indicate a **{severity}** risk level based on behavioral indicators.")
    st.write("This output is for educational purposes only and does not constitute a medical diagnosis.")
