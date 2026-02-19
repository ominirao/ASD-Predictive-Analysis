# app/streamlit_app.py
# Dashboard with inputs in main area (not sidebar)

import os
import json
import traceback
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Page config
st.set_page_config(page_title="ASD Risk Dashboard", layout="wide")
st.title("🧠 Autism Spectrum Disorder (ASD) Risk Dashboard")
st.markdown("Data-driven behavioral screening assessment tool — inputs visible in main view")

# File paths (adjust if needed)
MODEL_FILE = "models/asd_model_calibrated.joblib"
SCALER_FILE = "models/scaler.joblib"
META_FILE = "asd_metadata.json"

# Ensure artifacts exist
if not (os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE) and os.path.exists(META_FILE)):
    st.error(
        "Required files missing. Please ensure model, scaler and metadata are present in the repo "
        "and the paths at the top of this file are correct."
    )
    st.stop()

# Load model, scaler, metadata
try:
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    with open(META_FILE, "r") as f:
        metadata = json.load(f)
except Exception:
    st.error("Failed to load model/scaler/metadata. See traceback below.")
    st.text_area("Traceback", traceback.format_exc(), height=300)
    st.stop()

# Metadata fields (must exist)
feature_cols = metadata.get("feature_cols")
question_cols = metadata.get("question_order")
age_col = metadata.get("age_col")
gender_col = metadata.get("gender_col")
jaundice_col = metadata.get("jaundice_col")
family_col = metadata.get("family_col")

if not feature_cols or not question_cols:
    st.error("Metadata missing required fields: 'feature_cols' and/or 'question_order'.")
    st.stop()

# Friendly questions text (short labels for UI)
questions_text = [
    "1. Eye contact",
    "2. Responds to name",
    "3. Points to show interest",
    "4. Enjoys social interaction",
    "5. Uses gestures",
    "6. Repetitive behaviors",
    "7. Difficulty with routine changes",
    "8. Intense focused interests",
    "9. Unusual sensory response",
    "10. Imaginative play"
]

# Layout: two main columns — left for inputs (always visible), right for KPIs + chart
left_col, right_col = st.columns([3, 2])

with left_col:
    st.header("Screening Questions (Main View)")
    # show questions in 2 columns inside left_col for compactness
    qcols = st.columns(2)
    answers = []
    for i, q in enumerate(questions_text):
        with qcols[i % 2]:
            resp = st.radio(q, ["No", "Yes"], index=0, key=f"q{i}")
            answers.append(1 if resp == "Yes" else 0)

    st.markdown("---")
    st.subheader("Personal Information")
    age = st.number_input("Age (months or years depending on dataset)", min_value=0, max_value=200, value=24)
    gender = st.selectbox("Gender", ["Male", "Female"], key="gender_input")
    jaundice = st.selectbox("Jaundice at birth?", ["No", "Yes"], key="jaundice_input")
    family = st.selectbox("Family history of ASD?", ["No", "Yes"], key="family_input")

    # numeric encodings consistent with metadata/training
    gender_val = 0 if gender == "Male" else 1
    jaundice_val = 1 if jaundice == "Yes" else 0
    family_val = 1 if family == "Yes" else 0

    st.markdown("---")
    # Run button in left column
    run_button = st.button("Run Assessment", key="run_assessment")

with right_col:
    # placeholders for KPIs and chart; will be updated after running
    kpi_prob = st.empty()
    kpi_sev = st.empty()
    st.markdown("### Risk Distribution")
    chart_placeholder = st.empty()
    st.markdown("---")
    st.markdown("### Quick notes")
    st.markdown("- Probabilities are calibrated and expressed as percentage.")
    st.markdown("- Severity is a simple heuristic based on number of 'Yes' responses.")

# If user pressed Run Assessment, compute prediction and update right column
if run_button:
    # Build input mapping from metadata order
    input_dict = {}
    for i, col in enumerate(question_cols[: len(questions_text)]):
        input_dict[col] = answers[i]

    # add demographic fields if present in metadata
    if age_col:
        input_dict[age_col] = age
    if gender_col:
        input_dict[gender_col] = gender_val
    if jaundice_col:
        input_dict[jaundice_col] = jaundice_val
    if family_col:
        input_dict[family_col] = family_val

    # Build ordered vector according to feature_cols
    try:
        input_vector = [input_dict.get(col, 0) for col in feature_cols]
    except Exception:
        st.error("Failed to construct input vector from metadata feature columns.")
        st.text_area("Traceback", traceback.format_exc(), height=300)
        st.stop()

    X = np.array(input_vector).reshape(1, -1)

    # Match scaler expectation and adjust if necessary
    try:
        expected_n = getattr(scaler, "mean_", None).shape[0]
    except Exception:
        expected_n = None

    if expected_n is not None and X.shape[1] != expected_n:
        st.warning(f"Input length ({X.shape[1]}) != expected features ({expected_n}). Auto-adjusting.")
        if X.shape[1] < expected_n:
            pad = expected_n - X.shape[1]
            X = np.hstack([X, np.zeros((1, pad))])
        else:
            X = X[:, :expected_n]

    # Create DataFrame with column names (use subset of feature_cols matching X columns)
    try:
        n_cols = X.shape[1]
        df_cols = feature_cols[:n_cols]
        X_df = pd.DataFrame(X, columns=df_cols)
    except Exception:
        st.error("Failed to create DataFrame with feature names for the scaler.")
        st.text_area("Traceback", traceback.format_exc(), height=300)
        st.stop()

    # Scale and predict (suppress sklearn warnings)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            Xs = scaler.transform(X_df)
        except Exception:
            st.error("Error applying scaler to input. Feature types/order may be incompatible.")
            st.text_area("Traceback", traceback.format_exc(), height=300)
            st.stop()

    # Predict
    try:
        classes = list(getattr(model, "classes_", []))
        probs = model.predict_proba(Xs)[0] if hasattr(model, "predict_proba") else None
        pred = model.predict(Xs)[0]
    except Exception:
        st.error("Model prediction failed.")
        st.text_area("Traceback", traceback.format_exc(), height=300)
        st.stop()

    # Determine ASD label index
    if 0 in classes:
        asd_label = 0
    elif 1 in classes:
        asd_label = 1
    else:
        asd_label = classes[0] if classes else 1

    try:
        asd_index = classes.index(asd_label) if classes else None
    except Exception:
        asd_index = None

    # Compute ASD probability with safe fallback
    if probs is not None and asd_index is not None:
        asd_prob = float(probs[asd_index])
    else:
        asd_prob = 1.0 if str(pred) == str(asd_label) else 0.0

    # Compute severity (heuristic)
    score = sum(answers)
    if score <= 3:
        severity = "Low"
        color = "#28a745"  # green
    elif score <= 6:
        severity = "Mild"
        color = "#ff8c00"  # orange
    elif score <= 8:
        severity = "Moderate"
        color = "#ff4500"  # dark orange/red
    else:
        severity = "Severe"
        color = "#dc3545"  # red

    # Update KPIs on right column
    kpi_prob.markdown(f"### ASD Probability\n**{asd_prob*100:.1f}%**")
    kpi_sev.markdown(f"### Severity\n<h3 style='color:{color};'>{severity}</h3>", unsafe_allow_html=True)

    # Build and render chart
    risk_data = pd.DataFrame({
        "Category": ["ASD Risk", "No ASD Risk"],
        "Probability": [asd_prob, max(0.0, 1 - asd_prob)]
    })
    chart_placeholder.bar_chart(risk_data.set_index("Category"))

    # Summary below columns (full-width)
    st.markdown("---")
    st.subheader("Assessment Summary")
    st.write(f"- **Predicted class:** {pred}")
    st.write(f"- **Interpreting ASD as label:** `{asd_label}` (model classes: {classes})")
    st.write(f"- **Calibrated ASD probability:** **{asd_prob*100:.1f}%**")
    st.write(f"- **Severity (heuristic):** **{severity}** based on {score} positive responses")
    st.info("This tool is for educational purposes only and is not a medical diagnosis.")

