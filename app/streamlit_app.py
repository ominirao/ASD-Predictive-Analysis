# app/streamlit_app.py
# Cleaned and corrected Streamlit app for ASD screening

import os
import json
import traceback
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --- page config ---
st.set_page_config(page_title="ASD Screening Tool", layout="centered")
st.title("Autism Spectrum Disorder (ASD) Screening")

# --- file paths (adjust if you use different names) ---
MODEL_FILE = "models/asd_model_calibrated.joblib"
SCALER_FILE = "models/scaler.joblib"
META_FILE = "asd_metadata.json"

# --- check files ---
if not (os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE) and os.path.exists(META_FILE)):
    st.error(
        "Required files missing. Please place model, scaler, and metadata in the repository "
        "and update the paths in this file if necessary."
    )
    st.stop()

# --- load artifacts ---
try:
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    with open(META_FILE, "r") as f:
        metadata = json.load(f)
except Exception:
    st.error("Failed to load model/scaler/metadata. See traceback in app logs.")
    st.text_area("Traceback", traceback.format_exc(), height=300)
    st.stop()

# --- metadata fields ---
feature_cols = metadata.get("feature_cols")
question_cols = metadata.get("question_order")
age_col = metadata.get("age_col")
gender_col = metadata.get("gender_col")
jaundice_col = metadata.get("jaundice_col")
family_col = metadata.get("family_col")

if not feature_cols or not question_cols:
    st.error("Metadata missing required fields: 'feature_cols' and/or 'question_order'.")
    st.stop()

# --- Questions (UI) ---
questions_text = [
    "1. Does the child make eye contact?",
    "2. Does the child respond when their name is called?",
    "3. Does the child point to show interest?",
    "4. Does the child enjoy social interaction?",
    "5. Does the child use gestures like waving?",
    "6. Does the child show repetitive behaviors?",
    "7. Does the child have difficulty with routine changes?",
    "8. Does the child show intense interests in specific topics?",
    "9. Does the child show unusual sensory responses?",
    "10. Does the child engage in imaginative play?"
]

st.markdown("### Screening Questions")
cols = st.columns(2)
answers = []
for i, q in enumerate(questions_text):
    with cols[i % 2]:
        a = st.radio(q, ["No", "Yes"], index=0, key=f"q{i}")
        answers.append(1 if a == "Yes" else 0)

st.markdown("---")
st.markdown("### Personal Information")
age = st.number_input("Age (same format as training dataset)", min_value=0, max_value=200, value=24)
gender = st.selectbox("Gender", ["Male", "Female"])
jaundice = st.selectbox("Jaundice at birth?", ["No", "Yes"])
family = st.selectbox("Family member with ASD?", ["No", "Yes"])

# numeric encodings (match how metadata/training used)
gender_val = 0 if gender == "Male" else 1
jaundice_val = 1 if jaundice == "Yes" else 0
family_val = 1 if family == "Yes" else 0

st.markdown("---")

# --- Predict button ---
if st.button("Assess ASD Risk"):

    # Build input dict based on metadata question order
    input_dict = {}
    for i, col in enumerate(question_cols[: len(questions_text)]):
        input_dict[col] = answers[i]

    # Add optional demographic fields if present in metadata
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

    # Determine expected features from scaler if available
    try:
        expected_n = getattr(scaler, "mean_", None).shape[0]
    except Exception:
        expected_n = None

    # Pad or truncate to match expected length
    if expected_n is not None and X.shape[1] != expected_n:
        st.warning(
            f"Input length ({X.shape[1]}) != model expected features ({expected_n}). "
            "Attempting to adjust automatically."
        )
        if X.shape[1] < expected_n:
            pad = expected_n - X.shape[1]
            X = np.hstack([X, np.zeros((1, pad))])
            st.info(f"Padded input with {pad} zeros.")
        else:
            trunc = X.shape[1] - expected_n
            X = X[:, :expected_n]
            st.info(f"Truncated input by removing last {trunc} features.")

    # Create DataFrame with feature names so scaler doesn't warn about feature names mismatch
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
            st.error("Error applying scaler to input. Feature types/order may be incompatible with training.")
            st.text_area("Traceback", traceback.format_exc(), height=300)
            st.stop()

    # Predict probabilities and classes if available
    probs = None
    classes = list(getattr(model, "classes_", []))
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(Xs)[0]
    except Exception:
        probs = None

    # Predict class
    try:
        pred = model.predict(Xs)[0]
    except Exception:
        st.error("Model predict failed.")
        st.text_area("Traceback", traceback.format_exc(), height=300)
        st.stop()

    # Decide which label corresponds to ASD (heuristic)
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

    # Compute ASD probability (safe fallback if probs is None)
    if probs is not None and asd_index is not None:
        asd_prob = float(probs[asd_index])
        if len(probs) == 2:
            other_idx = 1 - asd_index
            other_prob = float(probs[other_idx])
            st.write(f"ASD probability: **{asd_prob*100:.1f}%** (label={asd_label})")
            st.write(f"Non-ASD probability: **{other_prob*100:.1f}%** (label={classes[other_idx]})")
        else:
            st.write(f"ASD probability (label={asd_label}): **{asd_prob*100:.1f}%**")
    else:
        # fallback: treat predicted class as probability 1.0 for that label
        asd_prob = 1.0 if str(pred) == str(asd_label) else 0.0
        st.write("Model does not support probability output. Predicted class: " + str(pred))

    # --- Visualization for BI/Analyst presentation ---
    try:
        risk_data = pd.DataFrame({
            "Category": ["ASD Risk", "No ASD Risk"],
            "Probability": [asd_prob, max(0.0, 1 - asd_prob)]
        })
        st.markdown("#### Risk probability")
        st.bar_chart(risk_data.set_index("Category"))
    except Exception:
        st.info("Could not render risk chart (see logs).")

    # Severity heuristic based on the raw yes/no answers
    score = sum(answers)
    if score <= 3:
        severity = "Low"
    elif score <= 6:
        severity = "Mild"
    elif score <= 8:
        severity = "Moderate"
    else:
        severity = "Severe"

    # Show final result
    is_positive = (str(pred) == str(asd_label))
    if is_positive:
        st.error(f"Screening result: Positive for ASD traits — Severity: {severity}")
    else:
        st.success(f"Screening result: Negative for ASD traits — Severity: {severity}")

    st.caption(f"Model class labels (trained): {classes}. Interpreting ASD as label `{asd_label}` based on model classes.")
