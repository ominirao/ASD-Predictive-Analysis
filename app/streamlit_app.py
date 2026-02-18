# streamlit_app.py
# Small bootstrap: if joblib missing in the environment, install it at runtime,
# then continue to import and run the app.
import sys
import subprocess
import importlib

# ---- Bootstrap to ensure joblib is available ----
try:
    import joblib
except Exception:
    try:
        # Try to install a known-compatible version
        subprocess.check_call([sys.executable, "-m", "pip", "install", "joblib==1.3.2"])
        joblib = importlib.import_module("joblib")
    except Exception as e:
        # If installation fails, log and raise so Streamlit shows error
        raise RuntimeError("Failed to install joblib at runtime: " + str(e))

# ---- Now normal imports ----
import streamlit as st
import numpy as np
import json
import os
import traceback
import pandas as pd  # for the chart

st.set_page_config(page_title="ASD Screening Tool", layout="centered")
st.title("Autism Spectrum Disorder (ASD) Screening")

# Filenames (must be in same folder or repo)
MODEL_FILE = "models/asd_model_calibrated.joblib"
SCALER_FILE = "models/scaler.joblib"
META_FILE = "asd_metadata.json"

# Check files exist
if not (os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE) and os.path.exists(META_FILE)):
    st.error("Required files missing. Please place model, scaler, and metadata in this folder (or update paths).")
    st.stop()

# Load artifacts
try:
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    with open(META_FILE, "r") as f:
        metadata = json.load(f)
except Exception:
    st.error("Failed to load model/scaler/metadata. See traceback in app logs.")
    st.caption(traceback.format_exc())
    st.stop()

# Extract feature order & question order from metadata
feature_cols = metadata.get("feature_cols", None)
question_cols = metadata.get("question_order", None)
age_col = metadata.get("age_col", None)
gender_col = metadata.get("gender_col", None)
jaundice_col = metadata.get("jaundice_col", None)
family_col = metadata.get("family_col", None)

if not feature_cols or not question_cols:
    st.error("Metadata missing required fields ('feature_cols' and 'question_order').")
    st.stop()

# Friendly question text (fixed)
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

# numeric encodings (match how we saved metadata/training)
gender_val = 0 if gender == "Male" else 1
jaundice_val = 1 if jaundice == "Yes" else 0
family_val = 1 if family == "Yes" else 0

st.markdown("---")

# Button to assess
if st.button("Assess ASD Risk"):

    # Build input dict based on metadata question order
    input_dict = {}

    # Map first 10 question items to metadata question columns (assumes question_cols order matches questions_text order)
    for i, col in enumerate(question_cols[:10]):
        input_dict[col] = answers[i]

    # Map extras if present in metadata
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
        st.stop()

    X = np.array(input_vector).reshape(1, -1)

    # Validate shape vs scaler expectation
    try:
        expected_n = getattr(scaler, "mean_", None).shape[0]
    except Exception:
        expected_n = None

    if expected_n is not None and X.shape[1] != expected_n:
        st.warning(f"Input length ({X.shape[1]}) != model expected features ({expected_n}). Attempting to adjust automatically.")
        if X.shape[1] < expected_n:
            # pad with zeros
            pad = expected_n - X.shape[1]
            X = np.hstack([X, np.zeros((1, pad))])
            st.info(f"Padded input with {pad} zeros.")
        else:
            # truncate
            trunc = X.shape[1] - expected_n
            X = X[:, :expected_n]
            st.info(f"Truncated input by removing last {trunc} features.")

    # Scale and predict
    try:
        Xs = scaler.transform(X)
    except Exception:
        st.error("Error applying scaler to input. Feature types/order may be incompatible with training.")
        st.caption(traceback.format_exc())
        st.stop()

    try:
        probs = model.predict_proba(Xs)[0]
        classes = list(getattr(model, "classes_", []))
    except Exception:
        # If predict_proba not available, fall back to predict
        probs = None
        classes = list(getattr(model, "classes_", []))

    # Determine which class index corresponds to ASD label.
    # Heuristic: if 0 present, treat 0 as ASD (fixes the inverted-label scenario); else if 1 present, use 1.
    asd_label = None
    if 0 in classes:
        asd_label = 0
    elif 1 in classes:
        asd_label = 1
    else:
        # fallback to first class (best-effort)
        asd_label = classes[0] if classes else 1

    # Find index of ASD label in classes
    try:
        asd_index = classes.index(asd_label)
    except Exception:
        asd_index = None

    # Predict class (for display)
    try:
        pred = model.predict(Xs)[0]
    except Exception:
        st.error("Model predict failed.")
        st.caption(traceback.format_exc())
        st.stop()

    # Compute ASD probability using asd_index if available, else show full probs
    # Also ensure asd_prob is set for chart even when predict_proba is not available
    if probs is not None and asd_index is not None:
        asd_prob = probs[asd_index]
        # show both probabilities if binary
        if len(probs) == 2:
            # other index
            other_idx = 1 - asd_index
            other_label = classes[other_idx]
            other_prob = probs[other_idx]
            st.write(f"ASD probability: **{asd_prob*100:.1f}%** (label={asd_label})")
            st.write(f"Non-ASD probability: **{other_prob*100:.1f}%** (label={other_label})")
        else:
            st.write(f"ASD probability (label={asd_label}): **{asd_prob*100:.1f}%**")
    else:
        # fallback: use predicted class as probability 100% for predicted label, 0% otherwise
        try:
            asd_prob = 1.0 if str(pred) == str(asd_label) else 0.0
        except Exception:
            asd_prob = 0.0
        st.write("Model does not support probability output. Predicted class: " + str(pred))

    # --- NEW: visualization for BI/analyst presentation ---
    try:
        risk_data = pd.DataFrame({
            "Category": ["ASD Risk", "No ASD Risk"],
            "Probability": [asd_prob, max(0.0, 1 - asd_prob)]
        })
        st.markdown("#### Risk probability")
        st.bar_chart(risk_data.set_index("Category"))
    except Exception:
        st.info("Could not render risk chart (see logs).")

    # Interpret prediction using ASD label
    is_positive = (str(pred) == str(asd_label))

    # Severity heuristic
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
    if is_positive:
        st.error(f"Screening result: Positive for ASD traits — Severity: {severity}")
    else:
        st.success(f"Screening result: Negative for ASD traits — Severity: {severity}")

    # Helpful note about class mapping
    st.caption(f"Model class labels (trained): {classes}. Interpreting ASD as label `{asd_label}` based on model classes.")
