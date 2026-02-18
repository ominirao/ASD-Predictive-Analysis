# streamlit_app.py
# Robust bootstrap: attempt to ensure requirements are installed at runtime,
# capture pip output for debugging, and only proceed if installation succeeds.

import sys
import subprocess
import importlib
import os
import pathlib
import traceback

# Helper to run a command and capture output
def run_cmd(cmd):
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        return proc.returncode, proc.stdout + "\n" + proc.stderr
    except Exception as e:
        return 999, f"Exception running command {cmd}: {e}\n{traceback.format_exc()}"

# Try importing joblib; if it fails, attempt to find requirements.txt and pip install it.
needs_install = False
try:
    import joblib  # noqa: E402
except Exception:
    needs_install = True

pip_output = ""
installed_ok = False

if needs_install:
    # Find repository root by walking up from this file until we find requirements.txt (limit depth)
    this_file = pathlib.Path(__file__).resolve()
    repo_root = None
    cur = this_file.parent
    for _ in range(6):  # go up up to 6 levels
        candidate = cur / "requirements.txt"
        if candidate.exists():
            repo_root = cur
            req_path = str(candidate)
            break
        cur = cur.parent
    # If not found, also check the common mount path used in Streamlit builds
    if repo_root is None:
        alt = pathlib.Path("/mount/src")
        if alt.exists():
            # try to find requirements somewhere under /mount/src by name
            for p in alt.rglob("requirements.txt"):
                repo_root = p.parent
                req_path = str(p)
                break

    # If we found a requirements.txt, try to pip install it. Otherwise install joblib only.
    if repo_root:
        pip_output += f"Found requirements.txt at: {req_path}\nAttempting to pip install -r {req_path}\n\n"
        # Upgrade pip first (sometimes necessary)
        rc, out = run_cmd([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        pip_output += f"Upgrade pip rc={rc}\n{out}\n"
        # Install requirements (no cache to reduce disk)
        rc, out = run_cmd([sys.executable, "-m", "pip", "install", "--no-cache-dir", "-r", req_path])
        pip_output += f"Install reqs rc={rc}\n{out}\n"
        installed_ok = (rc == 0)
    else:
        pip_output += "No requirements.txt found in repository tree. Attempting to pip install joblib directly.\n\n"
        rc, out = run_cmd([sys.executable, "-m", "pip", "install", "--no-cache-dir", "joblib==1.3.2"])
        pip_output += f"Install joblib rc={rc}\n{out}\n"
        installed_ok = (rc == 0)

    # Try to import joblib now if pip reported success
    if installed_ok:
        try:
            importlib.invalidate_caches()
            import joblib  # noqa: E402
            installed_ok = True
        except Exception as e:
            pip_output += f"Import after install failed: {e}\n{traceback.format_exc()}\n"
            installed_ok = False

# Now start the Streamlit app (or show pip output and stop if install failed)
import streamlit as st  # noqa: E402
st.set_page_config(page_title="ASD Screening Tool", layout="centered")
st.title("Autism Spectrum Disorder (ASD) Screening")

if needs_install and not installed_ok:
    st.error("Required Python packages were not available and automatic installation FAILED.")
    st.markdown("### Pip output (for debugging)")
    st.text_area("pip output", pip_output, height=320)
    st.markdown(
        """
        **Next steps (copy/paste the pip output above if asking for help):**
        - Ensure `requirements.txt` exists at the repository root and contains `joblib`.
        - Try adding a pinned `joblib==1.3.2` line.
        - If `pip install` failed due to network or build errors, paste the above output here and I'll diagnose.
        """
    )
    st.stop()

# ---- normal app imports now that joblib should be present ----
import numpy as np  # noqa: E402
import json  # noqa: E402
import traceback as tb  # noqa: E402

# ---- app code below (same logic as before) ----

# Filenames (adjust paths if your models are in models/ folder)
MODEL_FILE = "asd_model_calibrated.joblib"
SCALER_FILE = "scaler.joblib"
META_FILE = "asd_metadata.json"

# If your model files are in a subfolder named models/, uncomment these:
if not os.path.exists(MODEL_FILE) and os.path.exists("models/" + MODEL_FILE):
    MODEL_FILE = "models/" + MODEL_FILE
if not os.path.exists(SCALER_FILE) and os.path.exists("models/" + SCALER_FILE):
    SCALER_FILE = "models/" + SCALER_FILE

# Check files exist
if not (os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE) and os.path.exists(META_FILE)):
    st.error("Required files missing. Make sure model, scaler, and metadata files are present in the repo and paths are correct.")
    st.markdown(f"Searching paths found MODEL_FILE={MODEL_FILE}, SCALER_FILE={SCALER_FILE}, META_FILE={META_FILE}")
    st.stop()

# Load artifacts
try:
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    with open(META_FILE, "r") as f:
        metadata = json.load(f)
except Exception as e:
    st.error("Failed to load model/scaler/metadata. See traceback below.")
    st.text_area("Traceback", tb.format_exc(), height=300)
    st.stop()

# Extract metadata fields
feature_cols = metadata.get("feature_cols")
question_cols = metadata.get("question_order")
age_col = metadata.get("age_col")
gender_col = metadata.get("gender_col")
jaundice_col = metadata.get("jaundice_col")
family_col = metadata.get("family_col")

if not feature_cols or not question_cols:
    st.error("Metadata is missing required fields 'feature_cols' or 'question_order'.")
    st.stop()

# UI (questions)
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

gender_val = 0 if gender == "Male" else 1
jaundice_val = 1 if jaundice == "Yes" else 0
family_val = 1 if family == "Yes" else 0

st.markdown("---")

if st.button("Assess ASD Risk"):
    # Build input dict
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

    try:
        expected_n = getattr(scaler, "mean_", None).shape[0]
    except Exception:
        expected_n = None

    if expected_n is not None and X.shape[1] != expected_n:
        st.warning(f"Input length ({X.shape[1]}) != model expected features ({expected_n}). Attempting to adjust automatically.")
        if X.shape[1] < expected_n:
            pad = expected_n - X.shape[1]
            X = np.hstack([X, np.zeros((1, pad))])
            st.info(f"Padded input with {pad} zeros.")
        else:
            trunc = X.shape[1] - expected_n
            X = X[:, :expected_n]
            st.info(f"Truncated input by removing last {trunc} features.")

    try:
        Xs = scaler.transform(X)
    except Exception:
        st.error("Error applying scaler. See traceback.")
        st.text_area("Traceback", tb.format_exc(), height=300)
        st.stop()

    try:
        probs = model.predict_proba(Xs)[0]
        classes = list(getattr(model, "classes_", []))
    except Exception:
        probs = None
        classes = list(getattr(model, "classes_", []))

    # Determine ASD label
    asd_label = 0 if 0 in classes else (1 if 1 in classes else (classes[0] if classes else 1))
    try:
        asd_index = classes.index(asd_label)
    except Exception:
        asd_index = None

    try:
        pred = model.predict(Xs)[0]
    except Exception:
        st.error("Model predict failed.")
        st.text_area("Traceback", tb.format_exc(), height=300)
        st.stop()

    if probs is not None and asd_index is not None:
        asd_prob = probs[asd_index]
        if len(probs) == 2:
            other_idx = 1 - asd_index
            other_label = classes[other_idx]
            other_prob = probs[other_idx]
            st.write(f"ASD probability: **{asd_prob*100:.1f}%** (label={asd_label})")
            st.write(f"Non-ASD probability: **{other_prob*100:.1f}%** (label={other_label})")
        else:
            st.write(f"ASD probability (label={asd_label}): **{asd_prob*100:.1f}%**")
    else:
        st.write("Model does not support probability output. Predicted class:", str(pred))

    is_positive = (str(pred) == str(asd_label))

    score = sum(answers)
    if score <= 3:
        severity = "Low"
    elif score <= 6:
        severity = "Mild"
    elif score <= 8:
        severity = "Moderate"
    else:
        severity = "Severe"

    if is_positive:
        st.error(f"Screening result: Positive for ASD traits — Severity: {severity}")
    else:
        st.success(f"Screening result: Negative for ASD traits — Severity: {severity}")

    st.caption(f"Model class labels (trained): {classes}. Interpreting ASD as label `{asd_label}` based on model classes.")
