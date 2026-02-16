# Autism Spectrum Disorder (ASD) Screening Web App

This project is a Machine Learning-based web application that predicts Autism Spectrum Disorder (ASD) risk using behavioral screening questions.

---

## 🚀 Project Overview

The model predicts ASD probability based on:

- 10 behavioral yes/no screening questions
- Age
- Gender
- Jaundice at birth
- Family history of ASD

The model was trained using:

- Random Forest Classifier
- SMOTE (class balancing)
- Probability calibration (Isotonic Regression)
- Feature scaling

The web interface is built using Streamlit.

---

## 📊 Model Features

- 80/20 stratified train-test split
- Balanced training dataset
- Calibrated probability outputs
- Clean web interface for real-time predictions

---

## 🖥️ How to Run Locally (Mac)

1. Open Terminal

2. Go to project folder:
cd /Users/ominirao/ASD_Project

3. Create virtual environment:
cd /Users/ominirao/ASD_Project

4. Install dependencies:
pip install -r requirements.txt

5. Run the app:
pip install -r requirements.txt

6. Open browser:
http://localhost:8501


---

## 📁 Project Structure
ASD_Project/
│
├── streamlit_app.py
├── asd_model_calibrated.joblib
├── scaler.joblib
├── asd_metadata.json
├── requirements.txt
├── README.md
└── .gitignore


---

## ⚠️ Disclaimer

This tool is for educational purposes only and is not a medical diagnostic tool.  
Please consult a healthcare professional for clinical diagnosis.

---

## 👤 Author

Omini Rao  
Machine Learning & Data Analytics

