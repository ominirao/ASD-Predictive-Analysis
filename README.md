# 🧠 Autism Spectrum Disorder (ASD) Screening Web App

🔗 **Live Demo:** https://asd-predictive-analysis-ufhqomajchndsug9yuveyr.streamlit.app                         
💻 **GitHub Repository:** https://github.com/ominirao/ASD-Predictive-Analysis  

---

## 📌 Project Overview

This project is a Machine Learning-based web application that predicts Autism Spectrum Disorder (ASD) risk using behavioral screening indicators.

The model estimates ASD probability based on:

- 10 behavioral yes/no screening questions
- Age
- Gender
- Jaundice at birth
- Family history of ASD

The application provides:

- Calibrated probability output
- Severity estimation
- Real-time interactive predictions
- Public cloud deployment

---

## 🎯 Business Impact

This project demonstrates how behavioral screening data can be transformed into actionable risk insights using data analytics and predictive modeling.

Key outcomes:

- Cleaned and transformed raw behavioral and demographic data
- Identified key risk indicators using feature analysis
- Built a calibrated probability model for risk scoring
- Translated technical outputs into user-friendly severity categories
- Deployed an interactive dashboard for real-time decision support

This project showcases an end-to-end data workflow:                                                               
Data Collection → Data Cleaning → Feature Engineering → Model Development → Evaluation → Deployment.

---

## ⚙️ Machine Learning Pipeline

The model was built using:

- Random Forest Classifier
- SMOTE (Synthetic Minority Oversampling Technique)
- Probability Calibration (Isotonic Regression)
- Feature Scaling
- Stratified 80/20 Train-Test Split

---

## 📊 Model Performance

- Accuracy: 94.32%
- Precision: 94.11%
- Recall: 84.21%
- F1 Score: 88.88%
- ROC-AUC: 0.97

*(Metrics obtained from validation dataset — see training notebook for full evaluation.)*

---

## 🌍 Deployment

The application is deployed publicly using **Streamlit Cloud**.

To access the live application:

👉 **Click the Live Demo link above**

---

## 🖥️ Running Locally

If you would like to run the application locally:

1️⃣ Clone the repository:                                      
git clone https://github.com/ominirao/asd-predictive-analysis.git                        
cd asd-predictive-analysis

2️⃣ Create virtual environment:                           
python3 -m venv venv                           
source venv/bin/activate

3️⃣ Install dependencies:                            
pip install -r requirements.txt

4️⃣ Run the app:                                           
streamlit run ASD_Project/app/streamlit_app.py

The application will then be available locally.

---

## 📁 Project Structure

ASD-Predictive-Analysis/                                     
│                                                   
├── app/                                                           
│ └── streamlit_app.py                                                 
├── models/                                                                     
│ ├── asd_model_calibrated.joblib                                             
│ └── scaler.joblib                                                     
├── asd_metadata.json                                                          
├── requirements.txt                                                            
├── README.md                                                            
└── notebooks/                                                             
│ └── train_asd_colab.ipynb                                                      

---

## ⚠️ Disclaimer

This tool is intended for educational and research purposes only.  
It is **not a medical diagnostic tool**.  
For professional diagnosis, please consult a qualified healthcare provider.

---

## 👤 Author

Omini Rao  
Machine Learning | Data Analytics | Business Intelligence
