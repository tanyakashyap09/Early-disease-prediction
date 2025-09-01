import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ======================
# Load dataset
# ======================
# ⚠️ Place "diabetes.csv" in the same folder as this script
df = pd.read_csv("C:\\Users\\tanya\\Desktop\\disease prediction\\archive\\diabetes.csv")

# Split data
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model (Logistic Regression)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ======================
# Streamlit UI
# ======================
st.title("🩺 Early Disease Prediction - Diabetes")
st.write("Enter patient details below to check diabetes risk:")

# Factor explanations with CALCULATIONS
with st.expander("ℹ️ Explanation and Calculation of Factors"):
    st.markdown("""
    - **Pregnancies**: Number of times the patient has been pregnant. *(Recorded directly from patient history.)*  
    - **Glucose Level**: Blood sugar concentration measured in mg/dL.  
      - 📌 *Measured by a blood glucose test after fasting or random testing.*  
    - **Blood Pressure (BP)**: Diastolic blood pressure in mmHg.  
      - 📌 *Measured using a sphygmomanometer (BP machine).*  
    - **Skin Thickness**: Triceps skin fold thickness in mm.  
      - 📌 *Measured with skinfold calipers to estimate body fat.*  
    - **Insulin**: Serum insulin level (mu U/ml).  
      - 📌 *Measured using a blood test (fasting insulin test).*  
    - **BMI (Body Mass Index)**: Weight-to-height ratio.  
      - 📌 *Formula: **BMI = Weight (kg) / [Height (m)]²***  
      - Example: 70 kg person with height 1.75 m → BMI = 70 / (1.75²) = 22.9  
    - **Diabetes Pedigree Function (DPF)**: A score based on family history of diabetes.  
      - 📌 *Calculated using a formula that considers relatives with diabetes, their age, and relationship. In dataset it is pre-computed.*  
    - **Age**: Age of the patient in years. *(Recorded directly.)*  
    """)

# User inputs
preg = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=100)
bp = st.number_input("Blood Pressure", min_value=0, max_value=200, value=80)
skin = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
age = st.number_input("Age", min_value=1, max_value=120, value=30)

# Prepare data for prediction
user_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
user_data_scaled = scaler.transform(user_data)

# Prediction
if st.button("Predict"):
    prediction = model.predict(user_data_scaled)
    if prediction[0] == 1:
        st.error("⚠️ The patient is likely to have Diabetes.")
    else:
        st.success("✅ The patient is not likely to have Diabetes.")
