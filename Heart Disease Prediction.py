import streamlit as st
import numpy as np
import joblib

# Load the saved machine learning model
model = joblib.load('heart_disease_model.pkl')

# Define the Streamlit app
st.title("Heart Disease Prediction")
st.write("Enter the patient details to determine the likelihood of heart disease.")

# Create input fields for the patient details
sex = st.selectbox("Sex", [0, 1])
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting Electrocardiographic Results", [0, 1, 2])
exang = st.selectbox("Exercise Induced Angina", [0, 1])
slope = st.selectbox("Slope of the Peak Exercise ST Segment", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels Colored by Flourosopy", [0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia", [0, 1, 2, 3])
age = st.number_input("Age", min_value=0, max_value=100, value=50, step=1)
trestbps = st.number_input("Resting Blood Pressure", min_value=0, max_value=300, value=120, step=1)
chol = st.number_input("Serum Cholesterol", min_value=0, max_value=600, value=200, step=1)
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=0, max_value=300, value=150, step=1)

# Prepare the input data
X = np.array([[sex, cp, fbs, restecg, exang, slope, ca, thal, age, trestbps, chol, thalach]])

# Make the prediction using the saved model
prediction = model.predict(X)[0]

# Display the prediction result
if st.button("Predict"):
    if prediction == 0:
        st.write("The patient is likely not suffering from heart disease.")
    else:
        st.write("The patient is likely suffering from heart disease. Further tests or treatment may be recommended.")
