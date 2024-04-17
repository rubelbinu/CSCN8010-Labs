import streamlit as st
import pandas as pd
import numpy as np
import joblib

import os

model_path = 'best_model_pro.pkl'
if os.path.exists(model_path):
    model = joblib.load(model_path)

# Function to preprocess input data
def preprocess_input(sex, chest_pain, resting_bp, cholesterol, fasting_bs, resting_ecg, max_hr, exercise_angina, old_peak, st_slope, n_major_vessels, thalium, age_group):
    # Convert categorical variables to numerical
    sex = 1 if sex == 'Male' else 0
    chest_pain_mapping = {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-anginal Pain': 2, 'Asymptomatic': 3}
    chest_pain = chest_pain_mapping[chest_pain]
    fasting_bs = 1 if fasting_bs == 'True' else 0
    resting_ecg_mapping = {'Normal': 0, 'ST-T wave abnormality': 1, 'Left ventricular hypertrophy': 2}
    resting_ecg = resting_ecg_mapping[resting_ecg]
    exercise_angina = 1 if exercise_angina == 'Yes' else 0
    st_slope_mapping = {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2}
    st_slope = st_slope_mapping[st_slope]
    thalium_mapping = {'Normal': 0, 'Fixed Defect': 1, 'Reversible Defect': 2}
    thalium = thalium_mapping[thalium]
    
    # Return preprocessed data as a numpy array
    return np.array([[sex, chest_pain, resting_bp, cholesterol, fasting_bs, resting_ecg, max_hr, exercise_angina, old_peak, st_slope, n_major_vessels, thalium, age_group]])

def main():
    st.title('Heart Disease Prediction')
    
    # Inputs
    sex = st.radio('Sex', ('Male', 'Female'))
    chest_pain = st.selectbox('Chest Pain Type', ('Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'))
    resting_bp = st.slider('Resting Blood Pressure', 90, 200, 160)
    cholesterol = st.slider('Cholesterol', 100, 600, 208)
    fasting_bs = st.radio('Fasting Blood Sugar > 120 mg/dl', ('True', 'False'))
    resting_ecg = st.selectbox('Resting ECG', ( 'ST-T wave abnormality','Normal', 'Left ventricular hypertrophy'))
    max_hr = st.slider('Max Heart Rate', 60, 220, 110)
    exercise_angina = st.radio('Exercise Induced Angina', ('Yes', 'No'))
    old_peak = st.slider('Old Peak', 0.0, 6.2, 1.70, step=0.1)
    st_slope = st.selectbox('ST Slope', ('Upsloping', 'Flat', 'Downsloping'))
    n_major_vessels = st.selectbox('Number of Major Vessels Colored by Fluoroscopy', (0, 1, 2, 3))
    thalium = st.selectbox('Thalium Stress Test Result', ('Normal', 'Fixed Defect', 'Reversible Defect'))
    age_group = st.number_input('Age Group', min_value=0, max_value=100, value=40)

    # Preprocess input data
    input_data = preprocess_input(sex, chest_pain, resting_bp, cholesterol, fasting_bs, resting_ecg, max_hr, exercise_angina, old_peak, st_slope, n_major_vessels, thalium, age_group)

    if st.button('Predict'):
        prediction = model.predict(input_data)

        if prediction[0] == 1:
            st.error('Risk of Heart Disease Detected')
        else:
            st.success('No Risk of Heart Disease Detected')

if __name__ == "__main__":
    main()