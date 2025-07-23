
import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("salary_model_xgboost.pkl")

# Input fields
st.title("Employee Salary Prediction (XGBOOST)")
experience = st.slider("Experience (Years)", 0, 20, 1)
education = st.selectbox("Education Level", ["Bachelors", "Masters", "PhD"])
job_role = st.selectbox("Job Role", ["Developer", "Manager", "Analyst"])

# Encode inputs
edu_map = {"Bachelors": 0, "Masters": 1, "PhD": 2}
role_map = {"Developer": 1, "Manager": 2, "Analyst": 0}
education_encoded = edu_map[education]
job_role_encoded = role_map[job_role]

# Predict
if st.button("Predict Salary"):
    input_df = pd.DataFrame([[experience, education_encoded, job_role_encoded]],
                            columns=["Experience", "Education", "Job Role"])
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Salary: ₹{int(prediction):,}")
