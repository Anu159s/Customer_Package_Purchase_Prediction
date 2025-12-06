import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="Anu159/customer_package_purchase_prediction_model", filename="best_customer_package_purchase_prediction_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("Customer Package Purchase Prediction App")
st.write("""
This application predicts whether a customer will purchase the newly introduced Wellness Tourism Package before contacting them based on different parameters.
Please enter the data below to get a prediction.
""")

# User input
TypeofContact = st.selectbox("Contact Type", ["Self Enquiry", "Company Invited"])
Age = st.number_input("Age", min_value=18.0, max_value=90.0, value=37.0, step=1)
CityTier = st.number_input("CityTier", min_value=1.0, max_value=3.0, value=3.0, step=1.0)
DurationOfPitch = st.number_input("DurationOfPitch", min_value=1, max_value=60, value=15)
Occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
Gender = st.selectbox("Gender", ["Female", "Male", "Fe Male"])

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'TypeofContact': typeofcontact,
    'Age': age,
    'CityTier': citytier,
    'DurationOfPitch': durationofpitch,
    'Occupation': occupation,
    'Gender': gender
  }])


if st.button("Predict Purchase"):
    prediction = model.predict(input_data)[0]
    result = "Package Purchase" if prediction == 1 else "No Purchase"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
