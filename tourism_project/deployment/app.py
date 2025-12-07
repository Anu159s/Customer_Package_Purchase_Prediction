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
This application predicts whether a customer will purchase the Wellness Tourism Package.
Please enter the data below to get a prediction.
""")

# User input

TypeofContact = st.selectbox("Contact Type", ["Self Enquiry", "Company Invited"])

Age = st.number_input(
    "Age", min_value=18.0, max_value=90.0, value=35.0, step=1.0
)

CityTier = st.selectbox("City Tier", [1, 2, 3])

DurationOfPitch = st.number_input(
    "Duration Of Pitch", min_value=1.0, max_value=60.0, value=15.0, step=1.0
)

NumberOfPersonVisiting = st.number_input(
    "Number of Persons Visiting", min_value=1.0, max_value=10.0, value=2.0, step=1.0
)

NumberOfFollowups = st.number_input(
    "Number of Followups", min_value=0.0, max_value=10.0, value=2.0, step=1.0
)

PreferredPropertyStar = st.selectbox(
    "Preferred Property Star", [1, 2, 3, 4, 5]
)

NumberOfTrips = st.number_input(
    "Number of Trips", min_value=0.0, max_value=50.0, value=5.0, step=1.0
)

PitchSatisfactionScore = st.selectbox(
    "Pitch Satisfaction Score", [1, 2, 3, 4, 5]
)

NumberOfChildrenVisiting = st.number_input(
    "Number of Children Visiting", min_value=0.0, max_value=10.0, value=0.0, step=1.0
)

MonthlyIncome = st.number_input(
    "Monthly Income", min_value=1000.0, max_value=1000000.0, value=25000.0, step=1000.0
)

Occupation = st.selectbox(
    "Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"]
)

Gender = st.selectbox("Gender", ["Female", "Male", "Fe Male"])

MaritalStatus = st.selectbox(
    "Marital Status", ["Single", "Married", "Divorced", "Unmarried"]
)

Designation = st.selectbox(
    "Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
)

ProductPitched = st.selectbox(
    "Product Pitched",
    ["Basic", "Standard", "Deluxe", "Super Deluxe", "Premium"]
)

Passport = st.selectbox("Has Passport?", ["0", "1"])
OwnCar = st.selectbox("Owns a Car?", ["0", "1"])


# Assemble input into DataFrame
input_data = pd.DataFrame([{
    "TypeofContact": TypeofContact,
    "Age": Age,
    "CityTier": CityTier,
    "DurationOfPitch": DurationOfPitch,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "NumberOfFollowups": NumberOfFollowups,
    "PreferredPropertyStar": PreferredPropertyStar,
    "NumberOfTrips": NumberOfTrips,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "MonthlyIncome": MonthlyIncome,
    "Occupation": Occupation,
    "Gender": Gender,
    "MaritalStatus": MaritalStatus,
    "Designation": Designation,
    "ProductPitched": ProductPitched,
    "Passport": Passport,
    "OwnCar": OwnCar
}])

if st.button("Predict Purchase"):
    prediction = model.predict(input_data)[0]
    result = "Package Purchase" if prediction == 1 else "No Purchase"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
