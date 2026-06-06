import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
model = joblib.load("model/churn_model.pkl")
st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide"
)

st.title("📊 Customer Churn Prediction Dashboard")

st.markdown(
    "Predict whether a telecom customer is likely to churn."
)
col1, col2 = st.columns(2)
with col1:

    tenure = st.slider(
        "Tenure (Months)",
        0, 72, 12
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        0.0,
        200.0,
        70.0
    )

    total_charges = st.number_input(
        "Total Charges",
        0.0,
        10000.0,
        1000.0
    )

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    partner = st.selectbox(
        "Partner",
        ["No", "Yes"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["No", "Yes"]
    )

with col2:
    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )
    internet = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )
    paperless = st.selectbox(
        "Paperless Billing",
        ["No", "Yes"]
    )
    payment = st.selectbox(
        "Payment Method",
        [
            "Bank transfer (automatic)",
            "Credit card (automatic)",
            "Electronic check",
            "Mailed check"
        ]
    )
data = {
    'SeniorCitizen': 0,
    'tenure': tenure,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    'gender_Male': 1 if gender == "Male" else 0,
    'Partner_Yes': 1 if partner == "Yes" else 0,
    'Dependents_Yes': 1 if dependents == "Yes" else 0,
    'PhoneService_Yes': 1,
    'MultipleLines_No phone service': 0,
    'MultipleLines_Yes': 0,
    'InternetService_Fiber optic': 1 if internet == "Fiber optic" else 0,
    'InternetService_No': 1 if internet == "No" else 0,
    'OnlineSecurity_No internet service': 0,
    'OnlineSecurity_Yes': 0,
    'OnlineBackup_No internet service': 0,
    'OnlineBackup_Yes': 0,
    'DeviceProtection_No internet service': 0,
    'DeviceProtection_Yes': 0,
    'TechSupport_No internet service': 0,
    'TechSupport_Yes': 0,
    'StreamingTV_No internet service': 0,
    'StreamingTV_Yes': 0,
    'StreamingMovies_No internet service': 0,
    'StreamingMovies_Yes': 0,
    'Contract_One year': 1 if contract == "One year" else 0,
    'Contract_Two year': 1 if contract == "Two year" else 0,
    'PaperlessBilling_Yes': 1 if paperless == "Yes" else 0,
    'PaymentMethod_Credit card (automatic)': 1 if payment == "Credit card (automatic)" else 0,
    'PaymentMethod_Electronic check': 1 if payment == "Electronic check" else 0,
    'PaymentMethod_Mailed check': 1 if payment == "Mailed check" else 0
    }
input_df = pd.DataFrame([data])

if st.button("Predict"):
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)
    confidence = probability.max() * 100
    if prediction[0] == 1:
        st.error(f"Customer likely to churn ({confidence:.2f}% confidence)")
    else:
        st.success(f"Customer likely to stay ({confidence:.2f}% confidence)")
st.subheader("Top Features Influencing Churn")
features = ["TotalCharges","tenure","MonthlyCharges","Fiber Optic","Electronic Check","Two Year Contract"]
scores = [18.97,17.57,17.24,3.61,3.53,3.04]
fig, ax = plt.subplots()
ax.bar(features, scores)
plt.xticks(rotation=45)
st.pyplot(fig)
st.subheader("Model Accuracy Comparison")
compare = pd.DataFrame({
    "Model": [ "Logistic Regression","Random Forest"],
    "Accuracy": [82.11,78.92]
    })
st.bar_chart(compare.set_index("Model"))