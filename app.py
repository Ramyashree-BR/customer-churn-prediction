import streamlit as st
import pandas as pd
import joblib

# Load trained model and feature columns
model = joblib.load("best_churn_model.pkl")
model_features = joblib.load("model_features.pkl")

# Load original dataset for median Total Charges
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

df["Total Charges"] = pd.to_numeric(
    df["Total Charges"],
    errors="coerce"
)

median_total_charges = df["Total Charges"].median()


# Page title
st.title("Customer Churn Prediction System")
st.write("Predict whether a customer is likely to churn.")

st.divider()

# Customer information
st.subheader("Customer Information")

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

senior_citizen = st.selectbox(
    "Senior Citizen",
    [0, 1]
)

partner = st.selectbox(
    "Partner",
    ["Yes", "No"]
)

dependents = st.selectbox(
    "Dependents",
    ["Yes", "No"]
)

tenure = st.number_input(
    "Tenure Months",
    min_value=0,
    max_value=100,
    value=12
)

phone_service = st.selectbox(
    "Phone Service",
    ["Yes", "No"]
)

multiple_lines = st.selectbox(
    "Multiple Lines",
    ["Yes", "No", "No phone service"]
)

internet_service = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

online_security = st.selectbox(
    "Online Security",
    ["Yes", "No", "No internet service"]
)

online_backup = st.selectbox(
    "Online Backup",
    ["Yes", "No", "No internet service"]
)

device_protection = st.selectbox(
    "Device Protection",
    ["Yes", "No", "No internet service"]
)

tech_support = st.selectbox(
    "Tech Support",
    ["Yes", "No", "No internet service"]
)

streaming_tv = st.selectbox(
    "Streaming TV",
    ["Yes", "No", "No internet service"]
)

streaming_movies = st.selectbox(
    "Streaming Movies",
    ["Yes", "No", "No internet service"]
)

contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

paperless_billing = st.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0
)

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=monthly_charges * tenure
)


# Prediction
if st.button("Predict Churn"):

    customer = {
        "Gender": gender,
        "Senior Citizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "Tenure Months": tenure,
        "Phone Service": phone_service,
        "Multiple Lines": multiple_lines,
        "Internet Service": internet_service,
        "Online Security": online_security,
        "Online Backup": online_backup,
        "Device Protection": device_protection,
        "Tech Support": tech_support,
        "Streaming TV": streaming_tv,
        "Streaming Movies": streaming_movies,
        "Contract": contract,
        "Paperless Billing": paperless_billing,
        "Payment Method": payment_method,
        "Monthly Charges": monthly_charges,
        "Total Charges": total_charges
    }

    input_df = pd.DataFrame([customer])

    input_df["Total Charges"] = pd.to_numeric(
        input_df["Total Charges"],
        errors="coerce"
    )

    input_df["Total Charges"] = input_df["Total Charges"].fillna(
        median_total_charges
    )

    # Encode categorical variables
    input_df = pd.get_dummies(
        input_df,
        drop_first=True
    )

    # Match training features
    input_df = input_df.reindex(
        columns=model_features,
        fill_value=0
    )

    # Prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.divider()

    if prediction == 1:
        st.error("⚠️ Customer is likely to churn")
    else:
        st.success("✅ Customer is unlikely to churn")

    st.write(
        f"**Churn Probability: {probability * 100:.2f}%**"
    )
