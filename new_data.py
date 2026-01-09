import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -----------------------------
# Load Saved Model & Artifacts
# -----------------------------
@st.cache_resource
def load_artifacts():
    with open("model_artifacts/car_sales_price_prediction_ann.pkl", "rb") as f:
        model = pickle.load(f)

    with open("model_artifacts/x_preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)

    with open("model_artifacts/feature_metadata.pkl", "rb") as f:
        feature_metadata = pickle.load(f)

    return model, preprocessor, feature_metadata


model, preprocessor, feature_metadata = load_artifacts()

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Car Price Prediction", layout="centered")

st.title("🚗 Car Sales Price Prediction")
st.write("Predict car purchase amount using AI (ANN)")

st.divider()

# -----------------------------
# User Inputs
# -----------------------------
country = st.selectbox("Country", ["USA", "Canada", "India", "Germany", "UK"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=18, max_value=100, value=35)
annual_salary = st.number_input("Annual Salary", min_value=0.0, value=60000.0)
credit_card_debt = st.number_input("Credit Card Debt", min_value=0.0, value=10000.0)
net_worth = st.number_input("Net Worth", min_value=0.0, value=200000.0)

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("Predict Car Price 🚀"):
    epsilon = 1e-5

    # Feature Engineering (MUST match training)
    data = pd.DataFrame({
        "country": [country],
        "gender": [gender],
        "age": [age],
        "annual Salary": [annual_salary],
        "credit card debt": [credit_card_debt],
        "net worth": [net_worth]
    })

    data["age_squared"] = data["age"] ** 2
    data["debt_to_income"] = data["credit card debt"] / (data["annual Salary"] + epsilon)
    data["networth_to_income"] = data["net worth"] / (data["annual Salary"] + epsilon)
    data["disposable_income"] = data["annual Salary"] - data["credit card debt"]
    data["salary_x_age"] = data["annual Salary"] * data["age"]
    data["log_salary"] = np.log1p(data["annual Salary"])
    data["log_net_worth"] = np.log1p(data["net worth"])
    data["log_debt"] = np.log1p(data["credit card debt"])

    data["age_group"] = pd.cut(
        data["age"],
        bins=[18, 25, 35, 45, 60, 100],
        labels=["18-25", "26-35", "36-45", "46-60", "60+"]
    )

    # Preprocess
    X_processed = preprocessor.transform(data)

    # Predict
    prediction = model.predict(X_processed)

    st.success(f"💰 Predicted Car Purchase Amount: ${prediction[0][0]:,.2f}")
