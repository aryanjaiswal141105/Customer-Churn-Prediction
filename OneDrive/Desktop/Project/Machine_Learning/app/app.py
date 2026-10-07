# app.py: the Streamlit web app.
# Run with:  streamlit run app/app.py   (from the project root)
import sys
from pathlib import Path

# Streamlit runs this file from the app/ folder, so we add the project root
# to Python's search path. Without this, "import src" fails.
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from src.predict import load_artifacts, predict_churn

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="wide")


# cache_resource: load the model once, not on every click
@st.cache_resource
def get_artifacts():
    return load_artifacts()


artifacts = get_artifacts()

st.title("📉 Customer Churn Predictor")
st.write("Enter a customer's details and click **Predict** to see their churn risk.")

col1, col2, col3 = st.columns(3)

# ---------- Column 1: customer ----------
with col1:
    st.subheader("Customer")
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior citizen", ["No", "Yes"])
    partner = st.selectbox("Has a partner", ["No", "Yes"])
    dependents = st.selectbox("Has dependents", ["No", "Yes"])

# ---------- Column 2: account ----------
with col2:
    st.subheader("Account")
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless billing", ["Yes", "No"])
    payment = st.selectbox("Payment method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly = st.number_input("Monthly charges ($)", min_value=18.0,
                              max_value=120.0, value=70.0, step=1.0)
    auto_total = st.checkbox("Estimate total charges as monthly x tenure", value=True)
    if auto_total:
        total = monthly * tenure
    else:
        total = st.number_input("Total charges ($)", min_value=0.0,
                                max_value=9000.0, value=0.0, step=10.0)

# ---------- Column 3: services ----------
with col3:
    st.subheader("Services")
    phone = st.selectbox("Phone service", ["Yes", "No"])
    multiple = st.selectbox("Multiple lines", ["No", "Yes"])
    internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
    security = st.selectbox("Online security", ["No", "Yes"])
    backup = st.selectbox("Online backup", ["No", "Yes"])
    protection = st.selectbox("Device protection", ["No", "Yes"])
    tech = st.selectbox("Tech support", ["No", "Yes"])
    tv = st.selectbox("Streaming TV", ["No", "Yes"])
    movies = st.selectbox("Streaming movies", ["No", "Yes"])
    st.caption("If there is no internet service, add-ons are treated as 'No'.")

# Without internet, add-ons cannot exist. Without phone, no multiple lines.
if internet == "No":
    security = backup = protection = tech = tv = movies = "No"
if phone == "No":
    multiple = "No"

# Build ONE raw customer, using the same column names as the original CSV
customer = {
    "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
    "Partner": partner, "Dependents": dependents, "tenure": tenure,
    "PhoneService": phone, "MultipleLines": multiple, "InternetService": internet,
    "OnlineSecurity": security, "OnlineBackup": backup, "DeviceProtection": protection,
    "TechSupport": tech, "StreamingTV": tv, "StreamingMovies": movies,
    "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
    "MonthlyCharges": monthly, "TotalCharges": total,
}

st.divider()

if st.button("Predict churn risk", type="primary"):
    result = predict_churn(pd.DataFrame([customer]), artifacts)
    score = float(result["churn_probability"].iloc[0])
    risk = result["risk"].iloc[0]

    c1, c2 = st.columns(2)
    c1.metric("Churn risk score", f"{score:.1%}")
    c2.metric("Risk level", risk)
    st.progress(min(max(score, 0.0), 1.0))

    if risk == "High":
        st.error("High risk: this customer is above our decision threshold.")
    elif risk == "Medium":
        st.warning("Medium risk: worth monitoring.")
    else:
        st.success("Low risk.")

    # Ideas to TEST (from your Phase 8 findings), not proven causes.
    # Edit these so they match YOUR SHAP results.
    ideas = []
    if contract == "Month-to-month":
        ideas.append("Offer an incentive to move to a one- or two-year contract.")
    if payment == "Electronic check":
        ideas.append("Encourage automatic payment (bank transfer or card).")
    if internet != "No" and tech == "No":
        ideas.append("Offer a free trial of tech support.")
    if tenure <= 12:
        ideas.append("Add onboarding calls during the first year.")
    if ideas:
        st.subheader("Retention ideas to test")
        for idea in ideas:
            st.write("- " + idea)

    st.caption(
        f"The score is a risk score, not an exact probability: the model was trained "
        f"with class weights, which push scores upward. A customer is flagged 'High' "
        f"at or above {artifacts['threshold']:.2f} (threshold chosen in Phase 7)."
    )