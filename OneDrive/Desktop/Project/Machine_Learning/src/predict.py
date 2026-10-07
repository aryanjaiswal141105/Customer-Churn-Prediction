# predict.py: load the saved model and score customers
import json
import joblib
import pandas as pd

from src import config
from src.preprocessing import prepare_features

# One example customer (raw format). New, month-to-month, fiber, e-check, no add-ons.
SAMPLE_CUSTOMER = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
    "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 85.0, "TotalCharges": 170.0,
}


def load_artifacts():
    """Load everything needed for prediction (model, columns, scaler, threshold)."""
    info = config.read_model_info()
    with open(config.METRICS_PATH) as f:
        metrics = json.load(f)
    return {
        "info": info,
        "model": joblib.load(config.MODEL_PATH),
        "feature_cols": joblib.load(config.FEATURE_COLUMNS_PATH),
        "scaler": joblib.load(config.scaler_path(info["feature_set"])),
        "threshold": float(metrics["threshold"]),  # chosen in Phase 7
    }


def risk_level(score, threshold):
    """High: at/above our threshold. Medium: at/above half of it. Otherwise Low."""
    if score >= threshold:
        return "High"
    if score >= threshold / 2:
        return "Medium"
    return "Low"


def predict_churn(df_raw, artifacts=None):
    """Score one or more RAW customers. Returns a small table of results."""
    a = artifacts or load_artifacts()
    X = prepare_features(df_raw, a["feature_cols"], a["scaler"])
    score = a["model"].predict_proba(X)[:, 1]  # column 1 = class "churn"
    return pd.DataFrame({
        "churn_probability": score,
        "churn_flag": (score >= a["threshold"]).astype(int),
        "risk": [risk_level(s, a["threshold"]) for s in score],
    })


# Runs only when you type: python -m src.predict
if __name__ == "__main__":
    result = predict_churn(pd.DataFrame([SAMPLE_CUSTOMER]))
    print(result)