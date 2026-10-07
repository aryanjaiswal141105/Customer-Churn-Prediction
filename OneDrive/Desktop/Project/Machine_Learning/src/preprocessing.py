# preprocessing.py: turns RAW customer rows into model-ready features.
# It repeats the steps of Phases 4 and 5, but works for any number of rows.
import numpy as np
import pandas as pd

from src import config


def clean_and_engineer(df_raw):
    """Clean, build the new features, and one-hot encode.
    Input: raw columns (no customerID, no Churn)."""
    df = df_raw.copy()  # never change the caller's data

    # Fix TotalCharges: text -> number, blanks -> 0
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    # Merge "No internet service" / "No phone service" into "No"
    for col in config.SERVICE_COLS:
        df[col] = df[col].replace({"No internet service": "No", "No phone service": "No"})

    # Yes/No -> 1/0, and gender -> 1/0
    for col in config.BINARY_COLS:
        df[col] = df[col].map({"Yes": 1, "No": 0})
    df["gender"] = df["gender"].map({"Male": 1, "Female": 0})

    # --- New features (Phase 5) ---
    df["num_addons"] = df[config.ADDON_COLS].sum(axis=1)
    df["has_family"] = ((df["Partner"] == 1) | (df["Dependents"] == 1)).astype(int)
    df["is_autopay"] = df["PaymentMethod"].str.contains("automatic").astype(int)

    # Average monthly bill so far (tenure 0 would divide by zero)
    tenure_safe = df["tenure"].replace(0, 1)
    df["avg_charge"] = np.where(df["tenure"] == 0,
                                df["MonthlyCharges"],
                                df["TotalCharges"] / tenure_safe)
    df["charge_diff"] = df["MonthlyCharges"] - df["avg_charge"]

    df["tenure_group"] = pd.cut(df["tenure"], bins=[-1, 12, 24, 48, 72],
                                labels=["0-12", "13-24", "25-48", "49-72"])

    # One-hot encode WITHOUT drop_first. We pick the right columns later
    # using feature_columns.pkl, so the "dropped" ones disappear there.
    df = pd.get_dummies(df, columns=config.MULTI_COLS, dtype=int)
    return df


def prepare_features(df_raw, feature_cols, scaler):
    """Raw customers -> exact numeric table the model expects."""
    df = clean_and_engineer(df_raw)

    # Keep only the model's columns, in the SAME order as training.
    # A dummy column that does not exist for this customer becomes 0.
    df = df.reindex(columns=feature_cols, fill_value=0).astype(float)

    # Scale the same columns the scaler was fitted on (learned from train only)
    scaled_cols = list(scaler.feature_names_in_)
    df[scaled_cols] = scaler.transform(df[scaled_cols])

    # Safety check: a missing value means an unexpected input
    if df.isnull().any().any():
        raise ValueError("Missing values after preprocessing. Check the input values.")
    return df