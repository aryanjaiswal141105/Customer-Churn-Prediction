# config.py: all paths and constants live here, so no other file hardcodes them
from pathlib import Path

# This file is src/config.py, so the project root is two levels up
ROOT = Path(__file__).resolve().parent.parent

# Data paths
RAW_DATA = ROOT / "data" / "raw" / "Telco_Customer_Churn.csv"
PROCESSED_DIR = ROOT / "data" / "processed"

# Model files saved in Phases 6 and 7
MODEL_PATH = ROOT / "models" / "best_model.pkl"
FEATURE_COLUMNS_PATH = ROOT / "models" / "feature_columns.pkl"
MODEL_INFO_PATH = ROOT / "models" / "model_info.txt"
METRICS_PATH = ROOT / "models" / "metrics.json"

# Same settings as the notebooks (important for reproducible results)
RANDOM_SEED = 42
TEST_SIZE = 0.2

# Column groups (same as Phases 4 and 5)
SERVICE_COLS = ["MultipleLines", "OnlineSecurity", "OnlineBackup",
                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
ADDON_COLS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
              "TechSupport", "StreamingTV", "StreamingMovies"]
BINARY_COLS = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"] + SERVICE_COLS
MULTI_COLS = ["InternetService", "Contract", "PaymentMethod", "tenure_group"]

# Columns that were scaled, for each feature set
SCALED_COLS = {
    "base": ["tenure", "MonthlyCharges", "TotalCharges"],
    "fe": ["tenure", "MonthlyCharges", "TotalCharges",
           "num_addons", "avg_charge", "charge_diff"],
}


def read_model_info():
    """Read model_info.txt (written in Phase 6) into a dictionary."""
    info = {}
    with open(MODEL_INFO_PATH) as f:
        for line in f.read().splitlines():
            if "=" in line:
                key, value = line.split("=")
                info[key] = value
    return info


def scaler_path(feature_set):
    """Which scaler file belongs to the chosen feature set."""
    name = "scaler_fe.pkl" if feature_set == "fe" else "scaler.pkl"
    return ROOT / "models" / name