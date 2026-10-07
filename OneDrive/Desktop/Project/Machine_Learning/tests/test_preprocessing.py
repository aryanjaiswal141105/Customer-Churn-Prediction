# Tests: automatic checks that our pipeline is correct.
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config
from src.predict import SAMPLE_CUSTOMER, load_artifacts, predict_churn
from src.preprocessing import prepare_features


def test_pipeline_reproduces_saved_test_set():
    """Rebuild the test features from the RAW csv and compare with the notebook output."""
    a = load_artifacts()
    suffix = "_fe" if a["info"]["feature_set"] == "fe" else ""
    saved = pd.read_csv(config.PROCESSED_DIR / f"test{suffix}.csv")

    # Same split as the notebooks (same seed, same stratify)
    raw = pd.read_csv(config.RAW_DATA)
    y = (raw["Churn"] == "Yes").astype(int)
    X_raw = raw.drop(columns=["customerID", "Churn"])
    _, X_test_raw, _, _ = train_test_split(
        X_raw, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_SEED)

    rebuilt = prepare_features(X_test_raw, a["feature_cols"], a["scaler"])
    assert np.allclose(rebuilt.values, saved[a["feature_cols"]].values, atol=1e-6)


def test_single_customer_has_correct_columns():
    a = load_artifacts()
    X = prepare_features(pd.DataFrame([SAMPLE_CUSTOMER]), a["feature_cols"], a["scaler"])
    assert list(X.columns) == a["feature_cols"]      # same names, same order
    assert X.shape == (1, len(a["feature_cols"]))
    assert not X.isnull().any().any()                # no missing values


def test_score_is_between_0_and_1():
    score = predict_churn(pd.DataFrame([SAMPLE_CUSTOMER]))["churn_probability"].iloc[0]
    assert 0.0 <= score <= 1.0


def test_brand_new_customer_works():
    """tenure = 0 would divide by zero if we forgot to handle it."""
    customer = dict(SAMPLE_CUSTOMER, tenure=0, TotalCharges=0.0)
    result = predict_churn(pd.DataFrame([customer]))
    assert len(result) == 1