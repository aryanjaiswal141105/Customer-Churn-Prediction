# train.py: rebuild the final model from the RAW csv in one command.
# It uses the same settings as your tuned model from Phase 6.
import json
import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src import config
from src.preprocessing import clean_and_engineer


def main():
    info = config.read_model_info()
    feature_cols = joblib.load(config.FEATURE_COLUMNS_PATH)
    scaled_cols = config.SCALED_COLS[info["feature_set"]]

    # clone() copies the tuned settings but gives an UNTRAINED model
    model = clone(joblib.load(config.MODEL_PATH))

    # Load raw data and split exactly like the notebooks
    df = pd.read_csv(config.RAW_DATA)
    y = (df["Churn"] == "Yes").astype(int)
    X_raw = df.drop(columns=["customerID", "Churn"])
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=config.TEST_SIZE, stratify=y,
        random_state=config.RANDOM_SEED)

    # Clean, build features, and keep the model's columns in the right order
    X_train = clean_and_engineer(X_train_raw).reindex(columns=feature_cols, fill_value=0).astype(float)
    X_test = clean_and_engineer(X_test_raw).reindex(columns=feature_cols, fill_value=0).astype(float)

    # Scale: fit on TRAIN only, then apply to test (no leakage)
    scaler = StandardScaler()
    X_train[scaled_cols] = scaler.fit_transform(X_train[scaled_cols])
    X_test[scaled_cols] = scaler.transform(X_test[scaled_cols])

    # Train and check on the test set
    model.fit(X_train, y_train)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    with open(config.METRICS_PATH) as f:
        saved_auc = json.load(f)["roc_auc"]
    print(f"Rebuilt test ROC-AUC: {auc:.4f} | Phase 7 test ROC-AUC: {saved_auc:.4f}")

    # Safety: only overwrite the saved files if the result is the same as before
    if abs(auc - saved_auc) > 0.005:
        print("WARNING: results differ from Phase 7. Nothing was saved.")
        return

    joblib.dump(model, config.MODEL_PATH)
    joblib.dump(scaler, config.scaler_path(info["feature_set"]))
    print("Saved model and scaler. Reproducibility confirmed.")


if __name__ == "__main__":
    main()