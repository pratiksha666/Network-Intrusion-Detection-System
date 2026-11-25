"""predict.py
Responsible only for LOADING saved models and PREDICTING on new traffic.
"""

import os

import joblib

from src.data_loader import DROP_COLUMNS, TARGET_COLUMN
from src.train_model import MODEL_DIR


def list_saved_models():
    """Names of the .joblib files inside the models folder."""
    if not os.path.isdir(MODEL_DIR):
        return []
    return sorted(f for f in os.listdir(MODEL_DIR) if f.endswith(".joblib"))


def load_saved_model(file_name):
    """Load a model bundle saved by train_model.save_model()."""
    path = os.path.join(MODEL_DIR, file_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


def predict_traffic(bundle, df):
    """Predict Normal / Attack for every row of df.
    Returns a copy of df with 'Prediction' (0/1) and 'Result' (text) columns."""
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()

    # The model needs exactly the same feature columns it was trained on
    needed = bundle["feature_columns"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"The CSV is missing required columns: {', '.join(missing[:10])}"
                         + (" ..." if len(missing) > 10 else ""))

    X = df[needed]
    predictions = bundle["pipeline"].predict(X)

    result = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
    if TARGET_COLUMN in result.columns:   # keep the real label (if any) for comparison
        result = result.rename(columns={TARGET_COLUMN: "actual_label"})
    result["Prediction"] = predictions
    result["Result"] = ["Attack" if p == 1 else "Normal" for p in predictions]
    return result
