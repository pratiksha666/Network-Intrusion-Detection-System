"""train_model.py
Responsible only for TRAINING and SAVING models.
"""

import os
import time

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.preprocessing import build_preprocessor

MODEL_DIR = "models"   # relative path (inside the project folder)
MODEL_NAMES = ["Random Forest", "KNN", "SVM", "Logistic Regression"]


def get_classifier(name):
    """Return a fresh, untrained classifier for the given name."""
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    if name == "KNN":
        return KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    if name == "SVM":
        return SVC(kernel="rbf", random_state=42)
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=1000, random_state=42)
    raise ValueError(f"Unknown model: {name}")


def train_model(name, X_train, y_train, svm_max_rows=20000):
    """Build Pipeline (preprocessing + classifier) and train it.
    Returns (trained_pipeline, training_time_in_seconds)."""
    # SVM is very slow on big data, so train it on a stratified sample
    if name == "SVM" and len(X_train) > svm_max_rows:
        X_train, _, y_train, _ = train_test_split(
            X_train, y_train, train_size=svm_max_rows, stratify=y_train, random_state=42
        )

    pipeline = Pipeline([
        ("preprocessor", build_preprocessor(X_train)),
        ("classifier", get_classifier(name)),
    ])

    start = time.time()
    pipeline.fit(X_train, y_train)
    return pipeline, round(time.time() - start, 2)


def model_path(name):
    """File path for a saved model, e.g. models/random_forest.joblib"""
    return os.path.join(MODEL_DIR, name.lower().replace(" ", "_") + ".joblib")


def save_model(pipeline, name, feature_columns):
    """Save the trained pipeline with Joblib (plus the feature names it expects)."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "model_name": name,
                 "feature_columns": list(feature_columns)}, model_path(name))
    return model_path(name)
